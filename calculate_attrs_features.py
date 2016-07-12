# -*- encoding: utf-8 -*-

import os
import re

import cPickle

from time import time
from functools import partial
from operator import itemgetter

import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

import avito_utils

from prepare_attrs_tfidf import unwrap_dict, identity

def calc_jaccard(set1, set2, lam=0):
    union = len(set1 | set2)
    if not union: 
        return 0.0

    inter = len(set1 & set2)
    return inter * 1.0 / (union + lam)

norm = Normalizer()

def compute_vector_features(name, series1, series2, tfidf, svd, results):
    X1 = tfidf.transform(series1)
    X2 = tfidf.transform(series2)

    dot = X1.multiply(X2).sum(axis=1)
    results['attrs_%s_dot_tfidf' % name] = np.asarray(dot).reshape(-1)

    dot_bin = (X1 > 0).multiply(X2 > 0).sum(axis=1)
    results['attrs_%s_num_match' % name] = np.asarray(dot_bin).reshape(-1)

    X1_norm = norm.transform(X1)
    X2_norm = norm.transform(X2)

    cosine = X1_norm.multiply(X2_norm).sum(axis=1)
    results['attrs_%s_cosine' % name] = np.asarray(cosine).reshape(-1)

    sets_1 = series1.apply(set)
    sets_2 = series2.apply(set)

    jaccard_reg = partial(calc_jaccard, lam=1)
    results['attrs_%s_jaccard_reg' % name] = sets_1.combine(sets_2, jaccard_reg)
    results['attrs_%s_jaccard_full' % name] = sets_1.combine(sets_2, calc_jaccard)

    X1_svd = svd.transform(X1)
    X2_svd = svd.transform(X2)

    results['attrs_%s_dot_tfidf_svd' % name] = (X1_svd * X2_svd).sum(axis=1)

    X_diff = X1_svd - X2_svd
    results['attrs_%s_euclidean_tfidf_svd' % name] = (X_diff ** 2).sum(axis=1)
    results['attrs_%s_manhattan_tfidf_svd' % name] = np.abs(X_diff).sum(axis=1)

    X1_svd = norm.transform(X1_svd)
    X2_svd = norm.transform(X2_svd)
    results['attrs_%s_cosine_tfidf_svd' % name] = (X1_svd * X2_svd).sum(axis=1)


def process_batch(batch):    
    batch.reset_index(drop=True, inplace=1)

    item_ids = sorted(set(batch.itemID_1) | set(batch.itemID_2))

    df_attrs = avito_utils.get_item_infos(item_ids, columns=['attrsJSON'])
    tuples = df_attrs.attrsJSON.apply(unwrap_dict)
    df_attrs['pairs'] = tuples.apply(itemgetter(0))
    df_attrs['values'] = tuples.apply(itemgetter(1))

    item1 = df_attrs.loc[batch.itemID_1].reset_index(drop=True)
    item1.columns = [col + '_1' for col in item1.columns]
    item2 = df_attrs.loc[batch.itemID_2].reset_index(drop=True)
    item2.columns = [col + '_2' for col in item2.columns]
    batch = pd.concat([batch, item1, item2], axis=1)

    result = pd.DataFrame()

    compute_vector_features('pairs', batch.pairs_1, batch.pairs_2, tfidf, svd, result)
    compute_vector_features('values', batch.values_1, batch.values_2, tfidf_val, svd_val, result)

    return result


def prepare_batches(df, n):
    for batch_no, i in enumerate(range(0, len(df), n)):
        yield batch_no, df.iloc[i:i+n]


def append_to_csv(batch, csv_file):
    props = dict(encoding='utf-8', index=False)
    if not os.path.exists(csv_file):
        batch.to_csv(csv_file, **props)
    else:
        batch.to_csv(csv_file, mode='a', header=False, **props)      

def delete_file_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename)
        
if __name__ == "__main__":
    batch_size = 20000

    name = 'attrs'

    t0 = time()
    print 'loading tfidf and svd models...'
    with open('tfidf_svd_attrs_pairs.bin', 'rb') as f:
        tfidf, svd = cPickle.load(f)
    with open('tfidf_svd_attrs_vals.bin', 'rb') as f:
        tfidf_val, svd_val = cPickle.load(f)
    print 'done in %0.5fs' % (time() - t0)

    t0 = time()
    df = pd.read_csv('../input/ItemPairs_train.csv')
    delete_file_if_exists('features_%s_train.csv' % name)
    print 'read train set, start processing...'
    for batch_no, batch in tqdm(list(prepare_batches(df, batch_size))):
        batch = process_batch(batch)
        append_to_csv(batch, 'features_%s_train.csv' % name)
    print 'processing train set took %0.5fs' % (time() - t0)

    t0 = time()
    df =  pd.read_csv('../input/ItemPairs_test.csv')
    delete_file_if_exists('features_%s_test.csv' % name)
    print 'read test set, start processing...'
    for batch_no, batch in tqdm(list(prepare_batches(df, batch_size))):
        batch = process_batch(batch)
        append_to_csv(batch, 'features_%s_test.csv' % name)
    print 'processing test set took %0.5fs' % (time() - t0)
