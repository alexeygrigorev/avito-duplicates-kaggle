# -*- encoding: utf-8 -*-

import os
import codecs 
import cPickle

from functools import partial
from operator import itemgetter
from time import time

import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.preprocessing import normalize

import avito_utils

import warnings
warnings.filterwarnings("ignore")

mongo = avito_utils.mongo

def identity(l): return l

def compose2(f, g):
    return lambda x: g(f(x))

stopwords = set(u'в,на,и,с,по,для,не,от,за,к,до,из,или,у,один,вы,при,так,ваш,как,а,наш,быть,под,б,'
                u'р,мы,эт,же,также,что,это,раз,свой,он,если,но,я,о,ещё,тот,этот,то,они,ни,чем,где,'
                u'бы,оно,там,ж,она,ты'.split(','))

def load_corrections():
    corrections = {}

    with codecs.open('replacements.txt', 'r', 'utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            correct, others = line.split('=')
            for other in others.split(','):
                if not other:
                    continue
                corrections[other] = correct
    return corrections

def clean_tokens_list(tokens_list):
    result = []
    for sentence in tokens_list:
        for t in sentence.split():
            if t not in stopwords:
                result.append(corrections.get(t, t))
    return result

models_paths = {
    'title_count': 'text_models/title_count_(min_df=5).bin',
    'title_count_svd': 'text_models/title_count_svd_50.bin',
    'title_tfidf': 'text_models/title_tfidf_(norm=None,smooth_idf=True,sublinear_tf=False).bin',
    'title_tfidf_svd': 'text_models/title_tfidf_svd_50.bin',
    'title_bm25': 'text_models/title_bm25_(k1=2,b=075).bin',
    'title_bm25_svd': 'text_models/title_bm25_svd_50.bin',
    
    'desc_count': 'text_models/desc_count_(min_df=5).bin',
    'desc_count_svd': 'text_models/desc_count_svd_100.bin',
    'desc_tfidf': 'text_models/desc_tfidf_(norm=None,smooth_idf=True,sublinear_tf=False).bin',
    'desc_tfidf_svd': 'text_models/desc_tfidf_svd_100.bin',
    'desc_bm25': 'text_models/desc_bm25_(k1=2,b=075).bin',
    'desc_bm25_svd': 'text_models/desc_bm25_svd_100.bin',
    
    'all_count': 'text_models/all_count_(min_df=5).bin',
    'all_count_svd': 'text_models/all_count_svd_130.bin',
    'all_tfidf': 'text_models/all_tfidf_(norm=None,smooth_idf=True,sublinear_tf=False).bin',
    'all_tfidf_svd': 'text_models/all_tfidf_svd_130.bin',
    'all_bm25': 'text_models/all_bm25_(k1=2,b=075).bin',
    'all_bm25_svd': 'text_models/all_bm25_svd_130.bin',
}

def load_models():
    t0 = time()

    models = {}
    for name, path in models_paths.items():
        print 'loading %s from %s' % (name, path)
        with open(path, 'rb') as f:
            models[name] = cPickle.load(f)

    print 'loading models done in %0.5fs' % (time() - t0)
    return models

def calc_jaccard(set1, set2, lam=0):
    union = len(set1 | set2)
    if not union: 
        return 0.0

    inter = len(set1 & set2)
    return inter * 1.0 / (union + lam)

def intersect_len(set1, set2):
    return len(set1 & set2)

jaccard_reg = partial(calc_jaccard, lam=1)


def compute_jaccard_features(name, series1, series2, results):
    sets_1 = series1.apply(set)
    sets_2 = series2.apply(set)

    results['text_%s_jaccard_reg' % name] = sets_1.combine(sets_2, jaccard_reg)
    results['text_%s_jaccard_full' % name] = sets_1.combine(sets_2, calc_jaccard)
    results['text_%s_intersect' % name] = sets_1.combine(sets_2, intersect_len)

def compute_vector_features(name, series1, series2, tfidf_transform, svd, top_n_diff, results):
    X1 = tfidf_transform(series1)
    X2 = tfidf_transform(series2)

    dot = X1.multiply(X2).sum(axis=1)
    results['text_%s_dot' % name] = np.asarray(dot).reshape(-1)

    X1_norm = normalize(X1)
    X2_norm = normalize(X2)

    cosine = X1_norm.multiply(X2_norm).sum(axis=1)
    results['text_%s_cosine' % name] = np.asarray(cosine).reshape(-1)

    X1_svd = svd.transform(X1)
    X2_svd = svd.transform(X2)

    results['text_%s_dot_svd' % name] = (X1_svd * X2_svd).sum(axis=1)

    X_diff = X1_svd - X2_svd
    results['text_%s_euclidean_svd' % name] = (X_diff ** 2).sum(axis=1)
    results['text_%s_manhattan_svd' % name] = np.abs(X_diff).sum(axis=1)

    X1_svd = normalize(X1_svd)
    X2_svd = normalize(X2_svd)
    results['text_%s_cosine_svd' % name] = (X1_svd * X2_svd).sum(axis=1)
    
    for i in range(top_n_diff):
        results['text_%s_svd_diff_%d' % (name, i)] = np.abs(X_diff[:, i])


columns = ['title_lemmatized', 'description_lemmatized']

def process_batch(batch):    
    batch.reset_index(drop=True, inplace=1)

    item_ids = sorted(set(batch.itemID_1) | set(batch.itemID_2))

    df_items = mongo.get_df_by_ids(avito_utils.item_text, item_ids, columns=columns)
    df_items.rename(columns={'description_lemmatized': 'description', 
                             'title_lemmatized': 'title'}, inplace=1)

    df_items.description = df_items.description.apply(clean_tokens_list)
    df_items.title = df_items.title.apply(clean_tokens_list)
    df_items['all_text'] = df_items.title + df_items.description

    item1 = df_items.loc[batch.itemID_1].reset_index(drop=True)
    item1.columns = [col + '_1' for col in item1.columns]
    item2 = df_items.loc[batch.itemID_2].reset_index(drop=True)
    item2.columns = [col + '_2' for col in item2.columns]
    batch2 = pd.concat([batch, item1, item2], axis=1)

    results = pd.DataFrame(index=batch2.index)

    compute_jaccard_features('title', batch2.title_1, batch2.title_2, results)
    compute_jaccard_features('desc', batch2.description_1, batch2.description_2, results)
    compute_jaccard_features('all', batch2.all_text_1, batch2.all_text_2, results)

    compute_vector_features('title_count', batch2.title_1, batch2.title_2, 
                            models['title_count'].transform, models['title_count_svd'], 3, results)
    compute_vector_features('title_tfidf', batch2.title_1, batch2.title_2, 
                            compose2(models['title_count'].transform, models['title_tfidf'].transform), 
                            models['title_tfidf_svd'], 3, results)
    compute_vector_features('title_bm25', batch2.title_1, batch2.title_2, 
                            compose2(models['title_count'].transform, models['title_bm25'].transform), 
                            models['title_bm25_svd'], 3, results)

    compute_vector_features('desc_count', batch2.description_1, batch2.description_2, 
                            models['desc_count'].transform, models['desc_count_svd'], 6, results)
    compute_vector_features('desc_tfidf', batch2.description_1, batch2.description_2, 
                            compose2(models['desc_count'].transform, models['desc_tfidf'].transform), 
                            models['desc_tfidf_svd'], 6, results)
    compute_vector_features('desc_bm25', batch2.description_1, batch2.description_2, 
                            compose2(models['desc_count'].transform, models['desc_bm25'].transform), 
                            models['desc_bm25_svd'], 6, results)

    compute_vector_features('all_count', batch2.all_text_1, batch2.all_text_2, 
                            models['all_count'].transform, models['all_count_svd'], 10, results)
    compute_vector_features('all_tfidf', batch2.all_text_1, batch2.all_text_2, 
                            compose2(models['all_count'].transform, models['all_tfidf'].transform), 
                            models['all_tfidf_svd'], 10, results)
    compute_vector_features('all_bm25', batch2.all_text_1, batch2.all_text_2, 
                            compose2(models['all_count'].transform, models['all_bm25'].transform), 
                            models['all_bm25_svd'], 10, results)

    return results


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

    corrections = load_corrections()
    models = load_models()

    name = 'text'

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
