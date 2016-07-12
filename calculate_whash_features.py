# -*- encoding: utf-8 -*-

# wavelet features are taken from 
# https://www.kaggle.com/c/avito-duplicate-ads-detection/forums/t/22011/precomputed-wavelet-image-hashes


from time import time
import pickle, cPickle
import os

from tqdm import tqdm

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from collections import defaultdict

import avito_utils
mongo = avito_utils.mongo


def get_pickled_whashes(): 
    t0 = time()
    print 'reading whashes...'
    with open('whash_haar_py2.pkl', 'rb') as f:
        whash = pickle.load(f)
        print 'done in %0.5fs' % (time() - t0)
        return whash

def get_pickled_signatures(): 
    t0 = time()
    print 'reading image signatures...'
    with open('image-signatures.bin', 'rb') as f:
        res = cPickle.load(f)
        print 'done in %0.5fs' % (time() - t0)
        return res

def prepare_batches(df, n):
    for batch_no, i in enumerate(range(0, len(df), n)):
        yield batch_no, df.iloc[i:i+n]

def int_to_str(h):
    return np.binary_repr(h).rjust(64, '0')

def hamming_distance(h1, h2):
    h1 = int_to_str(h1)
    h2 = int_to_str(h2)
    return sum(1 for (i1, i2) in zip(h1, h2) if i1 == i2)


whash = None
signatures = None

def calc_features(images1, images2):
    if not images1 or not images2:
        return {}

    global whash, signatures

    sig_1 = {i: signatures[i] for i in images1 if i in signatures}
    sig_2 = {i: signatures[i] for i in images2 if i in signatures}
    common = set(sig_1.values()) & set(sig_2.values())

    hashes_1 = [whash[i] for i in images1 if i in whash and sig_1[i] not in common]
    hashes_2 = [whash[i] for i in images2 if i in whash and sig_2[i] not in common]    
    
    if len(hashes_1) == 0 or len(hashes_2) == 0:
        return {}

    result = {}

    values = []
    for h1 in hashes_1:
        for h2 in hashes_2:
            d = hamming_distance(h1, h2)
            values.append(d)

    result['whash_hamming_min'] = np.min(values)
    result['whash_hamming_mean'] = np.mean(values)
    result['whash_hamming_max'] = np.max(values)
    result['whash_hamming_std'] = np.std(values)
    result['whash_hamming_25p'] = np.percentile(values, q=25)
    result['whash_hamming_75p'] = np.percentile(values, q=75)
    result['whash_hamming_skew'] = skew(values)
    result['whash_hamming_kurtosis'] = kurtosis(values)

    return result


def batch_features(i1, i2):
    i1 = map(int, i1)
    i2 = map(int, i2)
    return calc_features(i1, i2)

def process_batch(batch, pool):    
    batch = batch.reset_index(drop=True)

    item_ids = sorted(set(batch.itemID_1) | set(batch.itemID_2))
    batch_images = mongo.get_df_by_ids(avito_utils.item_info, item_ids, columns=['images_array'])
    images_1 = batch_images.loc[batch.itemID_1].images_array.reset_index(drop=True)
    images_2 = batch_images.loc[batch.itemID_2].images_array.reset_index(drop=True)

    result = pool.process_parallel(batch_features, collections=(images_1, images_2))
    result = pd.DataFrame(result)

    return result

def append_to_csv(batch, csv_file):
    props = dict(encoding='utf-8', index=False)
    if not os.path.exists(csv_file):
        batch.to_csv(csv_file, **props)
    else:
        batch.to_csv(csv_file, mode='a', header=False, **props)

def delete_file_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename)
        
def run():
    batch_size = 16000

    global whash, signatures
    whash = get_pickled_whashes()
    signatures = get_pickled_signatures()

    pool = avito_utils.PoolWrapper(processes=8)
    name = 'whash'

    print 'processing train data...'
    t0 = time()
    df = pd.read_csv('../input/ItemPairs_train.csv')
    delete_file_if_exists('features_%s_train.csv' % name)

    for batch_no, batch in tqdm(list(prepare_batches(df, batch_size))):
        features = process_batch(batch, pool)
        append_to_csv(features, 'features_%s_train.csv' % name)

    print 'processing train data took %0.5fs' % (time() - t0)

    print 'processinig test data...'
    t0 = time()
    df = pd.read_csv('../input/ItemPairs_test.csv')
    delete_file_if_exists('features_%s_test.csv' % name)

    for batch_no, batch in tqdm(list(prepare_batches(df, batch_size))):
        features = process_batch(batch, pool)
        append_to_csv(features, 'features_%s_test.csv' % name)
        
    print 'processing test data took %0.5fs' % (time() - t0)


    pool.close()
    
if __name__ == "__main__":
    run()