# -*- encoding: utf-8 -*-

from time import time
import os

from tqdm import tqdm

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from collections import defaultdict

from fastcache import clru_cache as lru_cache

import avito_utils

def prepare_batches(df, n):
    for batch_no, i in enumerate(range(0, len(df), n)):
        yield batch_no, df.iloc[i:i+n]

@lru_cache(maxsize=256)
def count_ones(integer):
    return sum(ord(i) - 48 for i in np.binary_repr(integer))

@lru_cache(maxsize=100)
def hex_to_int(h):
    return np.frombuffer(h.decode('hex'), dtype=np.uint8)

def hamming_distance(h1, h2):
    b1 = hex_to_int(h1)
    b2 = hex_to_int(h2)
    return sum([count_ones(x) for x in np.bitwise_xor(b1, b2)])

def dot_product(h1, h2):
    b1 = hex_to_int(h1)
    b2 = hex_to_int(h2)
    return sum([count_ones(x) for x in np.bitwise_and(b1, b2)])

def bernoulli_mean(h):
    b = hex_to_int(h)
    n = len(h) * 8
    ones = sum([count_ones(x) for x in b])
    p = 1.0 * ones / n
    return p

def entropy(h):
    p = bernoulli_mean(h)

    if p == 0 or p == 1:
        return 0
    else:
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def abs_diff(v1, v2):
    return abs(v1 - v2)

def abs_entropy_diff(h1, h2):
    e1 = entropy(h1)
    e2 = entropy(h2)
    return abs_diff(e1, e2)

def cross_entropy(h1, h2):
    b1 = hex_to_int(h1)
    b2 = hex_to_int(h2)

    n = len(h2) * 8
    p1 = 1.0 * sum([count_ones(x) for x in b1]) / n
    p2 = 1.0 * sum([count_ones(x) for x in b2]) / n
    
    if p2 == 0 or p2 == 1:
        return 0
    else:
        return -(p1 * np.log(p2) + (1 - p1) * np.log(1 - p2))

def symmetric_cross_entropy(h1, h2):
    e1 = cross_entropy(h1, h2)
    e2 = cross_entropy(h2, h1)
    return (e1 + e2) / 2

hash_columns = ['script_dhash', 'ahash', 'dhash', 'phash']
functions = [hamming_distance, dot_product, abs_entropy_diff, symmetric_cross_entropy]

df_hashes = None
mongo = avito_utils.mongo


def calc_features(images1, images2):
    if not images1 or not images2:
        return {}

    global df_hashes

    hashes1 = df_hashes.ix[images1].dropna()
    hashes2 = df_hashes.ix[images2].dropna()
    if len(hashes1) == 0 or len(hashes2) == 0:
        return {}

    common = set(hashes1.signature) & set(hashes2.signature)
    hashes1 = hashes1[~hashes1.signature.isin(common)]
    hashes2 = hashes2[~hashes2.signature.isin(common)]
    
    if len(hashes1) == 0 or len(hashes2) == 0:
        return {}

    result = {}
    
    for c in hash_columns:
        ad1_hashes = set(hashes1[c].dropna())
        ad2_hashes = set(hashes2[c].dropna())

        sub_results = defaultdict(list)

        for f in functions:
            for h1 in ad1_hashes:
                for h2 in ad2_hashes:
                    sub_results[f.func_name].append(f(h1, h2))

        for k, values in sub_results.items():
            result['imagehash_%s_%s_min' % (c, k)] = np.min(values)
            result['imagehash_%s_%s_mean' % (c, k)] = np.mean(values)
            result['imagehash_%s_%s_max' % (c, k)] = np.max(values)
            result['imagehash_%s_%s_std' % (c, k)] = np.std(values)
            result['imagehash_%s_%s_25p' % (c, k)] = np.percentile(values, q=25)
            result['imagehash_%s_%s_75p' % (c, k)] = np.percentile(values, q=75)
            result['imagehash_%s_%s_skew' % (c, k)] = skew(values)
            result['imagehash_%s_%s_kurtosis' % (c, k)] = kurtosis(values)

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

    batch.drop(['itemID_1', 'itemID_2'], axis=1, inplace=1)
    batch = pd.concat([batch, result], axis=1)

    return batch

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
    batch_size = 4000

    print 'reading image hashes from image_hashes.csv...',
    t0 = time()
    global df_hashes
    df_hashes = pd.read_csv('image_hashes.csv')
    df_hashes.set_index('image_id', inplace=1)
    print 'took %0.5fs' % (time() - t0)

    pool = avito_utils.PoolWrapper(processes=4)

    print 'processing train data...'
    t0 = time()
    df = pd.read_csv('../input/ItemPairs_train.csv')
    delete_file_if_exists('features_imagehash_train.csv')

    for batch_no, batch in tqdm(list(prepare_batches(df, batch_size))):
        features = process_batch(batch, pool)
        append_to_csv(features, 'features_imagehash_train.csv')

    print 'processing train data took %0.5fs' % (time() - t0)

    print 'processinig test data...'
    t0 = time()
    df = pd.read_csv('../input/ItemPairs_test.csv')
    delete_file_if_exists('features_imagehash_test.csv')

    for batch_no, batch in tqdm(list(prepare_batches(df, batch_size))):
        features = process_batch(batch, pool)
        append_to_csv(features, 'features_imagehash_test.csv')

    print 'processing test data took %0.5fs' % (time() - t0)

    pool.close()
    
if __name__ == "__main__":
    run()