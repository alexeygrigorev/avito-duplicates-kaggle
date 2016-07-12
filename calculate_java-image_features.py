# -*- encoding: utf-8 -*-

import json
import os

from tqdm import tqdm

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

from time import time

import itertools
import avito_utils

def chunck_iterable(it, n):
    while 1:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            break
        yield chunk


def get_pickled_signatures():
    import cPickle
    t0 = time()
    print 'reading image signatures...'
    with open('image-signatures.bin', 'rb') as f:
        res = cPickle.load(f)
        print 'done in %0.5fs' % (time() - t0)
        return res

def get_source_indexes():
    print 'reading item pair files...'
    t0 = time()
    train = pd.read_csv('../input/ItemPairs_train.csv', usecols=['itemID_1', 'itemID_2'], dtype='object')
    test = pd.read_csv('../input/ItemPairs_test.csv', usecols=['itemID_1', 'itemID_2'], dtype='object')

    train = pd.DataFrame({'idx': train.index}, index=train.itemID_1 + '_' + train.itemID_2)
    train['source'] = 0

    test = pd.DataFrame({'idx': test.index}, index=test.itemID_1 + '_' + test.itemID_2)
    test['source'] = 1

    result = pd.concat([train, test])
    print 'done in %0.5fs' % (time() - t0)
    return result


signatures = None

def extract_features(line):
    id1 = line['ad_id_1']
    id2 =  line['ad_id_2']
    result = {'itemID_1': id1, 'itemID_2': id2}

    del line['ad_id_1'], line['ad_id_2']

    if not line:
        return result

    df = pd.Series(line).reset_index()
    df.columns = ['key', 'value']
    
    kp = df[df['key'].str.endswith('kp_no')]
    kp_df = kp['key'].str.split('_', n=2, expand=True)
    kp_df['value'] = kp['value']
    kp_df.columns = ['ad', 'image_id', 'dont_care', 'value']
    kp_df['signature'] = kp_df.image_id.astype(int).apply(signatures.get)

    kp1 = kp_df[kp_df.ad == 'ad1']
    kp2 = kp_df[kp_df.ad == 'ad2']

    if len(kp1) == 0 or len(kp2) == 0:
        return result

    same_signatures = set(kp1.signature) & set(kp2.signature)
    same_images = set(kp_df[kp_df.signature.isin(same_signatures)].image_id)

    # simple keypoints number features
    kp1 = kp1[~kp1.image_id.isin(same_images)]
    kp2 = kp2[~kp2.image_id.isin(same_images)]

    if len(kp1) == 0 or len(kp2) == 0:
        return result

    kp_diff = []
    for k1 in kp1['value']:
        for k2 in kp2['value']:
            kp_diff.append(abs(k1 - k2))
    
    result['kp_diff_min'] = np.min(kp_diff)
    result['kp_diff_mean'] = np.mean(kp_diff) 
    result['kp_diff_max'] = np.max(kp_diff) 
    result['kp_diff_std'] = np.std(kp_diff)
    result['kp_diff_25p'] = np.percentile(kp_diff, q=25)
    result['kp_diff_75p'] = np.percentile(kp_diff, q=75)
    result['kp_diff_skew'] = skew(kp_diff)
    result['kp_diff_kurtosis'] = kurtosis(kp_diff)

    # matching keypoints features
    matches = df[df['key'].str.startswith('keypoints')]
    df_matches = matches['key'].str.split('_', expand=True)
    df_matches.columns = ['pref', 'from_ad', 'to_ad', 'image_id1', 'image_id2']
    df_matches['matched_kps'] = matches['value']
    df_matches = df_matches[~df_matches.image_id1.isin(same_images) & ~df_matches.image_id2.isin(same_images)]
    matched_kps = df_matches.matched_kps.values
    result['kp_matched_min'] = np.min(matched_kps)
    result['kp_matched_mean'] = np.mean(matched_kps) 
    result['kp_matched_max'] = np.max(matched_kps) 
    result['kp_matched_std'] = np.std(matched_kps) 
    result['kp_matched_25p'] = np.percentile(matched_kps, q=25)
    result['kp_matched_75p'] = np.percentile(matched_kps, q=75)
    result['kp_matched_skew'] = skew(matched_kps)
    result['kp_matched_kurtosis'] = kurtosis(matched_kps)
    
    # histogram features
    hist = df[df['key'].str.startswith('hist_')]
    hist_df = hist.key.str.split('_', n=3, expand=True)
    hist_df.columns = [0, 'im1', 'im2', 'distance']
    hist_df.distance = hist_df.distance.str.lower()
    hist_df['value'] = df.value

    hist_df = hist_df[~hist_df.im1.isin(same_images) & ~hist_df.im2.isin(same_images)] 

    for dist, group in hist_df.groupby('distance'):
        values = group.value.values

        result['hist_%s_min' % dist] = np.min(values)
        result['hist_%s_mean' % dist] = np.mean(values)
        result['hist_%s_max' % dist] = np.max(values)
        result['hist_%s_std' % dist] = np.std(values)
        result['hist_%s_25p' % dist] = np.percentile(values, q=25)
        result['hist_%s_75p' % dist] = np.percentile(values, q=75)
        result['hist_%s_skew' % dist] = skew(values)
        result['hist_%s_kurtosis' % dist] = kurtosis(values)

    return result


pool = None

def safe_json_process(line):
    try:
        line = json.loads(line)
        return extract_features(line)
    except:
        print 'error processing', line
        return {}

def process_batch(batch):
    batch = pool.process_parallel(safe_json_process, batch)
    return pd.DataFrame(batch)

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
    batch_size = 8000
    signatures = get_pickled_signatures()

    pool = avito_utils.PoolWrapper()
    
    print 'processing java-image data...'
    t0 = time()
    delete_file_if_exists('features_java-image_all.csv')

    with open('../java-features/image-features.json') as f:
        for batch in tqdm(chunck_iterable(f, batch_size)):
            batch = process_batch(batch)
            append_to_csv(batch, 'features_java-image_all.csv')

    print 'processing took %0.5fs' % (time() - t0)
    pool.close()

    print 'creating csv train and test files...'
    t0 = time()

    df = pd.read_csv('../input/ItemPairs_train.csv')
    df = df.merge(df_java, how='left', on=('itemID_1', 'itemID_2'))
    delete_file_if_exists('features_java-image_train.csv')
    df.to_csv('features_java-image_train.csv', index=False)
    del df

    df = pd.read_csv('../input/ItemPairs_test.csv')
    df = train.merge(df_java, how='left', on=('itemID_1', 'itemID_2'))
    delete_file_if_exists('features_java-image_test.csv')
    df.to_csv('features_java-image_test.csv', index=False)

    print 'done in %0.5fs' % (time() - t0)