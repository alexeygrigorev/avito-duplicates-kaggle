import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from functools import partial
from itertools import islice

import os

from collections import Counter
from tqdm import tqdm
from time import time

import avito_utils
from mongo_utils import MongoWrapper

# https://en.wikipedia.org/wiki/Image_moment
# https://en.wikipedia.org/wiki/Image_moment#Moment_invariants
# http://www.fmwconcepts.com/imagemagick/moments/index.php

imstat_features = [u'imstat_blue_entropy', u'imstat_blue_kurtosis', u'imstat_blue_max', 
                   u'imstat_blue_mean', u'imstat_blue_min', u'imstat_blue_skewness', u'imstat_blue_std',
                   u'imstat_green_entropy', u'imstat_green_kurtosis', u'imstat_green_max',
                   u'imstat_green_mean', u'imstat_green_min', u'imstat_green_skewness', u'imstat_green_std',
                   u'imstat_overall_entropy', u'imstat_overall_kurtosis', u'imstat_overall_max',
                   u'imstat_overall_mean', u'imstat_overall_min', u'imstat_overall_skewness', u'imstat_overall_std',
                   u'imstat_red_entropy', u'imstat_red_kurtosis', u'imstat_red_max', u'imstat_red_mean',
                   u'imstat_red_min', u'imstat_red_skewness', u'imstat_red_std']
diff_features_list = ['filesize'] + imstat_features 

phash_features = [u'phash_blue_ph1', u'phash_blue_ph2', u'phash_blue_ph3', u'phash_blue_ph4', 
                  u'phash_blue_ph5', u'phash_blue_ph6', u'phash_blue_ph7', 
                  u'phash_chroma_ph1', u'phash_chroma_ph2', u'phash_chroma_ph3', u'phash_chroma_ph4', 
                  u'phash_chroma_ph5', u'phash_chroma_ph6', u'phash_chroma_ph7', 
                  u'phash_green_ph1', u'phash_green_ph2', u'phash_green_ph3', u'phash_green_ph4', 
                  u'phash_green_ph5', u'phash_green_ph6', u'phash_green_ph7', 
                  u'phash_hue_ph1', u'phash_hue_ph2', u'phash_hue_ph3', u'phash_hue_ph4', 
                  u'phash_hue_ph5', u'phash_hue_ph6', u'phash_hue_ph7', 
                  u'phash_luma_ph1', u'phash_luma_ph2', u'phash_luma_ph3', u'phash_luma_ph4', 
                  u'phash_luma_ph5', u'phash_luma_ph6', u'phash_luma_ph7', 
                  u'phash_red_ph1', u'phash_red_ph2', u'phash_red_ph3', u'phash_red_ph4', 
                  u'phash_red_ph5', u'phash_red_ph6', u'phash_red_ph7']

feature_groups = {
    'moment_invariants_blue': 
            [u'moments_blue_I1_1', u'moments_blue_I2_1', u'moments_blue_I3_1', u'moments_blue_I4_1',
             u'moments_blue_I5_1', u'moments_blue_I6_1', u'moments_blue_I7_1', u'moments_blue_I8_1'],
    'moment_invariants_green': 
            [u'moments_green_I1_1', u'moments_green_I2_1', u'moments_green_I3_1', u'moments_green_I4_1',
             u'moments_green_I5_1', u'moments_green_I6_1', u'moments_green_I7_1', u'moments_green_I8_1'],
    'moment_invariants_overall': 
            [u'moments_overall_I1_1', u'moments_overall_I2_1', u'moments_overall_I3_1', u'moments_overall_I4_1',
             u'moments_overall_I5_1', u'moments_overall_I6_1', u'moments_overall_I7_1', u'moments_overall_I8_1'],
    'moment_invariants_red': 
            [u'moments_red_I1_1', u'moments_red_I2_1', u'moments_red_I3_1', u'moments_red_I4_1',
             u'moments_red_I5_1', u'moments_red_I6_1', u'moments_red_I7_1', u'moments_red_I8_1'],
    'phash_all': phash_features,
    'phash_blue': [c for c in phash_features if 'blue' in c],
    'phash_chroma': [c for c in phash_features if 'chroma' in c],
    'phash_green': [c for c in phash_features if 'green' in c],
    'phash_hue': [c for c in phash_features if 'hue' in c],
    'phash_luma': [c for c in phash_features if 'luma' in c],
    'phash_red': [c for c in phash_features if 'red' in c],
}

moment_groups = [g for g in feature_groups.keys() if 'moments' in g]
moment_features = sum([feature_groups[g] for g in moment_groups], [])


def geometry_match(df1, df2):
    c1 = Counter(df1.geometry)
    c2 = Counter(df2.geometry)
    return 1.0 * sum((c1 & c2).values()) / min(len(df1), len(df2))

def num_exact_matches(df1, df2):
    return len(set(df1.signature) & set(df2.signature))

def distmat(X, C):    
    X2 = np.sum(X * X, axis=1, keepdims=True)
    C2 = np.sum(C * C, axis=1, keepdims=True)
    XC = np.dot(X, C.T)
    D = X2 - 2 * XC + C2.T
    return D

def manhattan(W1, W2):
    n, _ = W1.shape
    result = [np.abs(W1[[i], :] - W2).sum(axis=1) for i in range(n)]
    return np.array(result)


def pearson_cokurtosis(W1, W2):
    # https://en.wikipedia.org/wiki/Cokurtosis
    W1_mean = W1.mean(axis=1, keepdims=1)
    W2_mean = W2.mean(axis=1, keepdims=1)

    W1_2 = (W1 - W1_mean) ** 2
    W2_2 = (W2 - W2_mean) ** 2
    cov2 = W1_2.dot(W2_2.T)

    var1 = W1_2.sum(axis=1, keepdims=1)
    var2 = W2_2.sum(axis=1, keepdims=1)

    return cov2 / var1.dot(var2.T)

def row_normalize(M):
    norm2 = (M ** 2).sum(axis=1, keepdims=True)
    return M / np.sqrt(norm2)

def cosine(W1, W2):
    W1 = row_normalize(W1)
    W2 = row_normalize(W2)
    return W1.dot(W2.T)

def p25(values):
    return np.percentile(values, q=25)

def p75(values):
    return np.percentile(values, q=75)

functions = [np.min, p25, np.mean, p75, np.max, np.std, skew, kurtosis]

def distance_features(ad1, ad2, result):
    exact_match = set(ad1.signature) & set(ad2.signature) 
    if exact_match:
        ad1 = ad1[~ad1.signature.isin(exact_match)]
        ad2 = ad2[~ad2.signature.isin(exact_match)]

    if len(ad1) == 0 or len(ad2) == 0:
        return

    for group_name, group_columns in feature_groups.items():
        W1 = ad1[group_columns].values
        W2 = ad2[group_columns].values

        vectors = {}
        vectors['euclidean'] = distmat(W1, W2).reshape(-1)
        vectors['manhattan'] = manhattan(W1, W2).reshape(-1)
        vectors['cokurtosis'] = pearson_cokurtosis(W1, W2).reshape(-1)
        vectors['cosine'] = cosine(W1, W2).reshape(-1)

        for vname, values in vectors.items():
            for f in functions:
                name = 'imagemagick_%s_pairs_%s_%s' % (group_name, vname, f.func_name)
                result[name] = f(values)


def abs_diff_features(df1, df2, result):
    for c in diff_features_list:
        diffs = []
        for val1 in df1[c]:
            for val2 in df2[c]:
                diffs.append(abs(val1 - val2))

        for f in functions:
            name = 'imagemagick_abs_diff_%s_%s' % (c, f.func_name)
            result[name] = f(diffs)


centroids =  {
     'blue': (u'moments_blue_ellipse_centroid_x', u'moments_blue_ellipse_centroid_y'),
     'green': (u'moments_green_ellipse_centroid_x', u'moments_green_ellipse_centroid_y'),
     'overall': (u'moments_overall_ellipse_centroid_x', u'moments_overall_ellipse_centroid_y'),
     'red': (u'moments_red_ellipse_centroid_x', u'moments_red_ellipse_centroid_y')
}
            
def ellipse_centroid_features(df1, df2, result):
    for k, (xname, yname) in centroids.items():
        dists = []
        for x1, y1 in zip(df1[xname], df1[yname]):
            for x2, y2 in zip(df2[xname], df2[yname]):
                dist = (x2 - x1) ** 2 + (y2 - y1) ** 2
                dists.append(dist)

        for f in functions:
            name = 'imagemagick_abs_diff_ellipse_%s_%s' % (k, f.func_name)
            result[name] = f(dists)

mongo = None

def scale_features(scaler, sample):
    for c in moment_features:
        min_val = scaler[c]['2%']
        max_val = scaler[c]['98%']
        mean = scaler[c]['mean']
        std = scaler[c]['std']   

        sample.loc[sample[c] < min_val, c] = min_val
        sample.loc[sample[c] > max_val, c] = max_val
        sample[c] = (sample[c] - mean) / std

    return sample



def calc_image_features(ids1, ids2):
    result = {}
    if len(ids1) == 0 or len(ids2) == 0:
        return result

    global mongo, scaler
    image_features = mongo.get_df_by_ids(avito_utils.imagemagick, ids1 + ids2)
    image_features = scale_features(scaler, image_features)

    ad1 = image_features.ix[ids1].dropna(how='all')
    ad2 = image_features.ix[ids2].dropna(how='all')

    if len(ad1) == 0 or len(ad2) == 0:
        return result

    result['imagemagick_geometry_match'] = geometry_match(ad1, ad2)
    result['imagemagick_no_exact_matches'] = num_exact_matches(ad1, ad2)

    abs_diff_features(ad1, ad2, result)
    ellipse_centroid_features(ad1, ad2, result)
    distance_features(ad1, ad2, result)

    return result

def process_batch(batch, pool):
    batch.reset_index(drop=1, inplace=1)

    item_ids = sorted(set(batch.itemID_1) | set(batch.itemID_2))
    batch_items = mongo.get_df_by_ids(avito_utils.item_info, item_ids, columns=['images_array'])

    image_arrays_1 = batch_items.images_array.loc[batch.itemID_1]
    image_arrays_2 = batch_items.images_array.loc[batch.itemID_2]

    result = pool.process_parallel(calc_image_features, collections=(image_arrays_1, image_arrays_2))
    return pd.DataFrame(result)

def prepare_batches(df, n):
    for batch_no, i in enumerate(range(0, len(df), n)):
        yield batch_no, df.iloc[i:i+n]

def delete_file_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename)

def append_to_csv(batch, csv_file):
    props = dict(encoding='utf-8', index=False)
    if not os.path.exists(csv_file):
        batch.to_csv(csv_file, **props)
    else:
        batch.to_csv(csv_file, mode='a', header=False, **props)


def prepare_scaler():
    if os.path.exists('imagemagick_moment_invariants_scaler.csv'):
        sdf = pd.read_csv('imagemagick_moment_invariants_scaler.csv', index_col='index')
        return sdf.to_dict()

    gen = avito_utils.get_imagemagick_table().find()
    sample = list(islice(gen, 100000))
    sample = pd.DataFrame(sample)

    desc = sample[moment_features].describe(percentiles=[0.02, 0.98])
    minmax_scale = desc.loc[['2%', '98%'], :].to_dict()

    for c in moments_features:
        min_val = minmax_scale[c]['2%']
        max_val = minmax_scale[c]['98%']
        sample.loc[sample[c] < min_val, c] = min_val
        sample.loc[sample[c] > max_val, c] = max_val

    desc = sample[moment_features].describe(percentiles=[0.015, 0.985])
    z_scale = desc.loc[['mean', 'std'], :].to_dict()

    for k in minmax_scale.keys():
        minmax_scale[k].update(z_scale[k])

    sdf = pd.DataFrame(minmax_scale)
    sdf.to_csv('imagemagick_moment_invariants_scaler.csv', index_label='index')

    return minmax_scale
        
def run():
    global mongo, scaler
    mongo = MongoWrapper(avito_utils.avito_db)
    scaler = prepare_scaler()

    batch_size = 8000
    name = 'imagemagick'

    pool = avito_utils.PoolWrapper()

    t0 = time()
    df = pd.read_csv('../input/ItemPairs_train.csv')
    delete_file_if_exists('features_%s_train.csv' % name)
    print 'read train set, start processing...'
    for batch_no, batch in tqdm(list(prepare_batches(df, batch_size))):
        batch = process_batch(batch, pool)
        append_to_csv(batch, 'features_%s_train.csv' % name)
    print 'processing train set took %0.5fs' % (time() - t0)

    t0 = time()
    df =  pd.read_csv('../input/ItemPairs_test.csv')
    delete_file_if_exists('features_%s_test.csv' % name)
    print 'read test set, start processing...'
    for batch_no, batch in tqdm(list(prepare_batches(df, batch_size))):
        batch = process_batch(batch, pool)
        append_to_csv(batch, 'features_%s_test.csv' % name)
    print 'processing test set took %0.5fs' % (time() - t0)

    pool.close()

if __name__ == '__main__':
    run()