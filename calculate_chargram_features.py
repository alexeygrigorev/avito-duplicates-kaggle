from collections import Counter
from time import time
import os

from tqdm import tqdm

import pandas as pd
import numpy as np

import avito_utils

def cosine(c1, c2):
    keys = set(c1.keys()) & set(c2.keys())
    numerator = sum([c1[x] * c2[x] for x in keys])
    sum1 = sum([c1[x] ** 2 for x in c1.keys()])
    sum2 = sum([c2[x] ** 2 for x in c2.keys()])
    denominator = np.sqrt(sum1 * sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator
    
def calc_jaccard(set1, set2):
    union = len(set1 | set2)
    if not union: 
        return 0.0

    inter = len(set1 & set2)
    return inter * 1.0 / union

def jaccard_set(c1, c2):
    s1 = set(c1.keys())
    s2 = set(c2.keys())
    return calc_jaccard(s1, s2)


def get_first(s):
    try:
        if len(s) > 6:
            split = re.split('[-;,]', s)
            return int(split[0].strip())
        else:
            return int(s)
    except:
        return 0

def zip_to_int(zip2):
    zip2 = zip2.replace('None', '0')
    return zip2.apply(get_first)

def zip_code_diff(zip1, zip2):
    zip1 = zip_to_int(zip1)
    zip2 = zip_to_int(zip2)
    return (zip1 - zip2).abs()

def prepare_batches(df, n):
    for batch_no, i in enumerate(range(0, len(df), n)):
        yield batch_no, df.iloc[i:i+n]

def same_nan(series1, series2):
    eq = (series1 == series2)
    eq[series1.isnull() & series2.isnull()] = np.nan
    return eq

def combine_dict(cnt1, cnt2):
    res = cnt1.copy()
    for k, v in cnt2.items():
        res[k] = res.get(k, 0) + v
    return res

diff_list = [u'all_text_digits_char', u'all_text_digits_ratio', u'all_text_en_char', u'all_text_en_ratio',
             u'all_text_len', u'all_text_non_char', u'all_text_non_char_ratio', u'all_text_ru_char',
             u'all_text_ru_ratio', u'all_text_unique_char', 
             u'description_digits_char', u'description_digits_ratio', u'description_en_char', 
             u'description_en_ratio', u'description_len', u'description_non_char',
             u'description_non_char_ratio', u'description_ru_char', u'description_ru_ratio',
             u'description_unique_char',
             u'images_cnt', u'price',
             u'title_digits_char', u'title_digits_ratio', u'title_en_char', u'title_en_ratio', u'title_len',
             u'title_non_char', u'title_non_char_ratio', u'title_ru_char', u'title_ru_ratio', u'title_unique_char']

def process_batch(batch):    
    batch.reset_index(drop=True, inplace=1)

    item_ids = sorted(set(batch.itemID_1) | set(batch.itemID_2))
    df_items = avito_utils.get_item_infos(item_ids)
    df_chargrams = avito_utils.get_chargrams(item_ids)
    df_items = pd.concat([df_items, df_chargrams], axis=1)

    title = df_items['chargram_title']
    desc = df_items['chargram_desc']
    df_items['chargram_full'] = title.combine(desc, combine_dict)

    item1 = df_items.loc[batch.itemID_1].reset_index(drop=True)
    item1.columns = [col + '_1' for col in item1.columns]
    item2 = df_items.loc[batch.itemID_2].reset_index(drop=True)
    item2.columns = [col + '_2' for col in item2.columns]
    batch = pd.concat([batch, item1, item2], axis=1)

    batch['same_location'] = same_nan(batch.locationID_1, batch.locationID_2)
    batch['same_metro'] = same_nan(batch.metroID_1, batch.metroID_2)
    batch['same_region'] = same_nan(batch.region_1, batch.region_2)
    batch['same_city'] = same_nan(batch.city_1, batch.city_2)

    batch['zip_diff'] = zip_code_diff(batch.zip_code_1, batch.zip_code_2)
    
    batch['title_jaccard'] = batch.chargram_title_1.combine(batch.chargram_title_2, jaccard_set)
    batch['desc_jaccard'] = batch.chargram_desc_1.combine(batch.chargram_desc_2, jaccard_set)
    batch['all_text_jaccard'] = batch.chargram_full_1.combine(batch.chargram_full_2, jaccard_set)

    batch['title_cosine'] = batch.chargram_title_1.combine(batch.chargram_title_2, cosine)
    batch['desc_cosine'] = batch.chargram_desc_1.combine(batch.chargram_desc_2, cosine)
    batch['all_text_cosine'] = batch.chargram_full_1.combine(batch.chargram_full_2, cosine)

    batch['distance'] = np.sqrt((batch.lat_1 - batch.lat_2) ** 2 + (batch.lon_1 - batch.lon_2) ** 2)

    batch['category'] = batch.categoryID_1

    for col in diff_list:
        batch[col + '_diff'] = (batch[col + '_1'] - batch[col + '_2']).abs()

    to_drop = ['title', 'description', 'title_clean', 'description_clean',
               'chargram_title', 'chargram_desc', 'chargram_full', 'images_array', 
               'attrsJSON',
               'lat', 'lon', 'categoryID', 'parent_category'] + diff_list + ['region', 'city', 'zip_code']
    batch.drop([c + '_1' for c in to_drop], axis=1, inplace=1)
    batch.drop([c + '_2' for c in to_drop], axis=1, inplace=1)
    batch.drop(['itemID_1', 'itemID_2'], axis=1, inplace=1)
    
    return batch


def append_to_csv(batch, csv_file):
    props = dict(encoding='utf-8', index=False)
    if not os.path.exists(csv_file):
        batch.to_csv(csv_file, **props)
    else:
        batch.to_csv(csv_file, mode='a', header=False, **props)

if __name__ == "__main__":
    batch_size = 2500
    
    t0 = time()
    df = pd.read_csv('../input/ItemPairs_train.csv')
    print 'read train set, start processing...'
    for batch_no, batch in tqdm(list(prepare_batches(df, batch_size))):
        batch = process_batch(batch)
        append_to_csv(batch, 'features_chargrams_train.csv')
    print 'processing train set took %0.5fs' % (time() - t0)

    t0 = time()
    df =  pd.read_csv('../input/ItemPairs_test.csv')
    print 'read test set, start processing...'
    for batch_no, batch in tqdm(list(prepare_batches(df, batch_size))):
        batch = process_batch(batch)
        append_to_csv(batch, 'features_chargrams_test.csv')
    print 'processing test set took %0.5fs' % (time() - t0)
