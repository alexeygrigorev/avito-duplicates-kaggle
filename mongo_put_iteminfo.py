# -*- encoding: utf-8 -*-

from collections import Counter
import itertools 
import string
from time import time
import json
import re
from zipfile import ZipFile

from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import cPickle
import os

import avito_utils

from multiprocessing import Pool

def process_parallel(pool, series, function):
    return pool.map(function, series)

pool = None

def read_dictionaries():
    location_region = pd.read_csv('../input/Location.csv', index_col='locationID').regionID.to_dict()
    parent_category = pd.read_csv('../input/Category.csv', index_col='categoryID').parentCategoryID.to_dict()
    return location_region, parent_category

def read_geo_data():
    t0 = time()

    geo_zip = ZipFile('../ext_data/geo_stuff.zip')
    with geo_zip.open('geo-train.txt') as f:
        df_geo_train = pd.read_csv(f, sep='\t', header=None, 
                                   names=['itemID', 'region', 'city', 'zip_code'])
    with geo_zip.open('geo-test.txt') as f:
        df_geo_test = pd.read_csv(f, sep='\t',header=None, 
                                  names=['itemID', 'region', 'city', 'zip_code'])

    df_geo = pd.concat([df_geo_train, df_geo_test])
    df_geo.set_index('itemID', inplace=1)

    print 'preparing geo data took %0.5fs.' % (time() - t0)
    return df_geo


en = {unichr(i) for i in range(ord(u'A'), ord(u'Z') + 1) + range(ord(u'a'), ord(u'z') + 1)}
ru = {unichr(i) for i in range(ord(u'А'), ord(u'Я') + 1) + range(ord(u'а'), ord(u'я') + 1)} | set(u'Ёё')
digits = set(string.digits)
chars = en | ru | digits 

def count_chars(ctr, chars):
    return sum(v for (k, v) in ctr.items() if k in chars)

def count_chars_ru(ctr):
    return count_chars(ctr, ru)

def count_chars_en(ctr):
    return count_chars(ctr, en)

def count_chars_digits(ctr):
    return count_chars(ctr, digits)

def count_chars_non_alphanum(ctr):
    return sum(v for (k, v) in ctr.items() if k not in chars)

def safe_div(series1, series2):
    res = series1 / series2
    return res.replace(np.inf, 0.0)

def counter_keys_no(ctr):
    return len(ctr.keys())

def calculate_features(batch, pool):
    for col in ['title', 'description']:
        batch[col + '_len'] = batch[col].apply(len)
        chars_ctr = process_parallel(pool, batch[col], Counter)
        
        batch[col + '_unique_char'] = process_parallel(pool, chars_ctr, counter_keys_no)

        batch[col + '_ru_char'] = process_parallel(pool, chars_ctr, count_chars_ru)
        batch[col + '_en_char'] = process_parallel(pool, chars_ctr, count_chars_en)
        batch[col + '_digits_char'] = process_parallel(pool, chars_ctr, count_chars_digits)
        batch[col + '_non_char'] = process_parallel(pool, chars_ctr, count_chars_non_alphanum)

        batch[col + '_ru_ratio'] = safe_div(batch[col + '_ru_char'], batch[col + '_len'])
        batch[col + '_en_ratio'] = safe_div(batch[col + '_en_char'], batch[col + '_len'])
        batch[col + '_digits_ratio'] = safe_div(batch[col + '_digits_char'], batch[col + '_len'])
        batch[col + '_non_char_ratio'] = safe_div(batch[col + '_non_char'], batch[col + '_len'])

    all_text = batch['title'] + batch['description']
    batch['all_text_len'] = batch['title_len'] + batch['description_len']
    batch['all_text_unique_char'] = all_text.apply(set).apply(len)
    batch['all_text_ru_char'] = batch['title_ru_char'] + batch['description_ru_char']
    batch['all_text_en_char'] = batch['title_en_char'] + batch['description_en_char']
    batch['all_text_digits_char'] = batch['title_digits_char'] + batch['description_digits_char']
    batch['all_text_non_char'] = batch['title_non_char'] + batch['description_non_char']

    batch['all_text_ru_ratio'] = safe_div(batch['all_text_ru_char'], batch['all_text_len'])
    batch['all_text_en_ratio'] = safe_div(batch['all_text_en_char'], batch['all_text_len'])
    batch['all_text_digits_ratio'] = safe_div(batch['all_text_digits_char'], batch['all_text_len'])
    batch['all_text_non_char_ratio'] = safe_div(batch['all_text_non_char'], batch['all_text_len'])

def to_unicode(s):
    return unicode(s.decode('utf8'))

def clean_text(text):
    text = text.lower()
    text = text.replace(u'ё', u'е')
    text = re.sub(u'[^a-zа-я0-9]', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

def split_images_array(arr):
    return [z.strip() for z in arr.split(',') if z]

def process_batch(batch, location_region, parent_category, geo, pool):
    batch.set_index('itemID', inplace=1)

    batch['parent_category'] = batch.categoryID.apply(parent_category.get)
    batch['region'] = batch.locationID.apply(location_region.get)
    batch = pd.concat([batch, geo.loc[batch.index].fillna('None')], axis=1)

    batch.attrsJSON = process_parallel(pool, batch.attrsJSON.fillna('{}'), json.loads)
    batch.images_array = process_parallel(pool, batch.images_array.fillna(''), split_images_array)
    batch['images_cnt'] = batch.images_array.apply(len)

    batch.title = process_parallel(pool, batch.title.fillna(''), to_unicode)
    batch.description = process_parallel(pool, batch.description.fillna(''), to_unicode)
    
    batch['title_clean'] = process_parallel(pool, batch.title, clean_text)
    batch['description_clean'] = process_parallel(pool, batch.description, clean_text)

    calculate_features(batch, pool)

    return batch

def append_to_csv(batch, csv_file):
    if not os.path.exists(csv_file):
        batch.to_csv(csv_file, encoding='utf-8')
    else:
        batch.to_csv(csv_file, encoding='utf-8', mode='a', header=False)

mongo_items_info = avito_utils.get_item_info_table()

def save_to_mongo(batch):
    batch['_id'] = batch.index
    batch_dict = batch.to_dict(orient='records')
    mongo_items_info.insert_many(batch_dict)

def process_file(source, batch_size, pool):
    print 'reading dictionaries...'
    location_region, parent_category = read_dictionaries()
    print 'reading geo data...'
    geo = read_geo_data()

    t_all = time()
    filename = '../input/ItemInfo_%s.csv' % source
    print 'processing %s data from %s' % (source, filename)

    df_iter = pd.read_csv(filename, iterator=True, chunksize=batch_size)
    cnt = 1
    processed = 0

    for batch in df_iter:
        print 'processing batch #%d... (itemID=[%d .. %d])' % (cnt, batch.itemID.iloc[0], batch.itemID.iloc[-1]),
        t0 = time()
        batch = process_batch(batch, location_region, parent_category, geo, pool)
        # append_to_csv(batch, 'item_features_%s.txt' % source)
        print 'took %0.5fs.' % (time() - t0),
        processed = processed + len(batch)
        print '(so far processed %d rows)' % processed
        save_to_mongo(batch)
        cnt = cnt + 1

    print 'processing %s took %0.5fs.' % (source, time() - t_all),


def run():
    avito_utils.get_item_info_table().drop()

    pool = Pool(processes=8)

    batch_size = 250000
    process_file('train', batch_size, pool)
    process_file('test', batch_size, pool)

    pool.close()
    pool.join()

if __name__ == "__main__":
    run()
