from collections import Counter
from time import time
import os
from functools import partial

from tqdm import tqdm

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

import avito_utils

from multiprocessing import Pool

def process_parallel(pool, series, function, **args):
    if not args:
        return pool.map(function, series)
    else: 
        part_function = partial(function, **args)
        return pool.map(part_function, series)

def apply_func(args, func):
    return func(*args)

    
def prepare_batches(df, n):
    for batch_no, i in enumerate(range(0, len(df), n)):
        yield batch_no, df.iloc[i:i+n]

functions = [fuzz.QRatio, fuzz.UQRatio, fuzz.UWRatio, fuzz.UWRatio,
             fuzz.WRatio, fuzz.partial_ratio, fuzz.partial_token_set_ratio, fuzz.partial_token_sort_ratio,
             fuzz.token_set_ratio, fuzz.token_sort_ratio]

column_pairs = [('title_1', 'title_2'), ('description_1', 'description_2'),
                ('title_1', 'description_2'), ('description_1', 'title_2'), 
                ('all_text_1', 'all_text_2')]

def process_batch(batch, pool):    
    batch.reset_index(drop=True, inplace=1)

    item_ids = sorted(set(batch.itemID_1) | set(batch.itemID_2))
    df_items = avito_utils.get_item_infos(item_ids, columns=['title_clean', 'description_clean'])
    df_items.rename(columns={'title_clean': 'title', 'description_clean': 'description'}, inplace=1)
    df_items['all_text'] = df_items['title'] + df_items['description']

    item1 = df_items.loc[batch.itemID_1].reset_index(drop=True)
    item1.columns = [col + '_1' for col in item1.columns]
    item2 = df_items.loc[batch.itemID_2].reset_index(drop=True)
    item2.columns = [col + '_2' for col in item2.columns]
    batch = pd.concat([batch, item1, item2], axis=1)
    
    for f in functions:
        for c1, c2 in column_pairs:
            name = '%s_%s_%s' % (c1, c2, f.func_name)
            data = zip(batch[c1], batch[c2])
            batch[name] = process_parallel(pool, data, apply_func, func=f)

    to_drop = ['title', 'description', 'all_text', 'itemID']
    batch.drop([c + '_1' for c in to_drop], axis=1, inplace=1)
    batch.drop([c + '_2' for c in to_drop], axis=1, inplace=1)
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
        
if __name__ == "__main__":
    batch_size = 8000
    
    pool = Pool(processes=8)
    
    t0 = time()
    df = pd.read_csv('../input/ItemPairs_train.csv')
    delete_file_if_exists('features_fuzzy_train.csv')
    print 'read train set, start processing...'
    for batch_no, batch in tqdm(list(prepare_batches(df, batch_size))):
        batch = process_batch(batch, pool)
        append_to_csv(batch, 'features_fuzzy_train.csv')
    print 'processing train set took %0.5fs' % (time() - t0)

    t0 = time()
    df =  pd.read_csv('../input/ItemPairs_test.csv')
    delete_file_if_exists('features_fuzzy_test.csv')
    print 'read test set, start processing...'
    for batch_no, batch in tqdm(list(prepare_batches(df, batch_size))):
        batch = process_batch(batch, pool)
        append_to_csv(batch, 'features_fuzzy_test.csv')
    print 'processing test set took %0.5fs' % (time() - t0)

    pool.close()
    pool.join()