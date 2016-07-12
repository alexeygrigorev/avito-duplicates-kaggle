# -*- encoding: utf-8 -*-

from collections import Counter
from time import time
import os

from tqdm import tqdm

import pandas as pd
import numpy as np

from functools import partial

import avito_utils

rus_en = {u'а': u'a', u'б': u'b', u'в': u'v', u'г': u'g',  u'д': u'd',  u'е': u'e',  u'ё': u'e', 
          u'ж': u'g', u'з': u'z', u'и': u'i', u'й': u'i',  u'к': u'k',  u'л': u'l',  u'м': u'm', 
          u'н': u'n', u'о': u'o', u'п': u'p', u'р': u'r',  u'с': u's',  u'т': u't',  u'у': u'u', 
          u'ф': u'f', u'х': u'h', u'ц': u'c', u'ч': u'ch', u'ш': u'sh', u'щ': u'sh', u'ъ': None, 
          u'ы': u'i', u'ь': None, u'э': u'e', u'ю': u'y',  u'я': u'ya'}
rus_en = {ord(k): v for k, v in rus_en.items()}

def rus_en_translate(rus):
    return [r.translate(rus_en) for r in rus]

def ngrams(s, n_min=1, n_max=4):
    if n_min == 1:
        result = Counter(s)
    else:
        result = Counter()

    len_s = len(s)

    for n in range(min(2, n_min), n_max + 1):
        result.update(''.join(s[i:i+n]) for i in xrange(len_s - n + 1))

    return result

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

def calc_jaccard(set1, set2, lam=0):
    union = len(set1 | set2)
    if not union: 
        return 0.0

    inter = len(set1 & set2)
    return inter * 1.0 / (union + lam)

def jaccard_set(c1, c2, lam=0):
    s1 = set(c1.keys())
    s2 = set(c2.keys())
    return calc_jaccard(s1, s2, lam)


mongo = avito_utils.mongo

columns=['digits', 'english_digits_mix', 'russian_digits_mix', 'rus_eng_chars_mixed', 'unicode_chars',
         'english_only']

pool = None

def generate_simple_match(series_1, series_2, prefix, result):
    series1_cnt = pool.process_parallel(Counter, collection=series_1)
    series2_cnt = pool.process_parallel(Counter, collection=series_2)

    result[prefix + '_cos'] = \
            pool.process_parallel(cosine, collections=(series1_cnt, series2_cnt))
    result[prefix + '_jaccard'] = \
            pool.process_parallel(jaccard_set, collections=(series1_cnt, series2_cnt)) 
    result[prefix + '_jaccard_reg'] = \
            pool.process_parallel(jaccard_set, collections=(series1_cnt, series2_cnt), lam=1) 
    
    return result


def generate_match_features(series_1, series_2, prefix, result):
    result = generate_simple_match(series_1, series_2, prefix, result)
    
    series14_1 = pool.process_parallel(ngrams, collection=series_1, n_min=1, n_max=4)
    series14_2 = pool.process_parallel(ngrams, collection=series_2, n_min=1, n_max=4)

    result[prefix + '14_cos'] = \
            pool.process_parallel(cosine, collections=(series14_1, series14_2))
    result[prefix + '14_jaccard'] = \
            pool.process_parallel(jaccard_set, collections=(series14_1, series14_2))
    result[prefix + '14_jaccard_reg'] = \
            pool.process_parallel(jaccard_set, collections=(series14_1, series14_2), lam=1)

    series24_1 = pool.process_parallel(ngrams, collection=series_1, n_min=2, n_max=4)
    series24_2 = pool.process_parallel(ngrams, collection=series_2, n_min=2, n_max=4)

    result[prefix + '24_cos'] = \
            pool.process_parallel(cosine, collections=(series24_1, series24_2))
    result[prefix + '24_jaccard'] = \
            pool.process_parallel(jaccard_set, collections=(series24_1, series24_2))
    result[prefix + '24_jaccard_reg'] = \
            pool.process_parallel(jaccard_set, collections=(series24_1, series24_2), lam=1)

    return result

def process_batch(batch):    
    batch.reset_index(drop=True, inplace=1)

    item_ids = sorted(set(batch.itemID_1) | set(batch.itemID_2))
    text = mongo.get_df_by_ids(avito_utils.item_text, item_ids, columns=columns)
    text['rus_digits_mix_translated'] = pool.process_parallel(rus_en_translate, collection=text.russian_digits_mix)

    items_1 = text.loc[batch.itemID_1].reset_index(drop=True)
    items_1.columns = [col + '_1' for col in items_1.columns]
    items_2 = text.loc[batch.itemID_2].reset_index(drop=True)
    items_2.columns = [col + '_2' for col in items_2.columns]

    batch = pd.concat([batch, items_1, items_2], axis=1)
    result = pd.DataFrame()

    generate_match_features(batch.digits_1, batch.digits_2, 'digits', result)
    generate_match_features(batch.english_only_1, batch.english_only_2, 'en_only', result)
    generate_simple_match(batch.english_digits_mix_1, batch.english_digits_mix_2, 'en_digits_mix', result)
    generate_simple_match(batch.russian_digits_mix_1, batch.russian_digits_mix_2, 'ru_digits_mix', result)
    generate_simple_match(batch.unicode_chars_1, batch.unicode_chars_2, 'unicode_chars', result)

    en_rus_digits_translated_1 = batch.english_digits_mix_1 + batch.rus_digits_mix_translated_1
    en_rus_digits_translated_2 = batch.english_digits_mix_2 + batch.rus_digits_mix_translated_2

    generate_simple_match(en_rus_digits_translated_1, en_rus_digits_translated_2, 
                          'en+ru_translated_digits_mix', result)

    all_1 = batch.digits_1 + batch.english_only_1 + batch.english_digits_mix_1 + batch.russian_digits_mix_1 + \
             batch.rus_digits_mix_translated_1

    all_2 = batch.digits_2 + batch.english_only_2 + batch.english_digits_mix_2 + batch.russian_digits_mix_2 + \
             batch.rus_digits_mix_translated_2

    generate_simple_match(all_1, all_2, 'all_important_tokens', result)

    all_1 = all_1 + batch.unicode_chars_1
    all_2 = all_2 + batch.unicode_chars_2

    generate_simple_match(all_1, all_2, 'all_important_tokens+unicode_chars', result)

    result['rus_en_mixed'] = batch.rus_eng_chars_mixed_1 * batch.rus_eng_chars_mixed_2
    result['rus_en_mixed_one'] = (batch.rus_eng_chars_mixed_1 > 0) | (batch.rus_eng_chars_mixed_2 > 0)
    result['rus_en_mixed_one'] = result['rus_en_mixed_one'].astype(int)
    result['rus_en_mixed_both'] = (batch.rus_eng_chars_mixed_1 > 0) & (batch.rus_eng_chars_mixed_2 > 0)
    result['rus_en_mixed_both'] = result['rus_en_mixed_both'].astype(int)

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
    batch_size = 8000

    name = 'important_tokens'
    
    pool = avito_utils.PoolWrapper()
    
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
    
    pool.close()