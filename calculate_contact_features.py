# -*- encoding: utf-8 -*-

from collections import Counter
from time import time
import os

from tqdm import tqdm

import pandas as pd
import numpy as np

from functools import partial

import avito_utils

mongo = avito_utils.mongo

def calc_jaccard(set1, set2, lam=0):
    union = len(set1 | set2)
    if not union: 
        return 0.0

    inter = len(set1 & set2)
    return inter * 1.0 / (union + lam)

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

def process_batch(batch):
    item_ids = map(str, sorted(set(batch.itemID_1) | set(batch.itemID_2)))
    contacts = mongo.get_by_ids('item_contacts', item_ids)
    contacts = {int(d['_id']): d for d in contacts}

    result = []

    for id1, id2 in zip(batch.itemID_1, batch.itemID_2):
        rec_dict = {'itemID_1': id1, 'itemID_2': id2}

        if id1 not in contacts or id2 not in contacts:
            result.append(rec_dict)
            continue

        c1 = contacts[id1]
        c2 = contacts[id2]

        placeholders1 = Counter(c1.get('placeholders', []))
        placeholders2 = Counter(c2.get('placeholders', []))
        rec_dict['contact_cosine_placeholders'] = cosine(placeholders1, placeholders2)

        phones1 = c1.get('phones', [])
        phones2 = c2.get('phones', [])
        rec_dict['contact_phones_jaccard'] = calc_jaccard(set(phones1), set(phones2), lam=1)
        rec_dict['contact_phones_len_diff'] = abs(len(phones1) - len(phones2))

        links1 = c1.get('links', [])
        links2 = c2.get('links', [])
        rec_dict['contact_links_jaccard'] = calc_jaccard(set(links1), set(links2), lam=1)
        rec_dict['contact_links_len_diff'] = abs(len(links1) - len(links2))

        emails1 = c1.get('emails', [])
        emails2 = c2.get('emails', [])
        rec_dict['contact_emails_jaccard'] = calc_jaccard(set(emails1), set(emails2), lam=1)
        rec_dict['contact_emails_len_diff'] = abs(len(emails1) - len(emails2))

        hashtags1 = c1.get('hashtags', [])
        hashtags2 = c2.get('hashtags', [])
        rec_dict['contact_hashtags_jaccard'] = calc_jaccard(set(hashtags1), set(hashtags2), lam=1)
        rec_dict['contact_hashtags_len_diff'] = abs(len(hashtags1) - len(hashtags2))

        all_details1 = emails1 + links1 + phones1 + hashtags1
        all_details2 = emails2 + links2 + phones2 + hashtags2
        rec_dict['contact_all_jaccard'] = calc_jaccard(set(all_details1), set(all_details2), lam=1)    
        rec_dict['contact_all_len_diff'] = abs(len(all_details1) - len(all_details2))

        result.append(rec_dict)

    return pd.DataFrame(result)


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
    batch_size = 2000

    name = 'contact'
    
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