from collections import Counter
from time import time
import os

from tqdm import tqdm

import pandas as pd
import numpy as np

import avito_utils


def dict_combine(cnt1, cnt2):
    res = cnt1.copy()
    for k, v in cnt2.items():
        res[k] = res.get(k, 0) + v
    return res

def dict_union(cnt1, cnt2):
    res = {}
    for k, v1 in cnt1.items():
        if k not in cnt2: 
            continue
        v2 = cnt2[k]
        res[k] = min(v1, v2)
    return res

def dict_union_norm(cnt1, cnt2, lam=1):
    res = {}
    for k, v1 in cnt1.items():
        if k not in cnt2: 
            continue
        v2 = cnt2[k]
        res[k] = 1.0 * min(v1, v2) / (max(v1, v2) + lam)
    return res

def dict_combine_abs_diff(cnt1, cnt2, lam=1):
    res = {}
    for k, v1 in cnt1.items():
        if k not in cnt2: 
            res[k] = np.abs(v1)
        else:
            v2 = cnt2[k]
            res[k] = np.abs(v1 - v2)
    return res

def dict_count_union(cnt1, cnt2):
    res = 0
    for k, v in cnt1.items():
        if k not in cnt2: 
            continue
        res = res + min(v, cnt2[v])
    return res

def values_cnt(d):
    return sum(i for i in d.values())

def match_count_normalized(cnt1, cnt2):
    union_count = dict_count_union(cnt1, cnt2)
    val1_count = values_cnt(cnt1)
    val2_count = values_cnt(cnt2)
    return 1.0 * union_count / max(val1_count, val2_count)

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

def euclidean(c1, c2):
    res = 0
    keys = set(c1.keys()) | set(c2.keys())
    for k in keys:
        res = res + (c1.get(k, 0) - c2.get(k, 0)) ** 2
    return res



mongo = avito_utils.mongo

def na_to_dict(el):
    if isinstance(el, float) and np.isnan(el):
        return {}
    return el

def items_prepare(items, columns_suffix):
    items.fillna({'title_mistakes_cnt': 0, 'desc_mistakes_cnt': 0, 'all_mistakes_cnt': 0}, inplace=1)
    items.title = items.title.apply(na_to_dict)
    items.description = items.description.apply(na_to_dict)
    items.all_text = items.all_text.apply(na_to_dict)
    items.reset_index(inplace=1, drop=1)
    items.columns = [col + '_' + columns_suffix for col in items.columns]
    return items

diff_list = ['title_mistakes_cnt', 'desc_mistakes_cnt', 'all_mistakes_cnt']
cnt_list = ['title', 'description', 'all_text']

columns = [u'all_mistakes_cnt_diff', u'all_text_CommaWhitespaceRule_dist', u'all_text_CommaWhitespaceRule_union',
           u'all_text_CommaWhitespaceRule_union_norm', u'all_text_DoublePunctuationRule_dist', 
           u'all_text_DoublePunctuationRule_union', u'all_text_DoublePunctuationRule_union_norm',
           u'all_text_MorfologikRussianSpellerRule_dist', u'all_text_MorfologikRussianSpellerRule_union',
           u'all_text_MorfologikRussianSpellerRule_union_norm', u'all_text_MultipleWhitespaceRule_dist',
           u'all_text_MultipleWhitespaceRule_union', u'all_text_MultipleWhitespaceRule_union_norm',
           u'all_text_PatternRule_dist', u'all_text_PatternRule_union', u'all_text_PatternRule_union_norm',
           u'all_text_RussianCompoundRule_dist', u'all_text_RussianCompoundRule_union',
           u'all_text_RussianCompoundRule_union_norm', u'all_text_RussianSimpleReplaceRule_dist',
           u'all_text_RussianSimpleReplaceRule_union', u'all_text_RussianSimpleReplaceRule_union_norm',
           u'all_text_RussianUnpairedBracketsRule_dist', u'all_text_RussianUnpairedBracketsRule_union',
           u'all_text_RussianUnpairedBracketsRule_union_norm', u'all_text_UppercaseSentenceStartRule_dist',
           u'all_text_UppercaseSentenceStartRule_union', u'all_text_UppercaseSentenceStartRule_union_norm',
           u'all_text_WordRepeatRule_dist', u'all_text_WordRepeatRule_union', 
           u'all_text_WordRepeatRule_union_norm', u'all_text_spelling_cosine', 
           u'all_text_spelling_dist', u'all_text_spelling_jaccard', 
           u'desc_mistakes_cnt_diff', u'description_CommaWhitespaceRule_dist', u'description_CommaWhitespaceRule_union',
           u'description_CommaWhitespaceRule_union_norm', u'description_DoublePunctuationRule_dist',
           u'description_DoublePunctuationRule_union', u'description_DoublePunctuationRule_union_norm',
           u'description_MorfologikRussianSpellerRule_dist', u'description_MorfologikRussianSpellerRule_union',
           u'description_MorfologikRussianSpellerRule_union_norm', u'description_MultipleWhitespaceRule_dist',
           u'description_MultipleWhitespaceRule_union', u'description_MultipleWhitespaceRule_union_norm',
           u'description_PatternRule_dist', u'description_PatternRule_union', u'description_PatternRule_union_norm',
           u'description_RussianCompoundRule_dist', u'description_RussianCompoundRule_union',
           u'description_RussianCompoundRule_union_norm', u'description_RussianSimpleReplaceRule_dist',
           u'description_RussianSimpleReplaceRule_union', u'description_RussianSimpleReplaceRule_union_norm',
           u'description_RussianUnpairedBracketsRule_dist', u'description_RussianUnpairedBracketsRule_union',
           u'description_RussianUnpairedBracketsRule_union_norm', u'description_UppercaseSentenceStartRule_dist',
           u'description_UppercaseSentenceStartRule_union', u'description_UppercaseSentenceStartRule_union_norm',
           u'description_WordRepeatRule_dist', u'description_WordRepeatRule_union', 
           u'description_WordRepeatRule_union_norm', u'description_spelling_cosine', u'description_spelling_dist',
           u'description_spelling_jaccard', 
           u'title_CommaWhitespaceRule_dist', u'title_CommaWhitespaceRule_union', u'title_CommaWhitespaceRule_union_norm',
           u'title_MorfologikRussianSpellerRule_dist', u'title_MorfologikRussianSpellerRule_union',
           u'title_MorfologikRussianSpellerRule_union_norm', u'title_MultipleWhitespaceRule_dist', 
           u'title_PatternRule_dist', u'title_PatternRule_union', u'title_PatternRule_union_norm',
           u'title_RussianCompoundRule_dist', u'title_RussianCompoundRule_union', u'title_RussianCompoundRule_union_norm',
           u'title_RussianSimpleReplaceRule_dist', u'title_RussianSimpleReplaceRule_union',
           u'title_RussianSimpleReplaceRule_union_norm', u'title_RussianUnpairedBracketsRule_dist',
           u'title_RussianUnpairedBracketsRule_union', u'title_RussianUnpairedBracketsRule_union_norm',
           u'title_UppercaseSentenceStartRule_dist', u'title_UppercaseSentenceStartRule_union',
           u'title_UppercaseSentenceStartRule_union_norm', u'title_WordRepeatRule_dist', 
           u'title_WordRepeatRule_union', u'title_WordRepeatRule_union_norm', u'title_mistakes_cnt_diff',
           u'title_spelling_cosine', u'title_spelling_dist', u'title_spelling_jaccard'] 

def process_batch(batch):    
    batch.reset_index(drop=True, inplace=1)

    item_ids = map(str, sorted(set(batch.itemID_1) | set(batch.itemID_2)))
    df_spellings = mongo.get_df_by_ids('item_spelling', item_ids, index_col=None)
    df_spellings._id = df_spellings._id.astype(int)
    df_spellings['all_text'] = df_spellings.title.combine(df_spellings.description, dict_combine)
    df_spellings['title_mistakes_cnt'] = df_spellings.title.apply(values_cnt)
    df_spellings['desc_mistakes_cnt']  = df_spellings.description.apply(values_cnt)
    df_spellings['all_mistakes_cnt']   = df_spellings.all_text.apply(values_cnt)
    df_spellings.set_index('_id', inplace=1)

    items_1 = df_spellings.loc[batch.itemID_1].reset_index(drop=True)
    items_1 = items_prepare(items_1, '1')
    items_2 = df_spellings.loc[batch.itemID_2].reset_index(drop=True)
    items_2 = items_prepare(items_2, '2')
    
    batch = pd.concat([batch, items_1, items_2], axis=1)
    result = pd.DataFrame(columns=columns)

    for col in diff_list:
        result[col + '_diff'] = (batch[col + '_1'] - batch[col + '_2']).abs()

    dict_dfs = []
    for col in cnt_list:
        result[col + '_spelling_jaccard'] = batch[col + '_1'].combine(batch[col + '_2'], jaccard_set)
        result[col + '_spelling_cosine']  = batch[col + '_1'].combine(batch[col + '_2'], cosine)
        result[col + '_spelling_dist']    = batch[col + '_1'].combine(batch[col + '_2'], euclidean)

        d1 = batch[col + '_1'].combine(batch[col + '_2'], dict_union).apply(pd.Series)
        d2 = batch[col + '_1'].combine(batch[col + '_2'], dict_union_norm).apply(pd.Series)
        d3 = batch[col + '_1'].combine(batch[col + '_2'], dict_combine_abs_diff).apply(pd.Series)

        d1.columns = [col + '_' + c + '_union'      for c in d1.columns]
        d2.columns = [col + '_' + c + '_union_norm' for c in d2.columns]
        d3.columns = [col + '_' + c + '_dist'       for c in d3.columns]
        
        for c in d1.columns:
            result[c] = d1[c]
        for c in d2.columns:
            result[c] = d2[c]
        for c in d2.columns:
            result[c] = d2[c]

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
    
    t0 = time()
    df = pd.read_csv('../input/ItemPairs_train.csv')
    delete_file_if_exists('features_misspellings_train.csv')
    print 'read train set, start processing...'
    for batch_no, batch in tqdm(list(prepare_batches(df, batch_size))):
        batch = process_batch(batch)
        append_to_csv(batch, 'features_misspellings_train.csv')
    print 'processing train set took %0.5fs' % (time() - t0)

    t0 = time()
    df =  pd.read_csv('../input/ItemPairs_test.csv')
    delete_file_if_exists('features_misspellings_test.csv')
    print 'read test set, start processing...'
    for batch_no, batch in tqdm(list(prepare_batches(df, batch_size))):
        batch = process_batch(batch)
        append_to_csv(batch, 'features_misspellings_test.csv')
    print 'processing test set took %0.5fs' % (time() - t0)