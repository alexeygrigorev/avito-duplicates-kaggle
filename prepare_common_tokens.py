
# coding: utf-8

# In[1]:

import cPickle
import codecs
import csv

import numpy as np
import pandas as pd

from tqdm import tqdm

import avito_utils


# In[2]:

corrections = {}

with codecs.open('./replacements.txt', 'r', 'utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        correct, others = line.split('=')
        for other in others.split(','):
            if not other:
                continue
            corrections[other] = correct


# In[3]:

stopwords = set(u'в,на,и,с,по,для,не,от,за,к,до,из,или,у,один,вы,при,так,ваш,как,а,наш,быть,под,б,'
                u'р,мы,эт,же,также,что,это,раз,свой,он,если,но,я,о,ещё,тот,этот,то,они,ни,чем,где,'
                u'бы,оно,там,ж,она,ты'.split(','))


# In[4]:

def clean_tokens_list(tokens_list):
    result = []
    for sentence in tokens_list:
        for t in sentence.split():
            result.append(corrections.get(t, t))
    return result

token_to_idx = {}
current_cnt = 0

def save_memory(tokens):
    global current_cnt, token_to_idx
    
    res = []
    for t in tokens:
        if t in stopwords:
            continue
        if t not in token_to_idx:
            token_to_idx[t] = current_cnt
            current_cnt = current_cnt + 1
        res.append(token_to_idx[t])
    return res    


# In[6]:

df_train = pd.read_csv('../input/ItemPairs_train.csv', 
                    dtype={u'itemID_1': np.uint32, u'itemID_2': np.uint32, u'isDuplicate': np.uint8},
                    usecols=[u'itemID_1', u'itemID_2', u'isDuplicate'])
df_test = pd.read_csv('../input/ItemPairs_test.csv',
                    dtype={u'itemID_1': np.uint32, u'itemID_2': np.uint32},
                    usecols=[u'itemID_1', u'itemID_2'])
test_ids = set(df_test.itemID_1) |  set(df_test.itemID_2)

# In[13]:

title_records = {}
desc_records = {}

words_in_train = set()
words_in_test = set()

gen = avito_utils.select(avito_utils.item_text, columns=['title_lemmatized', 'description_lemmatized'])

for batch in tqdm(gen):
    for rec in batch:
        _id = np.uint32(rec['_id'])
        title = clean_tokens_list(rec["title_lemmatized"])
        title = set(save_memory(title))
        title_records[_id] = title

        desc = clean_tokens_list(rec["description_lemmatized"])
        desc = set(save_memory(desc))
        desc_records[_id] = desc

        if _id in test_ids:
            words_in_test.update(title)
            words_in_test.update(desc)
        else:
            words_in_train.update(title)
            words_in_train.update(desc)

# In[14]:

words_in_both = words_in_test & words_in_train
del words_in_test, words_in_train

# In[17]:

inverse_idx = {v: k for (k, v) in token_to_idx.items() if v in words_in_both}
token_to_idx = {v: k for (k, v) in inverse_idx.items()}

def restore_string(idx_list):
    str_list = []
    for sent in idx_list:
        str_list.append([inverse_idx[t] for t in sent])
    return str_list


# In[26]:

for _id in tqdm(title_records.keys()):
    title_records[_id].intersection_update(words_in_both)
    desc_records[_id].intersection_update(words_in_both)

# In[45]:

with open('text_models/tokens_idx.bin', 'wb') as f:
    cPickle.dump((inverse_idx, token_to_idx), f)


# ## Process Data

# In[41]:

def keep_common(s1, s2):
    return s1 & s2

def keep_diff(s1, s2):
    return (s1 | s2) - (s1 & s2)

def set_to_str(s):
    return ' '.join(map(str, s))

def process_row(row, train=True):
    result = {}
    if train:
        result['isDuplicate'] = row.isDuplicate
    itemID_1 = row.itemID_1
    itemID_2 = row.itemID_2

    title_1 = title_records[itemID_1]
    title_2 = title_records[itemID_2]
    result['title_common'] = set_to_str(keep_common(title_1, title_2))
    result['title_diff'] = set_to_str(keep_diff(title_1, title_2))

    desc_1 = desc_records[itemID_1]
    desc_2 = desc_records[itemID_2]
    result['desc_common'] = set_to_str(keep_common(desc_1, desc_2))
    result['desc_diff'] = set_to_str(keep_diff(desc_1, desc_2))

    all_1 = title_1 | desc_1
    all_2 = title_2 | desc_2
    result['all_common'] = set_to_str(keep_common(all_1, all_2))
    result['all_diff'] = set_to_str(keep_diff(all_1, all_2))
    return result


# In[40]:

fields = ['isDuplicate', 'title_common', 'title_diff', 'desc_common', 'desc_diff', 'all_common', 'all_diff']

with open('common_diff_tokens_train.csv', 'w') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()

    for i, row in tqdm(df_train.iterrows()):
        r = process_row(row, train=True)
        writer.writerow(r)


# In[ ]:

fields = ['title_common', 'title_diff', 'desc_common', 'desc_diff', 'all_common', 'all_diff']

with open('common_diff_tokens_test.csv', 'w') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()

    for i, row in tqdm(df_test.iterrows()):
        r = process_row(row, train=False)
        writer.writerow(r)

