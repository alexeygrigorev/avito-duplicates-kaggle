# -*- encoding: utf-8 -*-

from time import time
import cPickle
import codecs

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from bm25 import BM25Transformer

from tqdm import tqdm

import avito_utils
def identity(l): return l


# In[7]:

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


# In[8]:

stopwords = set(u'в,на,и,с,по,для,не,от,за,к,до,из,или,у,один,вы,при,так,ваш,как,а,наш,быть,под,б,'
                u'р,мы,эт,же,также,что,это,раз,свой,он,если,но,я,о,ещё,тот,этот,то,они,ни,чем,где,'
                u'бы,оно,там,ж,она,ты'.split(','))


# In[9]:

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


# In[10]:

all_titles = []
all_desc = []
all_text = []

gen = avito_utils.select(avito_utils.item_text, columns=['title_lemmatized', 'description_lemmatized'])

for batch in tqdm(gen):
    for rec in batch:
        title = clean_tokens_list(rec["title_lemmatized"])
        title = save_memory(title)
        all_titles.append(title)

        desc = clean_tokens_list(rec["description_lemmatized"])
        desc = save_memory(desc)
        all_desc.append(desc)

        both = title + desc
        all_text.append(both)

inverse_idx = {v: k for (k, v) in token_to_idx.items()}

def restore_string(idx_list):
    str_list = []
    for sent in tqdm(idx_list):
        str_list.append([inverse_idx[t] for t in sent])
    return str_list

all_titles = restore_string(all_titles)
all_desc = restore_string(all_desc)
all_text = restore_string(all_text)

del token_to_idx, inverse_idx


# Titles

# In[8]:

t0 = time()

count_title = CountVectorizer(analyzer=identity, min_df=5)
title_matrix = count_title.fit_transform(all_titles)

print 'count of titles took %0.2fs' % (time() - t0)

title_matrix


# In[9]:

del all_titles


# In[10]:

with open('text_models/title_count_(min_df=5).bin', 'wb') as f:
    cPickle.dump(count_title, f)


# In[11]:

t0 = time()

svd_title_count = TruncatedSVD(n_components=50, random_state=1)
svd_title_count.fit(title_matrix)

print 'svd of titles took %0.2fs' % (time() - t0)


# In[12]:

with open('text_models/title_count_svd_50.bin', 'wb') as f:
    cPickle.dump(svd_title_count, f)


# In[13]:

del svd_title_count


# In[15]:

t0 = time()

tfidf_title = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False)
tfidf_matrix = tfidf_title.fit_transform(title_matrix)

print 'tfidf of titles took %0.2fs' % (time() - t0)

tfidf_matrix


# In[16]:

with open('text_models/title_tfidf_(norm=None,smooth_idf=True,sublinear_tf=False).bin', 'wb') as f:
    cPickle.dump(tfidf_title, f)


# In[43]:

del tfidf_title


# In[17]:

t0 = time()

svd_title_tfidf = TruncatedSVD(n_components=50, random_state=1)
svd_title_tfidf.fit(tfidf_matrix)

print 'svd of titles took %0.2fs' % (time() - t0)


# In[18]:

with open('text_models/title_tfidf_svd_50.bin', 'wb') as f:
    cPickle.dump(svd_title_tfidf, f)


# In[19]:

del svd_title_tfidf, tfidf_matrix


# In[22]:

t0 = time()

b25_title = BM25Transformer(use_idf=True, k1=2.0, b=0.75)
b25_matrix = b25_title.fit_transform(title_matrix)

print 'bm25 of titles took %0.2fs' % (time() - t0)

b25_matrix


# In[26]:

with open('text_models/title_bm25_(k1=2,b=075).bin', 'wb') as f:
    cPickle.dump(b25_title, f)


# In[44]:

del b25_title


# In[32]:

t0 = time()

svd_title_bm25 = TruncatedSVD(n_components=50, random_state=1)
svd_title_bm25.fit(b25_matrix)

print 'svd of titles took %0.2fs' % (time() - t0)


# In[33]:

with open('text_models/title_bm25_svd_50.bin', 'wb') as f:
    cPickle.dump(svd_title_bm25, f)


# In[34]:

del b25_matrix, svd_title_bm25
del title_matrix


# ## Description

# In[35]:

t0 = time()

count_desc = CountVectorizer(analyzer=identity, min_df=5)
desc_matrix = count_desc.fit_transform(all_desc)

print 'tfidf of desc took %0.2fs' % (time() - t0)

with open('text_models/desc_count_(min_df=5).bin', 'wb') as f:
    cPickle.dump(count_desc, f)
del all_desc, count_desc

t0 = time()

svd_desc = TruncatedSVD(n_components=100, random_state=1)
svd_desc.fit(desc_matrix)

print 'svd of desc took %0.2fs' % (time() - t0)

with open('text_models/desc_count_svd_100.bin', 'wb') as f:
    cPickle.dump(svd_desc, f)
del svd_desc


t0 = time()

tfidf_desc = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False)
tfidf_matrix = tfidf_desc.fit_transform(desc_matrix)

print 'tfidf of desc took %0.2fs' % (time() - t0)

with open('text_models/desc_tfidf_(norm=None,smooth_idf=True,sublinear_tf=False).bin', 'wb') as f:
    cPickle.dump(tfidf_desc, f)
del tfidf_desc


t0 = time()

svd_desc_tfidf = TruncatedSVD(n_components=100, random_state=1)
svd_desc_tfidf.fit(tfidf_matrix)

print 'svd of desc took %0.2fs' % (time() - t0)

with open('text_models/desc_tfidf_svd_100.bin', 'wb') as f:
    cPickle.dump(svd_desc_tfidf, f)

del svd_desc_tfidf, tfidf_matrix


t0 = time()

bm25_desc = BM25Transformer(use_idf=True, k1=2.0, b=0.75)
bm25_matrix = bm25_desc.fit_transform(desc_matrix)

print 'bm25 of desc took %0.2fs' % (time() - t0)


with open('text_models/desc_bm25_(k1=2,b=075).bin', 'wb') as f:
    cPickle.dump(bm25_desc, f)
del bm25_desc


t0 = time()

svd_desc_bm25 = TruncatedSVD(n_components=100, random_state=1)
svd_desc_bm25.fit(bm25_matrix)

print 'svd of desc took %0.2fs' % (time() - t0)

with open('text_models/desc_bm25_svd_100.bin', 'wb') as f:
    cPickle.dump(svd_desc_bm25, f)

del svd_desc_bm25, bm25_matrix
del desc_matrix


# ## All text

# In[14]:

t0 = time()

count_all = CountVectorizer(analyzer=identity, min_df=5)
all_matrix = count_all.fit_transform(all_text)

print 'counting of all took %0.2fs' % (time() - t0)

with open('text_models/all_count_(min_df=5).bin', 'wb') as f:
    cPickle.dump(count_all, f)
del all_text, count_all

t0 = time()

svd_all = TruncatedSVD(n_components=130, random_state=1)
svd_all.fit(all_matrix)

print 'svd of all took %0.2fs' % (time() - t0)

with open('text_models/all_count_svd_130.bin', 'wb') as f:
    cPickle.dump(svd_all, f)
del svd_all


t0 = time()

tfidf_all = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False)
tfidf_matrix = tfidf_all.fit_transform(all_matrix)

print 'tfidf of all took %0.2fs' % (time() - t0)

with open('text_models/all_tfidf_(norm=None,smooth_idf=True,sublinear_tf=False).bin', 'wb') as f:
    cPickle.dump(tfidf_all, f)
del tfidf_all


t0 = time()

svd_all_tfidf = TruncatedSVD(n_components=130, random_state=1)
svd_all_tfidf.fit(tfidf_matrix)

print 'svd of all took %0.2fs' % (time() - t0)

with open('text_models/all_tfidf_svd_130.bin', 'wb') as f:
    cPickle.dump(svd_all_tfidf, f)

del svd_all_tfidf, tfidf_matrix


t0 = time()

bm25_all = BM25Transformer(use_idf=True, k1=2.0, b=0.75)
bm25_matrix = bm25_all.fit_transform(all_matrix)

print 'bm25 of all took %0.2fs' % (time() - t0)


with open('text_models/all_bm25_(k1=2,b=075).bin', 'wb') as f:
    cPickle.dump(bm25_all, f)
del bm25_all


t0 = time()

svd_all_bm25 = TruncatedSVD(n_components=130, random_state=1)
svd_all_bm25.fit(bm25_matrix)

print 'svd of all took %0.2fs' % (time() - t0)

with open('text_models/all_bm25_svd_130.bin', 'wb') as f:
    cPickle.dump(svd_all_bm25, f)

del svd_all_bm25, bm25_matrix
del all_matrix


# In[ ]: