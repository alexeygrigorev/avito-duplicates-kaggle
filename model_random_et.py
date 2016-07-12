# coding: utf-8

# In[2]:

import os
import random
import json
from pprint import pprint

from time import time

import pandas as pd
import numpy as np

from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier

genmethod_include = False

t0 = time()

seed = int(time() * 1000)
print 'using seed =', seed

random.seed(seed)
np.random.seed(seed % 4294967295)

def jitter(val, spread):
    rand = np.random.uniform(low=-spread, high=spread)
    return val + rand


n_estimators = np.random.randint(low=160, high=180)

et_params = dict( 
    criterion=random.choice(['entropy', 'gini']),
    max_depth=np.random.randint(low=35, high=50),
    min_samples_split=np.random.randint(low=2, high=6),
    min_samples_leaf=np.random.randint(low=2, high=6),
    max_features=np.random.randint(low=60, high=80), 
    bootstrap=random.choice([True, False]), 
    n_jobs=-1,
    random_state=np.random.randint(low=0, high=20),
)

print 'using params:'
pprint(et_params)

# In[114]:

def peek_columns(csv):
    return pd.read_csv(csv, nrows=1).columns

def columns_with(cols, text):
    if not isinstance(text, list):
        return cols[cols.str.contains(text)].tolist()
    else:
        result = []
        for t in text:
            c = cols[cols.str.contains(t)]
            result.extend(c)
        return result

def rand_empty(l, p=0.5):
    if random.random() <= p:
        return l
    else:
        return []


# In[4]:

label = pd.read_csv('features_chargrams_train.csv', usecols=['isDuplicate'], dtype=np.uint8)
y_full = label.isDuplicate.values


# In[8]:

print 'reading chargram train...'
chargram_cols = ['locationID_1', 'locationID_2', 'metroID_1', 'metroID_2', 'category', 
                 'distance', 'price_diff', 'zip_diff']

chargram_option1 = ['all_text_jaccard', 'title_jaccard']
chargram_option2 = ['all_text_cosine', 'title_cosine']
chargram_options = [chargram_option1, chargram_option2]

chargram_cols = chargram_cols + random.choice(chargram_options)
df_diffs_chargrams = pd.read_csv('features_chargrams_train.csv', usecols=chargram_cols, dtype=np.float32)


# In[48]:

print 'reading fuzzy train...'

fuzzy_options = ['QRatio', 'UQRatio', 'UWRatio', 'UWRatio', 'WRatio', 'partial_ratio', 'partial_token_set_ratio', 
                 'partial_token_sort_ratio', 'token_set_ratio', 'token_sort_ratio']

cols = peek_columns('features_fuzzy_train.csv')
fuzzy_cols = sorted(columns_with(cols, random.choice(fuzzy_options)))

df_fuzzy = pd.read_csv('features_fuzzy_train.csv', usecols=fuzzy_cols, dtype=np.float32)


# In[54]:

cols = peek_columns('features_imagemagick_train.csv')

colors = ['red', 'green', 'blue', 'overall']
ellipse_options = ['ellipse_%s' % c for c in colors]
stats = ['mean', 'std', 'kurtosis', 'skewness'] #['min', 'mean', 'max', 'std', 'kurtosis', 'skewness']
imstat_options = ['imstat_%s_%s' % (c, s) for c in colors for s in stats]

metrics = ['cosine', 'euclidean', 'manhattan', 'cokurtosis']
moment_invariants = ['moment_invariants_%s_pairs_%s' % (c, m) for c in colors for m in metrics]

colors = ['red', 'green', 'blue', 'chroma', 'hue', 'luma', 'all']
phash = ['phash_%s_pairs_%s' % (c, m) for c in colors for m in metrics]

always_include = ['imagemagick_no_exact_matches', 'imagemagick_geometry_match']

group1 = columns_with(cols, random.choice(ellipse_options))
group2 = columns_with(cols, random.choice(imstat_options))
group3 = columns_with(cols, random.choice(moment_invariants))
group4 = columns_with(cols, random.choice(phash))

imagemagick_cols = always_include + \
        rand_empty(group1, 0.7) + rand_empty(group2, 0.7) + rand_empty(group3, 0.75) + \
        rand_empty(group4, 0.9)
df_imagemagick = pd.read_csv('features_imagemagick_train.csv', usecols=imagemagick_cols, dtype=np.float32)


# In[88]:

imagehash_cols = []
df_imagehash = pd.DataFrame()

whash_cols = []
df_whash = pd.DataFrame()

ssim_cols = []
df_ssim = pd.DataFrame()


# In[91]:

hashes = ['ahash', 'dhash', 'phash', 'script_dhash', 'whash', 'ssim', 'ssim']
hash = random.choice(hashes)

if hash == 'whash':
    print 'reading whash'
    df_whash = pd.read_csv('features_whash_train.csv', dtype=np.float32)
    whash_cols = list(df_whash.columns)
elif hash == 'ssim':
    print 'reading ssim...'
    df_ssim = pd.read_csv('features_ssim_train.csv', dtype=np.float32)
    ssim_cols = list(df_ssim.columns)
else:
    print 'reading imagehash'
    cols = peek_columns('features_imagehash_train.csv')

    hashes = ['ahash', 'dhash', 'phash', 'script_dhash']
    entropies = ['abs_entropy_diff', 'symmetric_cross_entropy']
    entropy_cols = ['%s_%s' % (h, e) for h in hashes for e in entropies]

    dists = ['hamming_distance', 'dot_product']
    dist_cols = ['%s_%s' % (h, d) for h in hashes for d in dists]

    group1 = columns_with(cols, random.choice(entropy_cols))
    group2 = columns_with(cols, random.choice(dist_cols))

    imagehash_cols = rand_empty(group1, 0.5) + group2
    df_imagehash = pd.read_csv('features_imagehash_train.csv', usecols=imagehash_cols, dtype=np.float32)


# In[96]:

print 'reading java_image'
cols = peek_columns('features_java-image_train.csv')

hist_dists = ['bhattacharyya', 'chi_square', 'correlation', 'cosine', 'hamming', 
              'jaccard_distance', 'sum_square']

java_image_cols  = columns_with(cols, random.choice(hist_dists))

df_java = pd.read_csv('features_java-image_train.csv', usecols=java_image_cols, dtype=np.float32)


# In[124]:

print 'reading w2v'

cols = peek_columns('features_w2v_train.csv')

field_interactions = [
          'all_text_1_all_text_2', 'description_1_description_2',
          'description_1_title_2', 'title_1_description_2', 'nouns_1_nouns_2',
          'title_1_title_2']
w2v_type = ['all', 'cokurtosis', 'corr', 'cosine', 'euclidean', 'manhattan', 'same_excluded']
field_names = ['%s_%s' % (f1, f2) for f1 in field_interactions for f2 in w2v_type]

groups = sorted(np.random.choice(field_names, size=3, replace=False))

w2v_cols = sorted(columns_with(cols, groups))
df_w2v = pd.read_csv('features_w2v_train.csv', usecols=w2v_cols, dtype=np.float32)


# In[127]:

print 'reading w2v_wmd'

cols = peek_columns('features_w2v_wmd_train.csv')
w2v_wmd_cols = sorted(np.random.choice(cols, size=4, replace=False))
df_w2v_wmd = pd.read_csv('features_w2v_wmd_train.csv', usecols=w2v_wmd_cols, dtype=np.float32)


# In[201]:

print 'reading glove'

cols = peek_columns('features_glove_train.csv')

wmd_cols = cols[cols.str.contains('wmd')]
glove_wmd_cols = sorted(np.random.choice(wmd_cols, size=4, replace=False))

cols = cols[~cols.str.contains('wmd')]

field_interactions = [
          'all_text_1_all_text_2', 'description_1_description_2',
          'description_1_title_2', 'title_1_description_2', 'nouns_1_nouns_2',
          'title_1_title_2']
glove_type = ['all', 'cosine', 'euclidean', 'manhattan', 'same_excluded']
field_names = ['%s_%s' % (f1, f2) for f1 in field_interactions for f2 in glove_type]

groups = sorted(np.random.choice(field_names, size=3, replace=False))
glove_cols = sorted(columns_with(cols, groups)) + glove_wmd_cols

df_glove = pd.read_csv('features_glove_train.csv', usecols=glove_cols, dtype=np.float32)


# In[142]:

print 'reading text'

cols = peek_columns('features_text_train.csv')

jaccard = ['text_title_jaccard_reg', 'text_title_jaccard_full', 'text_title_intersect',
 'text_desc_jaccard_reg', 'text_desc_jaccard_full', 'text_desc_intersect',
 'text_all_jaccard_reg', 'text_all_jaccard_full', 'text_all_intersect']

text = ['all', 'title', 'desc']
vect = ['count', 'tfidf', 'bm25']
groups = ['%s_%s' % (f1, f2) for f1 in text for f2 in vect]

text_cols = \
    sorted(np.random.choice(jaccard, size=2, replace=False)) +  \
    sorted(columns_with(cols, random.choice(groups)))

df_text = pd.read_csv('features_text_train.csv', usecols=text_cols, dtype=np.float32)


# In[155]:

print 'reading common_tokens'

cols = peek_columns('features_common_tokens_train.csv')
svm = ['svm_title_common', 'svm_title_diff', 'svm_title_both', 
       'svm_desc_common', 'svm_desc_diff', 'svm_desc_both',
       'pos_common_tokens_sum', 'neg_common_tokens_sum', 'pos_diff_tokens_sum', 'neg_diff_tokens_sum']

groups = ['svd_%s' % c for c in ['title_common', 'title_diff', 'desc_common', 'desc_diff', 'all_text']]

common_tokens_cols = sorted(np.random.choice(svm, size=1, replace=False)) + \
    sorted(columns_with(cols, random.choice(groups)))

df_common_tokens = pd.read_csv('features_common_tokens_train.csv', usecols=common_tokens_cols, dtype=np.float32)


# In[168]:

print 'reading dimred...'

cols = peek_columns('features_dimred_train.csv')

groups = ['fuzzy', 'imhash', 'imagemagick', 'javaimage', 'w2v', 'text', 'glove']
dimred_cols = sorted(columns_with(cols, random.choice(groups)))

df_dimred = pd.read_csv('features_dimred_train.csv', usecols=dimred_cols, dtype=np.float32)


# In[170]:

print 'reading imp_tokens...'

imp_tokens_cols = ['digits_cos', 'en+ru_translated_digits_mix_cos', 'en_digits_mix_cos',
 'en_only_cos', 'ru_digits_mix_cos', 'rus_en_mixed', 'rus_en_mixed_both', 'rus_en_mixed_one',
 'unicode_chars_cos']

imp_tokens_cols = sorted(np.random.choice(imp_tokens_cols, size=4, replace=False))
df_imp_tokens = pd.read_csv('features_important_tokens_train.csv', usecols=imp_tokens_cols, dtype=np.float32)


# In[172]:

print 'reading misspelling train...'

misspelling_cols = ['all_mistakes_cnt_diff',
 'all_text_CommaWhitespaceRule_union_norm',
 'all_text_MorfologikRussianSpellerRule_union',
 'all_text_MorfologikRussianSpellerRule_union_norm',
 'all_text_MultipleWhitespaceRule_union',
 'all_text_MultipleWhitespaceRule_union_norm',
 'all_text_UppercaseSentenceStartRule_union_norm',
 'all_text_spelling_cosine', 'all_text_spelling_dist', 'all_text_spelling_jaccard',
 'description_CommaWhitespaceRule_dist',
 'description_MorfologikRussianSpellerRule_union_norm',
 'description_spelling_cosine',
 'title_MorfologikRussianSpellerRule_union_norm',
 'title_spelling_cosine']

misspelling_cols = sorted(np.random.choice(misspelling_cols, size=4, replace=False))

df_misspellings = pd.read_csv('features_misspellings_train.csv', usecols=misspelling_cols, dtype=np.float32)
df_misspellings.fillna(0, inplace=1)

# In[208]:

print 'reading attributes...'

cols = peek_columns('features_attrs_train.csv')
attrs_cols = sorted(np.random.choice(cols, size=6, replace=False))

df_attrs = pd.read_csv('features_attrs_train.csv', usecols=attrs_cols, dtype=np.float32)


# In[209]:

df_train = pd.concat([df_diffs_chargrams, df_fuzzy, df_imagemagick, df_imagehash, 
                      df_java, df_w2v, df_glove, df_attrs,
                      df_w2v_wmd, df_text, df_common_tokens, df_whash,
                      df_dimred, df_ssim, df_misspellings, df_imp_tokens], axis=1)
df_train.fillna(-999, inplace=1)

features = list(df_train.columns)


del df_diffs_chargrams, df_fuzzy, df_imagemagick, df_imagehash
del df_java, df_w2v, df_glove, df_attrs
del df_w2v_wmd, df_text, df_common_tokens, df_whash
del df_dimred, df_ssim, df_misspellings, df_imp_tokens


# In[21]:

X_full = df_train[features].values
del df_train

X_full[np.isposinf(X_full)] =  10000
X_full[np.isneginf(X_full)] = -10000

## 

cv = KFold(len(y_full), n_folds=3, shuffle=True, random_state=42)

ets = [ExtraTreesClassifier(warm_start=True, **et_params),
       ExtraTreesClassifier(warm_start=True, **et_params),
       ExtraTreesClassifier(warm_start=True, **et_params)]
print

for n in range(10, n_estimators + 1, 10):
    print '%3d ' % n, 
    scores = []
    for idx, (train, val) in enumerate(cv):
        et = ets[idx]
        et.n_estimators = n
        et.fit(X_full[train], y_full[train])
        y_pred = et.predict_proba(X_full[val])[:, 1]
        score = roc_auc_score(y_full[val], y_pred)
        scores.append(score)
    print 'mean: %0.5f' % np.mean(scores),
    print '[' + ', '.join('%0.5f' % s for s in scores) + ']'


train_preds = np.zeros(y_full.shape)

scores = []
for idx, (train, val) in enumerate(cv):
    et = ets[idx]
    y_pred = et.predict_proba(X_full[val])[:, 1]
    train_preds[val] = y_pred

    score = roc_auc_score(y_full[val], y_pred)
    scores.append(score)

# 

print 'mean auc:', np.mean(scores)   

mscore = np.mean(scores)
mscore = int(mscore * 10000)

filename = 'et9_%s_%s' % (seed, mscore)

print 'saving train scores to', filename

train_res = pd.DataFrame({'probability': train_preds})
train_res.to_csv(filename + '_train.csv.gz', index=False, compression='gzip')

settings = {
    'chargram_cols': chargram_cols,
    'fuzzy_cols': fuzzy_cols,
    'imagemagick_cols': imagemagick_cols,
    'imagehash_cols': imagehash_cols,
    'java_image_cols': java_image_cols,
    'glove_cols': glove_cols,
    'w2v_cols': w2v_cols,
    'attrs_cols': attrs_cols,
    'w2v_wmd_cols': w2v_wmd_cols,
    'text_cols': text_cols,
    'common_tokens_cols': common_tokens_cols,
    'dimred_cols': dimred_cols,
    'whash_cols': whash_cols,
    'ssim_cols': ssim_cols,
    'imp_tokens_cols': imp_tokens_cols,
    'misspelling_cols': misspelling_cols,
    'score': np.mean(scores),
    'et_params': et_params,
    'model': 'et',
    'n_estimators': n_estimators,
    'seed': seed,
}


with file(filename + '_settings.json', 'w') as f:
    json.dump(settings, f, indent=2)



# In[23]:


et_full = ExtraTreesClassifier(n_estimators=n_estimators, verbose=1, **et_params)
et_full.fit(X_full, y_full)

del X_full, y_full

# In[ ]:


# In[27]:
print 'reading chargrams...'
df_diffs_chargrams = pd.read_csv('features_chargrams_test.csv', usecols=chargram_cols, dtype=np.float32)

print 'reading fuzzy...'
df_fuzzy = pd.read_csv('features_fuzzy_test.csv', usecols=fuzzy_cols, dtype=np.float32)

print 'reading imagemagick...'
df_imagemagick = pd.read_csv('features_imagemagick_test.csv', usecols=imagemagick_cols, dtype=np.float32)

if imagehash_cols:
    print 'reading imagehash...'
    df_imagehash = pd.read_csv('features_imagehash_test.csv', usecols=imagehash_cols, dtype=np.float32)
else:
    df_imagehash = pd.DataFrame()

print 'reading java-image...'
df_java = pd.read_csv('features_java-image_test.csv', usecols=java_image_cols, dtype=np.float32)

print 'reading w2v...'
df_w2v = pd.read_csv('features_w2v_test.csv', usecols=w2v_cols, dtype=np.float32)

print 'reading glove...'
df_glove = pd.read_csv('features_glove_test.csv', usecols=glove_cols, dtype=np.float32)

print 'reading attributes...'
df_attrs = pd.read_csv('features_attrs_test.csv', usecols=attrs_cols, dtype=np.float32)

print 'reading w2v_wmd...'
df_w2v_wmd = pd.read_csv('features_w2v_wmd_test.csv', usecols=w2v_wmd_cols, dtype=np.float32)

print 'reading text...'
df_text = pd.read_csv('features_text_test.csv', usecols=text_cols, dtype=np.float32)

print 'reading common_tokens...'
df_common_tokens = pd.read_csv('features_common_tokens_test.csv', usecols=common_tokens_cols, dtype=np.float32)

if whash_cols:
    print 'reading whash...'
    df_whash = pd.read_csv('features_whash_test.csv', usecols=whash_cols, dtype=np.float32)
else:
    df_whash = pd.DataFrame()

print 'reading dimred...'
df_dimred = pd.read_csv('features_dimred_test.csv', usecols=dimred_cols, dtype=np.float32)

if ssim_cols:
    print 'reading ssim...'
    df_ssim = pd.read_csv('features_ssim_test.csv', usecols=ssim_cols, dtype=np.float32)
else:
    df_ssim = pd.DataFrame()

print 'reading misspelling...'
df_misspellings = pd.read_csv('features_misspellings_test.csv', usecols=misspelling_cols, dtype=np.float32)
df_misspellings.fillna(0, inplace=1)

print 'reading imp_tokens...'
df_imp_tokens = pd.read_csv('features_important_tokens_test.csv', usecols=imp_tokens_cols, dtype=np.float32)


df_test  = pd.concat([df_diffs_chargrams, df_fuzzy, df_imagemagick, df_imagehash, 
                      df_java, df_w2v, df_glove, df_attrs,
                      df_w2v_wmd, df_text, df_common_tokens, df_whash,
                      df_dimred, df_ssim, df_misspellings, df_imp_tokens], axis=1)
df_test.fillna(-999, inplace=1)

# In[35]:

del df_diffs_chargrams, df_fuzzy, df_imagemagick, df_imagehash
del df_java, df_w2v, df_glove, df_attrs
del df_w2v_wmd, df_text, df_common_tokens, df_whash
del df_dimred, df_ssim, df_misspellings, df_imp_tokens


# In[41]:
X_test = df_test[features].values
del df_test

X_test[np.isposinf(X_test)] =  10000
X_test[np.isneginf(X_test)] = -10000



# In[46]:
ids = pd.read_csv('../input/ItemPairs_test.csv', usecols=['id'], dtype=np.int)
test_ids = ids.id.values

y_result = et_full.predict_proba(X_test)[:, 1]


# In[58]:
submission = pd.DataFrame({'id': test_ids, 'probability': y_result})
submission.to_csv(filename + '_test.csv.gz', index=False, compression='gzip')
