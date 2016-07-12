
# coding: utf-8

# In[74]:

from time import time

import numpy as np
import scipy.sparse as sp
import pandas as pd

from tqdm import tqdm

import avito_utils


# In[14]:

from sklearn.decomposition import TruncatedSVD

from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score

from sklearn.svm import LinearSVC


# In[2]:

df_train = pd.read_csv('common_diff_tokens_train.csv')
df_train.fillna('', inplace=1)
y_full = df_train.isDuplicate.values
df_train.drop('isDuplicate', axis=1, inplace=1)

df_test = pd.read_csv('common_diff_tokens_test.csv')
df_test.fillna('', inplace=1)


# In[44]:

df_train_res = pd.DataFrame(index=df_train.index)
df_test_res = pd.DataFrame(index=df_test.index)


# In[122]:

cv = KFold(len(y_full), n_folds=4, shuffle=True, random_state=42)

def sigmoid(pred):
    return 1 / (1 + np.exp(-pred))


# ## SVM features

# Fit terms that are in both titles

# In[123]:

cnt_title_common = CountVectorizer(min_df=5, analyzer=str.split, dtype=np.uint8)
title_common = cnt_title_common.fit_transform(df_train.title_common)


# In[124]:

scores = []

train_preds = np.zeros(y_full.shape)
X = title_common

for train, val in cv:
    t0 = time()
    svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.3)
    svm.fit(X[train], y_full[train])

    y_pred = svm.decision_function(X[val])
    y_pred = sigmoid(y_pred)
    train_preds[val] = y_pred

    score = roc_auc_score(y_full[val], y_pred)
    print score,
    print 'fit finished in %0.2fs' % (time() - t0)
    scores.append(score)

df_train_res['svm_title_common'] = train_preds
print np.mean(scores)


# In[125]:

t0 = time()

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.3)
svm.fit(X, y_full)

title_common_test = cnt_title_common.transform(df_test.title_common)
y_pred = svm.decision_function(title_common_test)
y_pred = sigmoid(y_pred)

df_test_res['svm_title_common'] = y_pred

print 'full fit finished in %0.2fs' % (time() - t0)


# In[126]:

cnt_title_diff = CountVectorizer(min_df=5, analyzer=str.split, dtype=np.uint8)
title_diff = cnt_title_diff.fit_transform(df_train.title_diff)


# Fit terms that are either in one title, or another, but not in both

# In[127]:

scores = []

train_preds = np.zeros(y_full.shape)
X = title_diff

for train, val in cv:
    t0 = time()
    svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.4)
    svm.fit(X[train], y_full[train])

    y_pred = svm.decision_function(X[val])
    y_pred = sigmoid(y_pred)
    train_preds[val] = y_pred

    score = roc_auc_score(y_full[val], y_pred)
    print score,
    print 'fit finished in %0.2fs' % (time() - t0)
    scores.append(score)

df_train_res['svm_title_diff'] = train_preds
print np.mean(scores)


# In[129]:

t0 = time()

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.4)
svm.fit(X, y_full)

title_diff_test = cnt_title_diff.transform(df_test.title_diff)
y_pred = svm.decision_function(title_diff_test)
y_pred = sigmoid(y_pred)

df_test_res['svm_title_diff'] = y_pred

print 'full fit finished in %0.2fs' % (time() - t0)


# Fit terms that are common or in either

# In[130]:

title_both = sp.hstack([title_common, title_diff], format='csr')


# In[131]:

scores = []

train_preds = np.zeros(y_full.shape)
X = title_all

for train, val in cv:
    t0 = time()
    svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.4)
    svm.fit(X[train], y_full[train])

    y_pred = svm.decision_function(X[val])
    y_pred = sigmoid(y_pred)
    train_preds[val] = y_pred

    score = roc_auc_score(y_full[val], y_pred)
    print score,
    print 'fit finished in %0.2fs' % (time() - t0)
    scores.append(score)

df_train_res['svm_title_both'] = train_preds
print np.mean(scores)


# In[132]:

t0 = time()

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.4)
svm.fit(X, y_full)

X_test = sp.hstack([title_common_test, title_diff_test], format='csr')
y_pred = svm.decision_function(X_test)
y_pred = sigmoid(y_pred)

df_test_res['svm_title_both'] = y_pred

print 'full fit finished in %0.2fs' % (time() - t0)


# Fit terms that are common in description

# In[133]:

cnt_desc_common = CountVectorizer(min_df=5, analyzer=str.split, dtype=np.uint8)
desc_common = cnt_desc_common.fit_transform(df_train.desc_common)


# In[134]:

scores = []

train_preds = np.zeros(y_full.shape)
X = desc_common

for train, val in cv:
    t0 = time()
    svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.2)
    svm.fit(X[train], y_full[train])

    y_pred = svm.decision_function(X[val])
    y_pred = sigmoid(y_pred)
    train_preds[val] = y_pred

    score = roc_auc_score(y_full[val], y_pred)
    print score,
    print 'fit finished in %0.2fs' % (time() - t0)
    scores.append(score)

df_train_res['svm_desc_common'] = train_preds
print np.mean(scores)


# In[135]:

t0 = time()

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.2)
svm.fit(X, y_full)

desc_common_test = cnt_desc_common.transform(df_test.desc_common)
y_pred = svm.decision_function(desc_common_test)
y_pred = sigmoid(y_pred)

df_test_res['svm_desc_common'] = y_pred

print 'full fit finished in %0.2fs' % (time() - t0)


# Fit terms that are in either desc, but not in both

# In[136]:

cnt_desc_diff = CountVectorizer(min_df=5, analyzer=str.split, dtype=np.uint8)
desc_diff = cnt_desc_diff.fit_transform(df_train.desc_diff)


# In[137]:

scores = []

train_preds = np.zeros(y_full.shape)
X = desc_diff

for train, val in cv:
    t0 = time()
    svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.2)
    svm.fit(X[train], y_full[train])

    y_pred = svm.decision_function(X[val])
    y_pred = sigmoid(y_pred)
    train_preds[val] = y_pred

    score = roc_auc_score(y_full[val], y_pred)
    print score,
    print 'fit finished in %0.2fs' % (time() - t0)
    scores.append(score)

df_train_res['svm_desc_diff'] = train_preds
print np.mean(scores)


# In[138]:

t0 = time()

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.2)
svm.fit(X, y_full)

desc_diff_test = cnt_desc_diff.transform(df_test.desc_diff)
y_pred = svm.decision_function(desc_diff_test)
y_pred = sigmoid(y_pred)

df_test_res['svm_desc_diff'] = y_pred

print 'full fit finished in %0.2fs' % (time() - t0)


# Fit description terms that are common or in either

# In[139]:

desc_both = sp.hstack([desc_common, desc_diff], format='csr')
desc_both


# In[140]:

scores = []

train_preds = np.zeros(y_full.shape)
X = desc_both

for train, val in cv:
    t0 = time()
    svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.05)
    svm.fit(X[train], y_full[train])

    y_pred = svm.decision_function(X[val])
    y_pred = sigmoid(y_pred)
    train_preds[val] = y_pred

    score = roc_auc_score(y_full[val], y_pred)
    print score,
    print 'fit finished in %0.2fs' % (time() - t0)
    scores.append(score)

df_train_res['svm_desc_both'] = train_preds
print np.mean(scores)


# In[141]:

t0 = time()

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.05)
svm.fit(X, y_full)

X_test = sp.hstack([desc_common_test, desc_diff_test], format='csr')
y_pred = svm.decision_function(X_test)
y_pred = sigmoid(y_pred)

df_test_res['svm_desc_both'] = y_pred

print 'full fit finished in %0.2fs' % (time() - t0)


# Now use both title and desc

# In[ ]:

all_text = sp.hstack([title_common, title_diff, desc_common, desc_diff], format='csr')
all_text


# In[ ]:

scores = []

train_preds = np.zeros(y_full.shape)
X = all_text

for train, val in cv:
    t0 = time()
    svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.03)
    svm.fit(X[train], y_full[train])

    y_pred = svm.decision_function(X[val])
    y_pred = sigmoid(y_pred)
    train_preds[val] = y_pred

    score = roc_auc_score(y_full[val], y_pred)
    print score,
    print 'fit finished in %0.2fs' % (time() - t0)
    scores.append(score)

df_train_res['svm_all_text_common'] = train_preds
print np.mean(scores)


# In[ ]:

t0 = time()

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.03)
svm.fit(X, y_full)

X_test = sp.hstack([title_common_test, title_diff_test, desc_common_test, desc_diff_test], format='csr')
y_pred = svm.decision_function(X_test)
y_pred = sigmoid(y_pred)

df_test_res['svm_all_text_common'] = y_pred

print 'full fit finished in %0.2fs' % (time() - t0)


# In[ ]:

# df_train_res.to_csv('features_svm_tokens_train.csv', index=False)
# df_test_res.to_csv('features_svm_tokens_test.csv', index=False)


# ## SVD features

# In[172]:

def add_svd_features(X_train, X_test, n, name):
    print 'svd fit for %s' % name
    X = sp.vstack([X_train, X_test])
    t0 = time()

    svd = TruncatedSVD(n_components=n)
    svd.fit(X)

    print 'svd fit finished in %0.2fs' % (time() - t0)

    res_train = svd.transform(X_train)
    res_test = svd.transform(X_test)

    for i in range(n):
        col_name = 'svd_%s_%d' % (name, i)
        df_train_res[col_name] = res_train[:, i]
        df_test_res[col_name] = res_test[:, i]


# In[173]:

add_svd_features(X_train=title_common, X_test=title_common_test, n=5, name='title_common')
add_svd_features(X_train=title_diff, X_test=title_diff_test, n=5, name='title_diff')


# In[176]:

add_svd_features(X_train=desc_common, X_test=desc_common_test, n=7, name='desc_common')
add_svd_features(X_train=desc_diff, X_test=desc_diff_test, n=7, name='desc_diff')


# In[178]:

all_text_test = sp.hstack([title_common_test, title_diff_test, desc_common_test, desc_diff_test], format='csr')
add_svd_features(X_train=all_text, X_test=all_text_test, n=10, name='all_text')


# In[179]:

del title_common, title_common_test
del title_diff, title_diff_test
del desc_common, desc_common_test
del desc_diff, desc_diff_test
del title_both, desc_both
del all_text, all_text_test
del X, X_test


# In[181]:

del cnt_title_common, cnt_title_diff, cnt_desc_common, cnt_desc_diff


# ## Most important tokens

# In[185]:

cnt_all_common = CountVectorizer(min_df=5, analyzer=str.split, dtype=np.uint8)
all_common = cnt_all_common.fit_transform(df_train.all_common)
all_common_test = cnt_all_common.transform(df_test.all_common)


# In[200]:

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.0005)
svm.fit(all_common, y_full)


# In[206]:

coef = svm.coef_[0]
print (coef > 0).sum(), (coef > 0).mean()
print (coef < 0).sum(), (coef < 0).mean()


# In[218]:

def col_sum_mask(X, mask):
    res = X[:, mask].sum(axis=1)
    return np.array(res).reshape(-1)

df_train_res['pos_common_tokens_sum'] = col_sum_mask(all_common, coef > 0)
df_test_res['pos_common_tokens_sum'] = col_sum_mask(all_common_test, coef > 0)
df_train_res['neg_common_tokens_sum'] = col_sum_mask(all_common, coef < 0)
df_test_res['neg_common_tokens_sum'] = col_sum_mask(all_common_test, coef < 0)


# In[219]:

cnt_all_diff = CountVectorizer(min_df=5, analyzer=str.split, dtype=np.uint8)
all_diff = cnt_all_diff.fit_transform(df_train.all_diff)
all_diff_test = cnt_all_diff.transform(df_test.all_diff)


# In[220]:

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.0005)
svm.fit(all_diff, y_full)
coef = svm.coef_[0]
print (coef > 0).sum(), (coef > 0).mean()
print (coef < 0).sum(), (coef < 0).mean()


# In[221]:

df_train_res['pos_diff_tokens_sum'] = col_sum_mask(all_diff, coef > 0)
df_test_res['pos_diff_tokens_sum'] = col_sum_mask(all_diff_test, coef > 0)
df_train_res['neg_diff_tokens_sum'] = col_sum_mask(all_diff, coef < 0)
df_test_res['neg_diff_tokens_sum'] = col_sum_mask(all_diff_test, coef < 0)


# ## Save results

# In[ ]:

df_train_res.to_csv('features_common_tokens_train.csv', index=False)
df_test_res.to_csv('features_common_tokens_test.csv', index=False)


# In[ ]:

