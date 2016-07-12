
# coding: utf-8

# In[1]:

import os

import pandas as pd
import numpy as np
import scipy

from time import time
from natsort import natsorted

from tqdm import tqdm

from sklearn.svm import LinearSVC
from sklearn.cross_validation import KFold
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# In[2]:

y_full = pd.read_csv('../input/ItemPairs_train.csv', usecols=['isDuplicate'], dtype=np.int16).isDuplicate.values


# In[3]:

cv = KFold(len(y_full), n_folds=4, shuffle=True, random_state=42)

def sigmoid(pred):
    return 1 / (1 + np.exp(-pred))


# In[4]:

df_res_train = pd.DataFrame()
df_res_test =  pd.DataFrame()


# ## Fuzzy features

# In[5]:

df = pd.read_csv('features_fuzzy_train.csv', nrows=150000, dtype=np.float32)
cols_to_drop = ['isDuplicate', 'generationMethod']
df.drop(cols_to_drop, axis=1, inplace=1)
columns = list(df.columns)


# In[6]:

scaler = StandardScaler(with_std=False)
scaled = scaler.fit_transform(df)

pca = RandomizedPCA(n_components=15, random_state=1)
pca.fit(scaled)

pca.explained_variance_ratio_.cumsum()[-1]


# In[7]:

df_fuzzy = pd.read_csv('features_fuzzy_train.csv', usecols=columns,dtype=np.float32)
scaled = scaler.transform(df_fuzzy)
X = pca.transform(scaled)


# In[8]:

df_fuzzy_test = pd.read_csv('features_fuzzy_test.csv', usecols=columns,dtype=np.float32)
scaled = scaler.transform(df_fuzzy_test)
X_test = pca.transform(scaled)


# In[9]:

for i in range(5):
    df_res_train['pca_fuzzy_%d' % i] = X[:, i]
    df_res_test['pca_fuzzy_%d' % i] = X_test[:, i]


# In[10]:

scores = []

train_preds = np.zeros(y_full.shape)

for train, val in cv:
    t0 = time()
    svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.01)
    svm.fit(X[train], y_full[train])

    y_pred = svm.decision_function(X[val])
    y_pred = sigmoid(y_pred)
    train_preds[val] = y_pred

    score = roc_auc_score(y_full[val], y_pred)
    print score,
    print 'fit finished in %0.2fs' % (time() - t0)
    scores.append(score)

print np.mean(scores)


# In[11]:

t0 = time() 

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.01)
svm.fit(X, y_full)
y_pred = svm.decision_function(X_test)
y_pred = sigmoid(y_pred)

print 'fit finished in %0.2fs' % (time() - t0)


# In[12]:

df_res_train['svm_pca_fuzzy'] = train_preds
df_res_test['svm_pca_fuzzy'] = y_pred


# In[13]:

del df_fuzzy, df_fuzzy_test


# ## Image Hash

# In[14]:

df = pd.read_csv('features_imagehash_train.csv', nrows=150000, dtype=np.float32)
cols_to_drop = ['isDuplicate', 'generationMethod']
df.drop(cols_to_drop, axis=1, inplace=1)
columns = list(df.columns)


# In[15]:

df.isnull().sum(axis=1).unique()


# In[16]:

df = df.dropna(how='all')


# In[17]:

scaler = StandardScaler(with_std=True)
scaled = scaler.fit_transform(df)

pca = RandomizedPCA(n_components=28, random_state=1)
pca.fit(scaled)

pca.explained_variance_ratio_.cumsum()[-1]


# In[18]:

df = pd.read_csv('features_imagehash_train.csv', usecols=columns, dtype=np.float32)
nulls = df.ahash_abs_entropy_diff_25p.isnull()


# In[19]:

scaled = scaler.transform(df[~nulls])
X = pca.transform(scaled)


# In[20]:

df_imhash_train = pd.DataFrame(index=nulls.index)
_, n_features = X.shape

for i in range(n_features):
    df_imhash_train['pca_%d' % i] = 0
    df_imhash_train.loc[~nulls, 'pca_%d' % i] = X[:, i]
df_imhash_train['is_null'] = nulls.astype(int)

for i in range(10):
    df_res_train['pca_imhash_%d' % i] = np.nan
    df_res_train.loc[~nulls, 'pca_imhash_%d' % i] = X[:, i]


# In[21]:

X = df_imhash_train.values


# In[22]:

scores = []

train_preds = np.zeros(y_full.shape)

for train, val in cv:
    t0 = time()
    svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.001)
    svm.fit(X[train], y_full[train])

    y_pred = svm.decision_function(X[val])
    y_pred = sigmoid(y_pred)
    train_preds[val] = y_pred

    score = roc_auc_score(y_full[val], y_pred)
    print score,
    print 'fit finished in %0.2fs' % (time() - t0)
    scores.append(score)

print np.mean(scores)


# In[23]:

df_res_train['svm_pca_imhash'] = train_preds


# In[24]:

t0 = time() 

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.001)
svm.fit(X, y_full)

time() - t0


# In[25]:

df = pd.read_csv('features_imagehash_test.csv', usecols=columns, dtype=np.float32)
nulls = df.ahash_abs_entropy_diff_25p.isnull()


# In[26]:

scaled = scaler.transform(df[~nulls])
X_test = pca.transform(scaled)


# In[27]:

df_imhash_test = pd.DataFrame(index=nulls.index)
_, n_features = X_test.shape

for i in range(n_features):
    df_imhash_test['pca_%d' % i] = 0
    df_imhash_test.loc[~nulls, 'pca_%d' % i] = X_test[:, i]

df_imhash_test['is_null'] = nulls.astype(int)

for i in range(10):
    df_res_test['pca_imhash_%d' % i] = np.nan
    df_res_test.loc[~nulls, 'pca_imhash_%d' % i] = X_test[:, i]


# In[28]:

X_test = df_imhash_test.values
y_pred = svm.decision_function(X_test)
y_pred = sigmoid(y_pred)


# In[29]:

df_res_test['svm_pca_imhash'] = y_pred


# In[30]:

del df_imhash_test, df_imhash_train


# In[31]:

df_res_train.head()


# In[32]:

df_res_test.head()


# In[33]:

df_res_train[['svm_pca_fuzzy', 'svm_pca_imhash']].corr()


# ## Imagemagick

# In[34]:

df = pd.read_csv('features_imagemagick_train.csv', nrows=150000, dtype=np.float32)
columns = list(df.columns)

df[df > 1e5] = np.nan

nulls = df.isnull()
df.fillna(0, inplace=1)


# In[35]:

pca_nulls = RandomizedPCA(n_components=2, random_state=1)
pca_nulls.fit(nulls)

pca_nulls.explained_variance_ratio_.cumsum()[-1]


# In[36]:

scaler = StandardScaler(with_std=True)
scaled = scaler.fit_transform(df)

pca = RandomizedPCA(n_components=60, random_state=1)
pca.fit(scaled)

pca.explained_variance_ratio_.cumsum()[-1]


# In[37]:

X = []
X_test = []

df_iter = pd.read_csv('features_imagemagick_train.csv', usecols=columns, 
                      dtype=np.float32, iterator=True, chunksize=100000)

for df in tqdm(df_iter):
    df[df > 1e5] = np.nan
    nulls = df.isnull()
    df.fillna(0, inplace=1)
    
    X_nulls = pca_nulls.transform(nulls)
    
    scaled = scaler.transform(df)
    X_it = pca.transform(scaled)
    X.append(np.hstack([X_it, X_nulls]))

df_iter = pd.read_csv('features_imagemagick_test.csv', usecols=columns, 
                      dtype=np.float32, iterator=True, chunksize=100000)

for df in tqdm(df_iter):
    df[df > 1e5] = np.nan
    nulls = df.isnull()
    df.fillna(0, inplace=1)
    
    X_nulls = pca_nulls.transform(nulls)
    
    scaled = scaler.transform(df)
    X_it = pca.transform(scaled)
    X_test.append(np.hstack([X_it, X_nulls]))


# In[ ]:

X = np.vstack(X)
X_test = np.vstack(X_test)


# In[ ]:

scores = []

train_preds = np.zeros(y_full.shape)

for train, val in cv:
    t0 = time()
    svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.0001)
    svm.fit(X[train], y_full[train])

    y_pred = svm.decision_function(X[val])
    y_pred = sigmoid(y_pred)
    train_preds[val] = y_pred

    score = roc_auc_score(y_full[val], y_pred)
    print score,
    print 'fit finished in %0.2fs' % (time() - t0)
    scores.append(score)

print np.mean(scores)


# In[ ]:

t0 = time() 

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.0001)
svm.fit(X, y_full)
y_pred = svm.decision_function(X_test)
y_pred = sigmoid(y_pred)

print 'fit finished in %0.2fs' % (time() - t0)


# In[ ]:

for i in range(10):
    df_res_train['pca_imagemagick_%d' % i] = X[:, i]
    df_res_test['pca_imagemagick_%d' % i] = X_test[:, i]

df_res_train['svm_pca_imagemagick'] = train_preds
df_res_test['svm_pca_imagemagick'] = y_pred

df_res_train['pca_imagemagick_na_1'] = X[:, -2]
df_res_test['pca_imagemagick_na_1'] = X_test[:, -2]


# In[67]:

df_res_train[['svm_pca_fuzzy', 'svm_pca_imhash', 'svm_pca_imagemagick']].corr()


# ## Java image features

# In[ ]:

df = pd.read_csv('features_java-image_train.csv', nrows=150000)
cols_to_drop = [u'itemID_1', u'itemID_2', u'isDuplicate', u'generationMethod', 
               'kp_diff_25p','kp_diff_75p','kp_diff_kurtosis','kp_diff_max','kp_diff_mean',
               'kp_diff_min','kp_diff_skew','kp_diff_std','kp_matched_25p','kp_matched_75p',
               'kp_matched_kurtosis','kp_matched_max','kp_matched_mean','kp_matched_min',
               'kp_matched_skew','kp_matched_std']
df.drop(cols_to_drop, axis=1, inplace=1)

columns = list(df.columns)


# In[ ]:

df.isnull().sum(axis=1).unique()


# In[ ]:

df = df.dropna(how='all')


# In[ ]:

scaler = StandardScaler(with_std=True)
scaled = scaler.fit_transform(df)

pca = RandomizedPCA(n_components=16, random_state=1)
pca.fit(scaled)

pca.explained_variance_ratio_.cumsum()[-1]


# In[ ]:

X = []

df_iter = pd.read_csv('features_java-image_train.csv', usecols=columns, 
                      dtype=np.float32, iterator=True, chunksize=100000)

for df in tqdm(df_iter):
    nulls = df.hist_bhattacharyya_25p.isnull().values

    X_init = np.empty((len(df), 16), dtype=np.float32)
    X_init[:] = np.nan

    df = df.dropna(how='all')
    scaled = scaler.transform(df)
    X_init[~nulls, :] = pca.transform(scaled)
    X.append(np.hstack([X_init, nulls.reshape(-1, 1)]))


# In[ ]:

X_test = []

df_iter = pd.read_csv('features_java-image_test.csv', usecols=columns, 
                      dtype=np.float32, iterator=True, chunksize=100000)

for df in tqdm(df_iter):
    nulls = df.hist_bhattacharyya_25p.isnull().values

    X_init = np.empty((len(df), 16), dtype=np.float32)
    X_init[:] = np.nan

    df = df.dropna(how='all')
    scaled = scaler.transform(df)
    X_init[~nulls, :] = pca.transform(scaled)
    X_test.append(np.hstack([X_init, nulls.reshape(-1, 1)]))


# In[ ]:

X = np.vstack(X)
X_test = np.vstack(X_test)


# In[ ]:

for i in range(7):
    df_res_train['pca_javaimage_%d' % i] = X[:, i]
    df_res_test['pca_javaimage_%d' % i] = X_test[:, i]


# In[ ]:

X[np.isnan(X)] = 0
X_test[np.isnan(X_test)] = 0


# In[ ]:

scores = []

train_preds = np.zeros(y_full.shape)

for train, val in cv:
    t0 = time()
    svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.001)
    svm.fit(X[train], y_full[train])

    y_pred = svm.decision_function(X[val])
    y_pred = sigmoid(y_pred)
    train_preds[val] = y_pred

    score = roc_auc_score(y_full[val], y_pred)
    print score,
    print 'fit finished in %0.2fs' % (time() - t0)
    scores.append(score)

print np.mean(scores)


# In[ ]:

t0 = time() 

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.001)
svm.fit(X, y_full)
y_pred = svm.decision_function(X_test)
y_pred = sigmoid(y_pred)

print 'fit finished in %0.2fs' % (time() - t0)


# In[ ]:

df_res_train['svm_pca_javaimage'] = train_preds
df_res_test['svm_pca_javaimage'] = y_pred


# In[68]:

df_res_train.columns == df_res_test.columns


# ## Word2Vec

# In[ ]:

df = pd.read_csv('features_w2v_train.csv', nrows=150000)
columns = list(df.columns)


# In[ ]:

nulls = df.isnull()
pca_nulls = RandomizedPCA(n_components=5, random_state=1)
pca_nulls.fit(nulls)

pca_nulls.explained_variance_ratio_.cumsum()[-1]


# In[ ]:

df.fillna(0, inplace=1)
scaler = StandardScaler(with_std=True)
scaled = scaler.fit_transform(df)

pca = RandomizedPCA(n_components=45, random_state=1)
pca.fit(scaled)

pca.explained_variance_ratio_.cumsum()[-1]


# In[ ]:

X = []
X_test = []

df_iter = pd.read_csv('features_w2v_train.csv', usecols=columns, 
                      dtype=np.float32, iterator=True, chunksize=100000)

for df in tqdm(df_iter):
    nulls = df.isnull()
    df.fillna(0, inplace=1)
    
    X_nulls = pca_nulls.transform(nulls)
    
    scaled = scaler.transform(df)
    X_it = pca.transform(scaled)
    X.append(np.hstack([X_it, X_nulls]))

df_iter = pd.read_csv('features_w2v_test.csv', usecols=columns, 
                      dtype=np.float32, iterator=True, chunksize=100000)

for df in tqdm(df_iter):
    nulls = df.isnull()
    df.fillna(0, inplace=1)
    
    X_nulls = pca_nulls.transform(nulls)
    
    scaled = scaler.transform(df)
    X_it = pca.transform(scaled)
    X_test.append(np.hstack([X_it, X_nulls]))


# In[ ]:

X = np.vstack(X)
X_test = np.vstack(X_test)


# In[ ]:

scores = []

train_preds = np.zeros(y_full.shape)

for train, val in cv:
    t0 = time()
    svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.001)
    svm.fit(X[train], y_full[train])

    y_pred = svm.decision_function(X[val])
    y_pred = sigmoid(y_pred)
    train_preds[val] = y_pred

    score = roc_auc_score(y_full[val], y_pred)
    print score,
    print 'fit finished in %0.2fs' % (time() - t0)
    scores.append(score)

print np.mean(scores)


# In[ ]:

t0 = time() 

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.001)
svm.fit(X, y_full)
y_pred = svm.decision_function(X_test)
y_pred = sigmoid(y_pred)

print 'fit finished in %0.2fs' % (time() - t0)


# In[ ]:

for i in range(10):
    df_res_train['pca_w2v_%d' % i] = X[:, i]
    df_res_test['pca_w2v_%d' % i] = X_test[:, i]

for i in range(2):
    df_res_train['pca_w2v_na_%d' % i] = X[:, -5 + i]
    df_res_test['pca_w2v_na_%d' % i] = X_test[:, -5 + i]

df_res_train['svm_pca_w2v'] = train_preds
df_res_test['svm_pca_w2v'] = y_pred


# In[70]:

df_res_train.columns == df_res_test.columns


# ## "Important" Tokens

# In[72]:

df = pd.read_csv('features_important_tokens_train.csv', nrows=150000)
columns = list(df.columns)


# In[73]:

df.isnull().sum().sum()


# In[74]:

scaler = StandardScaler(with_std=True)
scaled = scaler.fit_transform(df)

pca = RandomizedPCA(n_components=12, random_state=1)
pca.fit(scaled)

pca.explained_variance_ratio_.cumsum()[-1]


# In[76]:

X = []
X_test = []

df_iter = pd.read_csv('features_important_tokens_train.csv', usecols=columns, 
                      dtype=np.float32, iterator=True, chunksize=100000)

for df in tqdm(df_iter):
    scaled = scaler.transform(df)
    X_it = pca.transform(scaled)
    X.append(X_it)


df_iter = pd.read_csv('features_important_tokens_test.csv', usecols=columns, 
                      dtype=np.float32, iterator=True, chunksize=100000)

for df in tqdm(df_iter):
    scaled = scaler.transform(df)
    X_it = pca.transform(scaled)
    X_test.append(X_it)


# In[77]:

X = np.vstack(X)
X_test = np.vstack(X_test)


# In[81]:

scores = []

train_preds = np.zeros(y_full.shape)

for train, val in cv:
    t0 = time()
    svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.001)
    svm.fit(X[train], y_full[train])

    y_pred = svm.decision_function(X[val])
    y_pred = sigmoid(y_pred)
    train_preds[val] = y_pred

    score = roc_auc_score(y_full[val], y_pred)
    print score,
    print 'fit finished in %0.2fs' % (time() - t0)
    scores.append(score)

print np.mean(scores)


# In[82]:

t0 = time() 

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.001)
svm.fit(X, y_full)
y_pred = svm.decision_function(X_test)
y_pred = sigmoid(y_pred)

print 'fit finished in %0.2fs' % (time() - t0)


# In[83]:

df_res_train['svm_pca_imptokens'] = train_preds
df_res_test['svm_pca_imptokens'] = y_pred


# ## Misspellings

# In[84]:

df = pd.read_csv('features_misspellings_train.csv', nrows=150000, dtype=np.float32)
columns = list(df.columns)


# In[87]:

df.fillna(0, inplace=1)


# In[91]:

scaler = StandardScaler(with_std=True)
scaled = scaler.fit_transform(df)

pca = RandomizedPCA(n_components=30, random_state=1)
pca.fit(scaled)

pca.explained_variance_ratio_.cumsum()[-1]


# In[92]:

X = []
X_test = []

df_iter = pd.read_csv('features_misspellings_train.csv', usecols=columns, 
                      dtype=np.float32, iterator=True, chunksize=100000)

for df in tqdm(df_iter):
    df.fillna(0, inplace=1)
    scaled = scaler.transform(df)
    X_it = pca.transform(scaled)
    X.append(X_it)


df_iter = pd.read_csv('features_misspellings_test.csv', usecols=columns, 
                      dtype=np.float32, iterator=True, chunksize=100000)

for df in tqdm(df_iter):
    df.fillna(0, inplace=1)
    scaled = scaler.transform(df)
    X_it = pca.transform(scaled)
    X_test.append(X_it)


# In[93]:

X = np.vstack(X)
X_test = np.vstack(X_test)


# In[98]:

scores = []

train_preds = np.zeros(y_full.shape)

for train, val in cv:
    t0 = time()
    svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.01)
    svm.fit(X[train], y_full[train])

    y_pred = svm.decision_function(X[val])
    y_pred = sigmoid(y_pred)
    train_preds[val] = y_pred

    score = roc_auc_score(y_full[val], y_pred)
    print score,
    print 'fit finished in %0.2fs' % (time() - t0)
    scores.append(score)

print np.mean(scores)


# In[99]:

t0 = time() 

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.01)
svm.fit(X, y_full)
y_pred = svm.decision_function(X_test)
y_pred = sigmoid(y_pred)

print 'fit finished in %0.2fs' % (time() - t0)


# In[100]:

df_res_train['svm_pca_misspellings'] = train_preds
df_res_test['svm_pca_misspellings'] = y_pred


# ## Word Mover Distance

# In[101]:

df = pd.read_csv('features_w2v_wmd_train.csv', nrows=150000, dtype=np.float32)
columns = list(df.columns)


# In[108]:

nulls = df.isnull()
pca_nulls = RandomizedPCA(n_components=5, random_state=1)
pca_nulls.fit(nulls)

pca_nulls.explained_variance_ratio_.cumsum()[-1]


# In[109]:

df.fillna(0, inplace=1)


# In[119]:

scaler = StandardScaler(with_std=True)
scaled = scaler.fit_transform(df)

pca = RandomizedPCA(n_components=10, random_state=1)
pca.fit(scaled)

pca.explained_variance_ratio_.cumsum()[-1]


# In[120]:

X = []
X_test = []

df_iter = pd.read_csv('features_w2v_wmd_train.csv', usecols=columns, 
                      dtype=np.float32, iterator=True, chunksize=100000)

for df in tqdm(df_iter):
    nulls = df.isnull()
    df.fillna(0, inplace=1)
    
    X_nulls = pca_nulls.transform(nulls)
    
    scaled = scaler.transform(df)
    X_it = pca.transform(scaled)
    X.append(np.hstack([X_it, X_nulls]))

df_iter = pd.read_csv('features_w2v_wmd_test.csv', usecols=columns, 
                      dtype=np.float32, iterator=True, chunksize=100000)

for df in tqdm(df_iter):
    nulls = df.isnull()
    df.fillna(0, inplace=1)
    
    X_nulls = pca_nulls.transform(nulls)
    
    scaled = scaler.transform(df)
    X_it = pca.transform(scaled)
    X_test.append(np.hstack([X_it, X_nulls]))


# In[121]:

X = np.vstack(X)
X_test = np.vstack(X_test)


# In[124]:

scores = []

train_preds = np.zeros(y_full.shape)

for train, val in cv:
    t0 = time()
    svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.01)
    svm.fit(X[train], y_full[train])

    y_pred = svm.decision_function(X[val])
    y_pred = sigmoid(y_pred)
    train_preds[val] = y_pred

    score = roc_auc_score(y_full[val], y_pred)
    print score,
    print 'fit finished in %0.2fs' % (time() - t0)
    scores.append(score)

print np.mean(scores)


# In[125]:

t0 = time() 

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.01)
svm.fit(X, y_full)
y_pred = svm.decision_function(X_test)
y_pred = sigmoid(y_pred)

print 'fit finished in %0.2fs' % (time() - t0)


# In[126]:

df_res_train['svm_pca_w2v_wmd'] = train_preds
df_res_test['svm_pca_w2v_wmd'] = y_pred


# ## Text features

# In[127]:

df = pd.read_csv('features_text_train.csv', nrows=150000, dtype=np.float32)
columns = list(df.columns)


# In[128]:

df.isnull().sum().sum()


# In[141]:

scaler = StandardScaler(with_std=True)
scaled = scaler.fit_transform(df)

pca = RandomizedPCA(n_components=45, random_state=1)
pca.fit(scaled)

pca.explained_variance_ratio_.cumsum()[-1]


# In[142]:

X = []
X_test = []

df_iter = pd.read_csv('features_text_train.csv', usecols=columns, 
                      dtype=np.float32, iterator=True, chunksize=100000)

for df in tqdm(df_iter):
    df.fillna(0, inplace=1)
    scaled = scaler.transform(df)
    X_it = pca.transform(scaled)
    X.append(X_it)


df_iter = pd.read_csv('features_text_test.csv', usecols=columns, 
                      dtype=np.float32, iterator=True, chunksize=100000)

for df in tqdm(df_iter):
    df.fillna(0, inplace=1)
    scaled = scaler.transform(df)
    X_it = pca.transform(scaled)
    X_test.append(X_it)


# In[144]:

X = np.vstack(X)
X_test = np.vstack(X_test)


# In[146]:

scores = []

train_preds = np.zeros(y_full.shape)

for train, val in cv:
    t0 = time()
    svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.001)
    svm.fit(X[train], y_full[train])

    y_pred = svm.decision_function(X[val])
    y_pred = sigmoid(y_pred)
    train_preds[val] = y_pred

    score = roc_auc_score(y_full[val], y_pred)
    print score,
    print 'fit finished in %0.2fs' % (time() - t0)
    scores.append(score)

print np.mean(scores)


# In[148]:

t0 = time() 

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.001)
svm.fit(X, y_full)
y_pred = svm.decision_function(X_test)
y_pred = sigmoid(y_pred)

print 'fit finished in %0.2fs' % (time() - t0)


# In[149]:

df_res_train['svm_pca_text'] = train_preds
df_res_test['svm_pca_text'] = y_pred


# In[150]:

for i in range(5):
    df_res_train['pca_text_%d' % i] = X[:, i]
    df_res_test['pca_text_%d' % i] = X_test[:, i]


# ## GloVe

# In[151]:

df = pd.read_csv('features_glove_train.csv', nrows=150000, dtype=np.float32)
columns = list(df.columns)


# In[154]:

nulls = df.isnull()
pca_nulls = RandomizedPCA(n_components=5, random_state=1)
pca_nulls.fit(nulls)

pca_nulls.explained_variance_ratio_.cumsum()[-1]


# In[155]:

df.fillna(0, inplace=1)


# In[156]:

scaler = StandardScaler(with_std=True)
scaled = scaler.fit_transform(df)

pca = RandomizedPCA(n_components=45, random_state=1)
pca.fit(scaled)

pca.explained_variance_ratio_.cumsum()[-1]


# In[157]:

X = []
X_test = []

df_iter = pd.read_csv('features_glove_train.csv', usecols=columns, 
                      dtype=np.float32, iterator=True, chunksize=100000)

for df in tqdm(df_iter):
    nulls = df.isnull()
    df.fillna(0, inplace=1)
    
    X_nulls = pca_nulls.transform(nulls)
    
    scaled = scaler.transform(df)
    X_it = pca.transform(scaled)
    X.append(np.hstack([X_it, X_nulls]))

df_iter = pd.read_csv('features_glove_test.csv', usecols=columns, 
                      dtype=np.float32, iterator=True, chunksize=100000)

for df in tqdm(df_iter):
    nulls = df.isnull()
    df.fillna(0, inplace=1)
    
    X_nulls = pca_nulls.transform(nulls)
    
    scaled = scaler.transform(df)
    X_it = pca.transform(scaled)
    X_test.append(np.hstack([X_it, X_nulls]))


# In[159]:

X = np.vstack(X)
X_test = np.vstack(X_test)


# In[160]:

scores = []

train_preds = np.zeros(y_full.shape)

for train, val in cv:
    t0 = time()
    svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.001)
    svm.fit(X[train], y_full[train])

    y_pred = svm.decision_function(X[val])
    y_pred = sigmoid(y_pred)
    train_preds[val] = y_pred

    score = roc_auc_score(y_full[val], y_pred)
    print score,
    print 'fit finished in %0.2fs' % (time() - t0)
    scores.append(score)

print np.mean(scores)


# In[161]:

t0 = time() 

svm = LinearSVC(penalty='l1', dual=False, random_state=1, C=0.001)
svm.fit(X, y_full)
y_pred = svm.decision_function(X_test)
y_pred = sigmoid(y_pred)

print 'fit finished in %0.2fs' % (time() - t0)


# In[ ]:

for i in range(10):
    df_res_train['pca_glove_%d' % i] = X[:, i]
    df_res_test['pca_glove_%d' % i] = X_test[:, i]

for i in range(2):
    df_res_train['pca_glove_na_%d' % i] = X[:, -5 + i]
    df_res_test['pca_glove_na_%d' % i] = X_test[:, -5 + i]

df_res_train['svm_pca_glove'] = train_preds
df_res_test['svm_pca_glove'] = y_pred


# Let's have a look at correlation of feature learved with SVM

# In[170]:

svm_cols = df_res_train.columns[df_res_train.columns.str.startswith('svm')]
df_res_train[svm_cols].corr()


# ## Save everything

# In[171]:

df_res_train.to_csv('features_dimred_train.csv', index=False)
df_res_test.to_csv('features_dimred_test.csv', index=False)

