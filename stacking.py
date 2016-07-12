
# coding: utf-8

# In[18]:

from glob import glob
from time import time

from tqdm import tqdm

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

import xgboost as xgb

from sklearn.cross_validation import KFold



df_train = pd.read_csv('../input/ItemPairs_train.csv', usecols=['isDuplicate'], dtype=np.uint8)
df_test = pd.read_csv('../input/ItemPairs_test.csv', usecols=['id'], dtype=np.uint32)


data = sorted(glob('models*/*.csv*'))

trains = []
tests = []

names = []

for i in tqdm(range(0, len(data), 2)):
    name = data[i + 1]
    folder = name[:name.find('/')]
    name = name[name.find('/')+1:name.rfind('_')]
    names.append(name)
    
    train = pd.read_csv(data[i + 1], usecols=['probability'], dtype=np.float32)
    train.columns = [name]
    trains.append(train)

    test = pd.read_csv(data[i], usecols=['probability'], dtype=np.float32)
    test.columns = [name]
    tests.append(test)


trains = pd.concat(trains, axis=1)
tests = pd.concat(tests, axis=1)


y_full = df_train.isDuplicate.values
X = trains[names].values
X_test = tests[names].values


lr = LogisticRegression(random_state=1, penalty='l2', C=0.001)
lr.fit(X, y_full)
y_pred = lr.predict_proba(X_test)[:, 1]


test_ids = df_test.id.values
submission = pd.DataFrame({'id': test_ids, 'probability': y_pred})
submission.to_csv('stacking.csv.gz', index=False, compression='gzip')
