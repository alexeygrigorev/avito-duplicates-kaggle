# -*- encoding: utf-8 -*-

from time import time
import os

from tqdm import tqdm

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, pearsonr

import avito_utils

from gensim.models import Word2Vec
mongo = avito_utils.mongo
    
def prepare_batches(df, n):
    for batch_no, i in enumerate(range(0, len(df), n)):
        yield batch_no, df.iloc[i:i+n]

        
stopwords = set(u'в,на,и,с,по,для,не,от,за,к,до,из,или,у,один,вы,при,так,ваш,как,а,наш,быть,под,б,'
                u'р,мы,эт,же,также,что,это,раз,свой,он,если,но,я,о,ещё,тот,этот,то,они,ни,чем,где,'
                u'бы,оно,там,ж,она,ты'.split(','))

columns = ['title_lemmatized', 'description_lemmatized', 'nouns']

model = None

def get_model():
    ''' lazy initialization for w2v model so it works in pool '''
    global model
    if model == None:
        print 'loading the w2v model...'
        model = Word2Vec.load('w2v/lemma_stopwords')
    return model

def tokens(sentences):
    result = []
    for s in sentences:
        result.extend(t for t in s.split() if t not in stopwords)
    return result

def row_normalize(M):
    norm2 = (M ** 2).sum(axis=1, keepdims=True)
    return M / np.sqrt(norm2)

def only_in_model(words):
    model = get_model()
    return {w for w in set(words) if w in model}

def distmat(X, C):    
    X2 = np.sum(X * X, axis=1, keepdims=True)
    C2 = np.sum(C * C, axis=1, keepdims=True)
    XC = np.dot(X, C.T)
    D = X2 - 2 * XC + C2.T
    return D

def manhattan(W1, W2):
    n, _ = W1.shape
    result = [np.abs(W1[[i], :] - W2).sum(axis=1) for i in range(n)]
    return np.array(result)

def word_mover_distances(D):
    D12 = D.min(axis=1)
    D21 = D.min(axis=0)
    symmetric_dist = (D12.sum() + D21.sum()) / 2
    mean_dist = np.concatenate([D12, D21]).mean()
    return symmetric_dist, mean_dist

def full_text_features(c1, c2, row, result):
    n1 = only_in_model(row[c1])
    n2 = only_in_model(row[c2])

    if not n1 or not n2:
        return

    n1_only = n1 - n2
    n2_only = n2 - n1

    if not n1_only or not n2_only:
        return 

    model = get_model()
    W1 = model[n1_only]
    W2 = model[n2_only]

    D = distmat(W1, W2)
    symmetric_dist, mean_dist = word_mover_distances(D)
    result['w2v_%s_%s_euclid_wmd_sym' % (c1, c2)] = symmetric_dist
    result['w2v_%s_%s_euclid_wmd_mean' % (c1, c2)] = mean_dist

    D = manhattan(W1, W2)
    symmetric_dist, mean_dist = word_mover_distances(D)
    result['w2v_%s_%s_manhattan_wmd_sym' % (c1, c2)] = symmetric_dist
    result['w2v_%s_%s_manhattan_wmd_mean' % (c1, c2)] = mean_dist


column_pairs = [('title_1', 'title_2'), ('description_1', 'description_2'),
                ('title_1', 'description_2'), ('description_1', 'title_2'), 
                ('all_text_1', 'all_text_2'), ('nouns_1', 'nouns_2')]

def extract_w2v_features(row):
    result = {}
    for c1, c2 in column_pairs:
        full_text_features(c1, c2, row, result)
    return result

def process_batch(batch, pool):    
    batch.reset_index(drop=True, inplace=1)

    item_ids = sorted(set(batch.itemID_1) | set(batch.itemID_2))
    df_items = mongo.get_df_by_ids(avito_utils.item_text, item_ids, columns=columns)
    df_items.rename(columns={'description_lemmatized': 'description', 
                             'title_lemmatized': 'title'}, inplace=1)
    df_items.description = df_items.description.apply(tokens)
    df_items.title = df_items.title.apply(tokens)
    df_items['all_text'] = df_items.title + df_items.description

    item1 = df_items.loc[batch.itemID_1].reset_index(drop=True)
    item1.columns = [col + '_1' for col in item1.columns]
    item2 = df_items.loc[batch.itemID_2].reset_index(drop=True)
    item2.columns = [col + '_2' for col in item2.columns]
    batch = pd.concat([batch, item1, item2], axis=1)

    batch_dict = batch.to_dict(orient='records')
    result = pool.process_parallel(extract_w2v_features, collection=batch_dict)
    return pd.DataFrame(result)

def append_to_csv(batch, csv_file):
    props = dict(encoding='utf-8', index=False)
    if not os.path.exists(csv_file):
        batch.to_csv(csv_file, **props)
    else:
        batch.to_csv(csv_file, mode='a', header=False, **props)

def delete_file_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename)
        
def run():
    batch_size = 1000
    pool = avito_utils.PoolWrapper(processes=1)

    name = 'w2v_wmd'
  
    print 'processing train data...'
    t0 = time()
    df = pd.read_csv('../input/ItemPairs_train.csv')
    delete_file_if_exists('features_%s_train.csv' % name)

    for batch_no, batch in tqdm(list(prepare_batches(df, batch_size))):
        features = process_batch(batch, pool)
        append_to_csv(features, 'features_%s_train.csv' % name)

    print 'processing train data took %0.5fs' % (time() - t0)

    print 'processinig test data...'
    t0 = time()
    df = pd.read_csv('../input/ItemPairs_test.csv')
    delete_file_if_exists('features_%s_test.csv' % name)

    for batch_no, batch in tqdm(list(prepare_batches(df, batch_size))):
        features = process_batch(batch, pool)
        append_to_csv(features, 'features_%s_test.csv' % name)
        
    print 'processing test data took %0.5fs' % (time() - t0)


    pool.close()

if __name__ == "__main__":
    run()