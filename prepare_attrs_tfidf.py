# -*- encoding: utf-8 -*-

import os
import re
from time import time

import cPickle


import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import avito_utils
mongo = avito_utils.mongo


def transform_autocode_report(text, result={}):
    for s in text.split(', '):
        if '=>' not in s:
            continue
        k, v = s.split('=>')
        k = k.strip('"')
        v = v.strip('"')
        if v:
            result['autocode_report_' + k] = v
        else:
            result['autocode_report_' + k] = 'NA'
    return result

def extract_autocode_report(attrs):
    if u'Отчёт Автокод' not in attrs:
        return {}

    text = attrs[u'Отчёт Автокод']
    return transform_autocode_report(text)

def normalize(s):
    return s.lower().replace(' ', '_')

def unwrap_dict(attrs):
    if u'Отчёт Автокод' in attrs:
        attrs = attrs.copy()
        autocode = attrs[u'Отчёт Автокод']
        del attrs[u'Отчёт Автокод']
        transform_autocode_report(autocode, attrs)
    pairs = [(normalize(k), normalize(v)) for (k, v) in attrs.items()]
    values = [p[1] for p in pairs]
    return [u'%s=%s' % p for p in pairs], values

def identity(l): return l

def run():
    t0 = time()
    print 'reading data from mongo...'
    columns = ['attrsJSON']
    attrs_batches = mongo.select(avito_utils.item_info, columns, batch_size=1000)

    attr_pairs = []
    attr_values = []

    for batch in tqdm(attrs_batches):
        for attrs in batch:
            pairs, values = unwrap_dict(attrs['attrsJSON'])
            attr_pairs.append(pairs)
            attr_values.append(values)

    print 'reading took %0.5s' % (time() - t0)

    t0 = time()
    print 'creating tf-idf and svd for pairs'

    tfidf = TfidfVectorizer(analyzer=identity, min_df=5, norm=None)
    attrs_matrix = tfidf.fit_transform(attr_pairs)

    svd = TruncatedSVD(n_components=70, random_state=1)
    svd.fit(attrs_matrix)

    with open('tfidf_svd_attrs_pairs.bin', 'wb') as f:
        cPickle.dump((tfidf, svd), f)

    print 'done in %0.5s' % (time() - t0)


    t0 = time()
    print 'creating tf-idf and svd for pairs'
    tfidf_val = TfidfVectorizer(analyzer=identity, min_df=5, norm=None)
    vals_matrix = tfidf_val.fit_transform(attr_values)

    svd_val = TruncatedSVD(n_components=50, random_state=1)
    svd_val.fit(vals_matrix)
    
    with open('tfidf_svd_attrs_vals.bin', 'wb') as f:
        cPickle.dump((tfidf_val, svd_val), f)

    print 'done in %0.5s' % (time() - t0)

if __name__ == "__main__":
    run()