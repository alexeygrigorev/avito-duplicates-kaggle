# -*- encoding: utf-8 -*-

import json

import pandas as pd
import itertools

from mongo_utils import MongoWrapper

avito_db = 'avito'
mongo = MongoWrapper(avito_db)

item_info = 'item_info'
item_chargrams = 'item_chargrams'
item_text = 'item_text'
imagemagick = 'imagemagick'

def get_mongo_connection():
    return mongo.client

def get_mongo_table(table):
    return mongo.table(table)
    
def get_item_info_table():
    return mongo.table(item_info)

def get_item_chargrams_table():
    return mongo.table(item_chargrams)

def get_imagemagick_table():
    return mongo.table(imagemagick)

def get_item_text_table():
    return mongo.table(item_text)

def select(table, columns, include_id=True, batch_size=10000):
    return mongo.select(table, columns, include_id, batch_size)

def get_item_infos(item_ids, columns=None):
    return mongo.get_df_by_ids(item_info, item_ids, columns)

def get_chargrams(item_ids):
    return mongo.get_df_by_ids(item_chargrams, item_ids)

def get_imagemagick_features(image_ids):
    return mongo.get_df_by_ids(imagemagick, image_ids)
    

def print_rus(obj):
    print json.dumps(obj, ensure_ascii=False, indent=2)


    
from multiprocessing import Pool
from functools import partial

def apply_func(args, kwargs, func):
    return func(*args, **kwargs)

class LocalPool():
    def map(self, f, it):
        return [f(el) for el in it]
    def close(self): pass
    def join(self): pass

class PoolWrapper():
    def __init__(self, processes=8):
        if processes == 1:
            self.pool = LocalPool()
        else:
            self.pool = Pool(processes=processes)

    def process_parallel(self, function, collection=None, collections=None, **kwargs):
        """
        :param function: function to apply
        :param collection: one collection to which the function is applied. If this is set, 
                    collections argument should be None
        :param collections: several collections to be zipped. function must have at least the same number of 
                    params as there are collections. if this is set, collection argument should be None
        :param kwargs: some parameters to pass to the function 
        """
        if collection is not None and collections is not None:
            raise Exception('cannot use both collection and collections argument')

        if collection is None and collections is None:
            raise Exception('no collection is provided')

        if collection is None and collections is not None:
            collection=zip(*collections)
            map_function = partial(apply_func, func=function, kwargs=kwargs)
        elif kwargs is not None:
            map_function = partial(function, **kwargs)
        else:
            map_function = function

        return self.pool.map(map_function, collection)

    def close(self):
        self.pool.close()
        self.pool.join()

    def __enter__(self): 
        return self
    
    def __exit__(self, type, value, traceback):
        self.close()