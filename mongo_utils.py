# -*- encoding: utf-8 -*-

import json

from pymongo import MongoClient
import pandas as pd
import itertools

mongo_url = '172.17.0.2'

class MongoWrapper():
    def __init__(self, db_name):
        self.db_name = db_name
        self.client = MongoClient(mongo_url, 27017, connect=False)
        self._db = None

    def table(self, name):
        if self._db is None:
            self._db = self.client[self.db_name]

        return self._db[name]
    
    def select_gen(self, table_name, columns, include_id=True):
        params = {c: 1 for c in columns}
        if not include_id:
            params['_id'] = 0

        mongo_table = self.table(table_name)
        for rec in mongo_table.find({}, params):
            yield rec

    def select(self, table_name, columns, include_id=True, batch_size=10000):
        gen = self.select_gen(table_name, columns, include_id)

        while 1:
            batch = list(itertools.islice(gen, batch_size))
            if batch:
                yield batch
            else:
                break

    def get_by_ids(self, table_name, ids_list, columns=None):
        mongo_table = self.table(table_name)
        if columns is not None:
            query_params = {c: 1 for c in columns}
            cursor = mongo_table.find({'_id': {'$in': ids_list}}, query_params)
        else:
            cursor = mongo_table.find({'_id': {'$in': ids_list}})

        return list(cursor)
    
    def get_df_by_ids(self, table_name, ids_list, columns=None, index_col='_id'):
        results = self.get_by_ids(table_name, ids_list, columns)
        df_res = pd.DataFrame(results)
        if index_col is not None:
            df_res.set_index(index_col, inplace=1)
        return df_res

    def drop_table(self, table_name):
        mongo_table = self.table(table_name)
        mongo_table.drop()