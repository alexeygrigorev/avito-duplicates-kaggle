# -*- encoding: utf-8 -*-


import json
from time import time

import pandas as pd
import itertools
from tqdm import tqdm

from mongo_utils import MongoWrapper


avito_db = 'avito'
mongo = MongoWrapper(avito_db)

def load_json(collection, filename):
    print 'loading json data from %s to the collection %s' % (filename, collection)
    t0 = time()

    table = mongo.table(collection)
    table.drop()

    for line in tqdm(open(filename, 'r')):
        d = json.loads(line)
        table.insert(d)

    print 'loading json took %0.5fs' % (time() - t0)


if __name__ == "__main__":
    import sys
    command, collection, filename = sys.argv[1:4]
    #print command, collection, filename
    
    if command == 'load_json':
        load_json(collection, filename)