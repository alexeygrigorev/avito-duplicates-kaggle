import pandas as pd
import codecs
import json
import os, sys
from tqdm import tqdm

from time import sleep
from zipfile import ZipFile
from urllib2 import urlopen

print 'reading data...'

data = sys.argv[1]

zip_file = ZipFile('../input/ItemInfo_%s.csv.zip' % data).open('ItemInfo_%s.csv' % data)
df_train_info = pd.read_csv(zip_file, usecols=['itemID', 'lat', 'lon'])

result_file = 'geo-%s.txt' % data

processed = set()
if os.path.exists(result_file):
    print 'result file %s aready exists, appending...' % result_file
    with codecs.open(result_file, 'r', 'utf8') as in_f:
        for line in in_f:
            split = line.split('\t')
            ad_id = split[0]
            processed.add(int(ad_id))
    print 'already processed %d records, picking up where we stopped...' % len(processed)
    result = codecs.open(result_file, 'a', 'utf8')
else:
    result = codecs.open(result_file, 'w', 'utf8')

df = df_train_info[~df_train_info.itemID.isin(processed)]
geo_triples = zip(df.itemID, df.lat, df.lon)

print 'processing...'

nominatim = 'http://172.17.0.2:8080/reverse?format=json&lat=%0.5f&lon=%0.5f'

def reverse(lat, lon):
    res = urlopen(nominatim % (lat, lon)).read()
    loc = json.loads(res)
    address = loc['address']
    postcode = address.get('postcode')
    state = address.get('state')
    city = address.get('city', state)
    return state, city, postcode

def try_process(id, lat, lon):
    state, city, postcode = reverse(lat, lon)
    result.write(str(id))
    result.write('\t')
    result.write(unicode(state))
    result.write('\t')
    result.write(unicode(city))
    result.write('\t')
    result.write(str(postcode))
    result.write('\n')
    result.flush()

for id, lat, lon in tqdm(geo_triples):
    try:
        try_process(id, lat, lon)
    except: 
        print 'error processing %d, lat=%0.5f&lon=%0.5f' % (id, lat, lon)

print 'done'

result.close()
