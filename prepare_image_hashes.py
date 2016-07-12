import numpy as np

import zipfile
import os
import io

from PIL import Image
import imagehash

from tqdm import tqdm
from time import time

import avito_utils


# function taken from 
# https://www.kaggle.com/iezepov/avito-duplicate-ads-detection/get-hash-from-images-slightly-daster/code

def extract_dhash(img, hash_size=20):
    img = img.convert('LA').resize((hash_size+1, hash_size), Image.ANTIALIAS)
    mat = np.array(img.getdata())[:, 0].reshape(hash_size, hash_size + 1)
    diff = np.packbits(np.diff(mat) >= 0)
    return ''.join('%0.2x' % x for x in diff)


def get_signatures_from_mongo():
    print 'reading image signatures...'
    t0 = time()
    mongo = avito_utils.mongo
    image_signatures = mongo.select(avito_utils.imagemagick, columns=['signature'], batch_size=10000)
    signatures = {}
    for batch in tqdm(image_signatures):
        for d in batch:
            signatures[d['_id']] = d['signature']
    print 'done in %0.5fs' % (time() - t0)
    return signatures

def get_signatures_from_file():
    import pandas as pd
    print 'reading image signatures...'
    t0 = time()
    it = pd.read_csv('images-features-full.csv.gz', compression='gzip', 
                 usecols=['_id', 'signature'], iterator=True, chunksize=250000)
    signatures = {}
    for batch in tqdm(it):
        for d in batch.to_dict(orient='records'):
            signatures[d['_id']] = d['signature']
    print 'done in %0.5fs' % (time() - t0)
    return signatures

def get_pickled_signatures():
    import cPickle
    t0 = time()
    print 'reading image signatures...'
    with open('image-signatures.bin', 'rb') as f:
        res = cPickle.load(f)
        print 'done in %0.5fs' % (time() - t0)
        return res

def run():
    signatures = get_pickled_signatures()

    import csv
    hashes_file = open('image_hashes.csv', 'w')
    columns = ['image_id', 'script_dhash', 'ahash', 'dhash', 'phash', 'signature']
    csv_writer = csv.DictWriter(hashes_file, fieldnames=columns)
    csv_writer.writeheader()

    t0 = time()

    for zip_counter in range(0, 10):
        filename = '../input/Images_%d.zip' % zip_counter
        print 'processing %s...' % filename

        imgzipfile = zipfile.ZipFile(filename)
        namelist = imgzipfile.namelist()

        for name in tqdm(namelist):
            if not name.endswith('.jpg'):
                continue
            filename = name.split('/')[-1]
            img_id = filename[:-4]
            try:
                imgdata = imgzipfile.read(name)

                if len(imgdata) == 0:
                    print '%s is empty' % img_id 
                    continue

                stream = io.BytesIO(imgdata)
                img = Image.open(stream)

                ahash = imagehash.average_hash(img)
                dhash = imagehash.dhash(img)
                phash = imagehash.phash(img)
                script_dhash = extract_dhash(img)

                csv_writer.writerow({'image_id': img_id, 'script_dhash': script_dhash, 
                                     'ahash': str(ahash), 'dhash': str(dhash), 'phash': str(phash),
                                     'signature': signatures[int(img_id)]})
            except:
                print 'error with ' + img_id

    hashes_file.flush()
    hashes_file.close()

    print 'took %0.5fm' % ((time() - t0) / 60)

if __name__ == "__main__":
    run()