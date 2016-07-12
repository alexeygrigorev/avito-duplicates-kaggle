# -*- encoding: utf-8 -*-

from time import time
import cPickle
import os

from tqdm import tqdm

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from collections import defaultdict

from PIL import Image
from ssim import SSIMImage, SSIM

import avito_utils
mongo = avito_utils.mongo

def get_path(image_id):
    if len(image_id) == 1:
        big = '0'
    else:
        big = image_id[-2]

    last = image_id[-2:]
    last = str(int(last))
    return '../input/Images_' + big + '/' + last + '/' + image_id + '.jpg' 


def get_pickled_signatures(): 
    t0 = time()
    print 'reading image signatures...'
    with open('image-signatures.bin', 'rb') as f:
        res = cPickle.load(f)
        print 'done in %0.5fs' % (time() - t0)
        return res

def prepare_batches(df, n):
    for batch_no, i in enumerate(range(0, len(df), n)):
        yield batch_no, df.iloc[i:i+n]


signatures = None

def read_ssim_files(images):
    res = {}
    file_handles = []
    for im in images:
        try:
            path = get_path(im)
            img = Image.open(path)
            file_handles.append(img)

            ssim_img = SSIMImage(img, size=(32, 32))
            ssim = SSIM(ssim_img.img)
            res[im] = ssim
        except Exception, e:
            print e
            continue
    return res, file_handles


def calc_features(images1, images2):
    if not images1 or not images2:
        return {}

    global signatures

    sig_1 = {i: signatures[i] for i in images1 if i in signatures}
    sig_2 = {i: signatures[i] for i in images2 if i in signatures}
    if not sig_1 or not sig_2: 
        return {}

    common = set(sig_1.values()) & set(sig_2.values())
    images1 = [str(i) for (i, s) in sig_1.items() if s not in common]
    images2 = [str(i) for (i, s) in sig_2.items() if s not in common]
    
    if not images1 or not images2:
        return {}

    handles = []
    try:
        ssim_dict, handles = read_ssim_files(images1 + images2)
        result = {}
        values = []
        
        for im1 in images1:
            if im1 not in ssim_dict:
                continue

            ssim1 = ssim_dict[im1]
            for im2 in images2:
                if im2 not in ssim_dict:
                    continue
                ssim2 = ssim_dict[im2]
                sim = ssim1.cw_ssim_value(ssim2.img)
                values.append(sim)

        result['ssim_min'] = np.min(values)
        result['ssim_mean'] = np.mean(values)
        result['ssim_max'] = np.max(values)
        result['ssim_std'] = np.std(values)
        result['ssim_25p'] = np.percentile(values, q=25)
        result['ssim_75p'] = np.percentile(values, q=75)
        result['ssim_skew'] = skew(values)
        result['ssim_kurtosis'] = kurtosis(values)

        return result
    except Exception, e:
        print e
        return {}
    finally:
        for im in handles:
            try:
                im.close()
            except:
                continue

def batch_features(i1, i2):
    i1 = map(int, i1)
    i2 = map(int, i2)
    return calc_features(i1, i2)

def process_batch(batch, pool):    
    batch = batch.reset_index(drop=True)

    item_ids = sorted(set(batch.itemID_1) | set(batch.itemID_2))
    batch_images = mongo.get_df_by_ids(avito_utils.item_info, item_ids, columns=['images_array'])
    images_1 = batch_images.loc[batch.itemID_1].images_array.reset_index(drop=True)
    images_2 = batch_images.loc[batch.itemID_2].images_array.reset_index(drop=True)

    result = pool.process_parallel(batch_features, collections=(images_1, images_2))
    result = pd.DataFrame(result)

    return result

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
    batch_size = 4000

    global signatures
    signatures = get_pickled_signatures()

    pool = avito_utils.PoolWrapper(processes=4)
    name = 'ssim'

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