import avito_utils
import json

import numpy as np
import pandas as pd

from tqdm import tqdm
import csv
import itertools

from multiprocessing import Pool

def extract_parentesis(s):
    res = s.split()[1]
    return res[1:-1]

def extract_both(s):
    split = s.split()
    return split[0], split[1][1:-1]

def moment_features(pref, in_dict, out_dict={}):
    ellipse_major, ellipse_minor = in_dict['Ellipse Semi-Major/Minor axis'].split(',')

    out_dict['moments_' + pref + '_ellipse_major'] = float(ellipse_major)
    out_dict['moments_' + pref + '_ellipse_minor'] = float(ellipse_minor)
    out_dict['moments_' + pref + '_ellipse_eccentricity'] = float(in_dict['Ellipse eccentricity'])
    out_dict['moments_' + pref + '_ellipse_intensity'] = float(extract_parentesis(in_dict['Ellipse intensity']))
    out_dict['moments_' + pref + '_ellipse_angle'] = float(in_dict[u'Ellipse angle'])

    centroid_x, centroid_y = in_dict[u'Centroid'].split(',')
    out_dict['moments_' + pref + '_ellipse_centroid_x'] = float(centroid_x)
    out_dict['moments_' + pref + '_ellipse_centroid_y'] = float(centroid_y)

    for i in range(1, 9):
        v1, v2 = extract_both(in_dict['I%d' % i])
        out_dict['moments_' + pref + '_I%d_1' % i] = float(v1)
        out_dict['moments_' + pref + '_I%d_2' % i] = float(v2)

def phash_features(in_dict, out_dict={}):
    for k, phdict in in_dict.items():
        k1, k2 = k.lower().split(', ')
        for phk, phv in phdict.items():
            phv1, phv2 = phv.split(', ')
            out_dict['phash_%s_%s' % (k1, phk.lower())] = float(phv1)
            out_dict['phash_%s_%s' % (k2, phk.lower())] = float(phv2)
        
def image_stats_features(pref, in_dict, out_dict={}):
    out_dict['imstat_' + pref + '_skewness'] = float(in_dict['skewness'])
    out_dict['imstat_' + pref + '_min'] = float(extract_parentesis(in_dict['min']))
    out_dict['imstat_' + pref + '_max'] = float(extract_parentesis(in_dict['max']))
    out_dict['imstat_' + pref + '_mean'] = float(extract_parentesis(in_dict['mean']))
    out_dict['imstat_' + pref + '_std'] = float(extract_parentesis(in_dict['standard deviation']))
    out_dict['imstat_' + pref + '_entropy'] = float(in_dict['entropy'])
    out_dict['imstat_' + pref + '_kurtosis'] = float(in_dict['kurtosis'])

def chromaticity_features(in_dict, out_dict={}):
    for k, v in in_dict.items():
        k = k.split()[0]
        v1, v2 = eval(v)
        out_dict['croma_%s_1' % k] = v1
        out_dict['croma_%s_2' % k] = v2    

def number_pixels(s):
    if type(s) in [int, float]:
        return s
    if s.endswith('K'):
        return 1000 * float(s[:-1])

    raise Exception('number_pixels: unknown format' + s)

def file_size(s):
    if s.endswith('KB'):
        return 1024 * float(s[:-2])
    elif s.endswith('B'):
        return float(s[:-1])
    elif s.endswith('MB'):
        return 1024 * 1024 * float(s[:-2])
    raise Exception('file_size: unknown format ' + s)

def process_line(line):
    image_id, features = line.split('\t')
    features = json.loads(features)

    res_dict = {'_id': image_id}

    is_gray = features[u'Type'] in [u'Grayscale', u'Bilevel']
    if is_gray:
        moment_features('overall', features[u'Channel moments']['Gray'], res_dict)
        moment_features('red', features[u'Channel moments']['Gray'], res_dict)
        moment_features('green', features[u'Channel moments']['Gray'], res_dict)
        moment_features('blue', features[u'Channel moments']['Gray'], res_dict)

        image_stats_features('overall', features[u'Channel statistics']['Gray'], res_dict)
        image_stats_features('red', features[u'Channel statistics']['Gray'], res_dict)
        image_stats_features('green', features[u'Channel statistics']['Gray'], res_dict)
        image_stats_features('blue', features[u'Channel statistics']['Gray'], res_dict)
    else:
        moment_features('overall', features[u'Image moments']['Overall'], res_dict)
        moment_features('red', features[u'Channel moments']['Red'], res_dict)
        moment_features('green', features[u'Channel moments']['Green'], res_dict)
        moment_features('blue', features[u'Channel moments']['Blue'], res_dict)

        image_stats_features('overall', features[u'Image statistics']['Overall'], res_dict)
        image_stats_features('red', features[u'Channel statistics']['Red'], res_dict)
        image_stats_features('green', features[u'Channel statistics']['Green'], res_dict)
        image_stats_features('blue', features[u'Channel statistics']['Blue'], res_dict)

    phash_features(features[u'Channel perceptual hash'], res_dict)
    res_dict['no_pixels'] = number_pixels(features[u'Number pixels'])
    res_dict['filesize'] = file_size(features[u'Filesize'])

    res_dict['quality'] = features[u'Quality']
    res_dict['geometry'] = features[u'Geometry'][:-4]
    res_dict['type'] = features[u'Type']
    res_dict['colorspace'] = features[u'Colorspace']

    props = features[u'Properties']
    res_dict['date'] = props[u'date:modify']
    res_dict['signature'] = props[u'signature']
    
    return res_dict


def process_parallel(pool, series, function):
    return pool.map(function, series)

def partition(iterable, size):
    while 1:
        batch = list(itertools.islice(iterable, size))
        if not batch:
            break
        yield batch

def run():
    csvfile = open('images-features-full.csv', 'w')
    writer = None

    mongo_table = avito_utils.get_imagemagick_table()  
    mongo_table.drop()

    pool = Pool(processes=8)
    
    files = [open('../input/Images_%d-_-_.jpg.txt' % i) for i in range(10)]
    batch_size = 8000

    for batch in tqdm(partition(itertools.chain(*files), batch_size)):
        batch_result = process_parallel(pool, batch, process_line)

        for res_dict in batch_result:
            if writer is None:
                fieldnames = sorted(res_dict.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

            writer.writerow(res_dict)

        mongo_table.insert_many(batch_result)

    pool.close()
    pool.join()

    csvfile.close()

                
if __name__ == "__main__":
    run()