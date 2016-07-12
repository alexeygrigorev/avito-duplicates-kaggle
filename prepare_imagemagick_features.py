from subprocess import check_output
import yaml
import json
from tqdm import tqdm
from time import time
from glob import glob
import os

hist = 'Histogram:'
colormap = 'Colormap:'

def extract_image_id(file_path):
    file_name = os.path.basename(file_path)
    return file_name[:-4]

def process(input_list, result_file, error_file):
    for file_path in tqdm(input_list):
        data = process_image(file_path)
        if data is None:
            continue

        image_id = extract_image_id(file_path)
        result.write(image_id)
        result.write('\t')
        result.write(json.dumps(data))
        result.write('\n')

        result.flush()

def process_image(file_path):
    try:
        output = check_output(['identify', '-verbose', '-moments', file_path])
        return try_process_output(output)
    except:
        print 'error processing %s' % fn
        errors.write(fn)
        errors.write('\n')
        errors.flush()
        return None

def try_process_output(output):
    form = [s[2:] for s in output.split('\n')[1:-4] if s and (not 'comment:' in s)]

    remove_entry(hist, form)
    remove_entry(colormap, form)

    form = '\n'.join(form)
    form = yaml.load(form)

    del form['Artifacts']
    del form['Class']
    del form['Dispose']
    del form['Endianess']
    del form['Compression']
    del form['Units']
    del form['Properties']['date:create']
    form['Properties']['date:modify'] = str(form['Properties']['date:modify'])
    return form

def remove_entry(name, form):
    if name not in form:
        return
    idx = form.index(name)
    end_idx = idx + 1

    while end_idx < len(form):
        if not form[end_idx].startswith(' '):
            break
        end_idx = end_idx + 1

    del form[idx:end_idx]

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input_pattern")
    group.add_argument("--input_file")
    
    parser.add_argument("--append", action="store_true", default=False)
    parser.add_argument("--error_file")
    parser.add_argument("--output_file")

    args = parser.parse_args()
    print args
    
    if args.input_pattern:
        pat = args.input_pattern
        print 'reading using pattern %s' % pat
        input_list = [fn for fn in glob(pat)]
        print '%d files matched the pattern' % len(input_list)
    elif args.input_file:
        print 'ololo'
    else:
        raise Exception('no input specified')

    print 'list head:', input_list[:5]

    error_file = args.error_file
    if error_file is None:
        error_file = 'error_%d.txt' % time()
    print 'using error file %s' % error_file

    output_file = args.output_file
    if output_file is None:
        output_file = pat.replace('/', '-').replace('*', '_') + '.txt'
        print 'output file is %s' % output_file

    if args.append:
        result = open(output_file, 'a')
    elif os.path.exists(output_file):
        print 'file already exists, trying to continue'
        processed = open(output_file, 'r')
        print 'reading processed file...'
        already_processed = set()
        for line in processed:
            split = line.split('\t')
            image_id = split[0]
            already_processed.add(image_id)
        print 'already processed %d files' % len(already_processed)
        input_list = [fn for fn in input_list if extract_image_id(fn) not in already_processed]
        print 'left to process: %d' % len(input_list)
        result = open(output_file, 'a')
    else:
        result = open(output_file, 'w')

    errors = open(error_file, 'w')
    
    print 'starting...'
    process(input_list, result, errors)
    
    result.close()
    errors.close()

# python prepare_imagemagick_features.py --input_pattern 'Images_0/*/*.jpg'
# repeat for 1..9
