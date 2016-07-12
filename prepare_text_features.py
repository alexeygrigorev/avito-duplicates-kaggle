# -*- encoding: utf-8 -*-

from time import time
from collections import Counter
import itertools 
import re

from tqdm import tqdm

import avito_utils


### chargrams

def char_ngrams(s):
    result = Counter()
    len_s = len(s)
    for n in [3, 4, 5]:
        result.update(s[i:i+n] for i in xrange(len_s - n + 1))
    return result

def compute_chargrams(batch_size=5000):
    chargram_table = avito_utils.get_item_chargrams_table()
    texts = avito_utils.select(avito_utils.item_info, ['title_clean', 'description_clean'], 
                                 batch_size=batch_size) 
    cnt = 1
    processed = 0

    for batch in texts:
        t0 = time()
        batch_result = []
        for rec in tqdm(batch):
            d = dict(_id=rec['_id'],
                     chargram_title=char_ngrams(rec['title_clean'].replace(' ', '')),
                     chargram_desc=char_ngrams(rec['description_clean'].replace(' ', '')))
            batch_result.append(d)

        chargram_table.insert_many(batch_result)
        print 'batch %d finished in %0.5fs' % (cnt, time() - t0),

        processed = processed + len(batch)
        print 'so far processed %d rows' % processed
        cnt = cnt + 1
    
    print 'done'
    
    
### NLP stuff

import pymorphy2
morph = pymorphy2.MorphAnalyzer(result_type=None)
from fastcache import clru_cache as lru_cache

shortenings = [u'мм', u'см', u'м', u'км', u'мл', u'л', u'г', u'кг', u'т', u'лит',
               u'р', u'руб', u'сот', u'га', u'шт', u'ш', u'дб', u'вт', 
               u'ул', u'пр', u'д', u'кв', u'чел', u'жен', u'муж', u'тыс', 
               u'др']

shortenings = u'|'.join(shortenings)

def remove_dots_from_shortenings(match):
    s = match.string[match.start():match.end()]
    return s.replace('.', '')

def normalize(text):
    text = text.lower().replace(u'ё', u'е')
    text = text.replace(u'²', '2')
    text = re.sub(r'(\d+)[.](\d+)', r'\1,\2', text)
    text = re.sub(u'([a-zа-я][.]){2,}', remove_dots_from_shortenings, text)
    text = re.sub(ur'(?<=[^а-я])(' + shortenings + ')[.]', ur'\1 ', text)
    return text

def clean_text(text):
    #text = re.sub(r'(\d+),(\d+)', r'\1.\2', text)
    text = re.sub(u'[^a-zа-я0-9.]', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

def sentence_split(text):
    text = normalize(text)
    res = []
    for s in re.split('[.!?;\n]', text):
        s = clean_text(s)
        if s: 
            res.append(s)
    return res

def en_chars_cnt(s):
    return sum(1 for c in s if u'a' <= c <= u'z')

def ru_chars_cnt(s):
    return sum(1 for c in s if u'а' <= c <= u'я')

ru_en = {u'а': u'a', u'у': u'y', u'к': u'k', u'е': u'e', u'н': u'h', 
    u'г': u'r', u'х': u'x', u'в': u'b', u'а': u'a', u'р': u'p', 
    u'о': u'o', u'с': u'c', u'м': u'm', u'т': u't'}

ru_translatable = set(ru_en.keys())
en_translatable = set(ru_en.values())

en_ru = {ord(v): k for k, v in ru_en.items()}
ru_en = {ord(k): v for k, v in ru_en.items()}

def translate(s):
    # None - no need to translate
    en_cnt = en_chars_cnt(s)
    ru_cnt = ru_chars_cnt(s)
    
    if en_cnt == 0 or ru_cnt == 0:
        return None
   
    if en_cnt > ru_cnt: 
        return s.translate(ru_en)
    if ru_cnt > en_cnt:
        return s.translate(en_ru)
    
    # en == ru
    chars = set(s)
    en_cnt = len(en_translatable & chars)
    ru_cnt = len(ru_translatable & chars)
    
    if en_cnt > 0 and ru_cnt == 0:
        return s.translate(en_ru)
    if ru_cnt > 0 and en_cnt == 0:
        return s.translate(ru_en)

    # can't do anything here
    return s

def translate_tokens(sentence):
    mixed = 0
    result = []
    for token in sentence.split():
        trans = translate(token)
        if trans is None:
            result.append(token)
        else:
            result.append(trans)
            mixed = mixed + 1
    return ' '.join(result), mixed

def is_digit(s):
    if s.isdecimal():
        return True
    if re.match(ur'^\d+[.]\d+$', s):
        return True
    
    return False

def is_rus(s):
    return ru_chars_cnt(s) == len(s)

def is_eng(s):
    return en_chars_cnt(s) == len(s)

def is_rus_digit_mixed(s):
    if re.match(ur'^[а-я0-9]+$', s):
        return 0 < ru_chars_cnt(s) < len(s)
    return False

def is_eng_digit_mixed(s):
    if re.match(ur'^[a-z0-9]+$', s):
        return 0 < en_chars_cnt(s) < len(s)
    return False


@lru_cache(maxsize=1000000)
def lemmatize_pos(word):
    _, tag, norm_form, _, _ = morph.parse(word)[0]
    return norm_form, tag.POS


def extract_text_features(rec):
    original_text = rec['title'] + rec['description']
    digits = re.findall(ur'\d+', original_text)
    unicode_chars = list({c for c in original_text if ord(c) > 1105})

    title = sentence_split(rec['title'])
    if title:
        title_cleaned, title_mixed_count = zip(*[translate_tokens(s) for s in title])
    else:
        title_cleaned, title_mixed_count = (), ()
    
    desc = sentence_split(rec['description'])
    if desc:
        desc_cleaned, desc_mixed_count = zip(*[translate_tokens(s) for s in desc])
    else:
        desc_cleaned, desc_mixed_count = (), ()
    
    total_mixed_count = sum(title_mixed_count) + sum(desc_mixed_count)

    english_only = []
    en_digits = []
    ru_digits = []

    for sentence in title_cleaned + desc_cleaned:
        for s in sentence.split():
            if is_eng(s):
                english_only.append(s)
            elif is_rus_digit_mixed(s):
                ru_digits.append(s)
            elif is_eng_digit_mixed(s):
                en_digits.append(s)

    nouns = []
    title_lemmas = []
    desc_lemmas = []
    
    for sentence in title_cleaned:
        lemmatized = []
        for s in sentence.split():
            s, pos = lemmatize_pos(s)
            lemmatized.append(s)
            if pos == 'NOUN':
                nouns.append(s)
        title_lemmas.append(' '.join(lemmatized))

    for sentence in desc_cleaned:
        lemmatized = []
        for s in sentence.split():
            s, pos = lemmatize_pos(s)
            lemmatized.append(s)
            if pos == 'NOUN':
                nouns.append(s)
        desc_lemmas.append(' '.join(lemmatized))

    return dict(_id=rec['_id'], 
                title_cleaned=list(title_cleaned), description_cleaned=list(desc_cleaned),
                title_lemmatized=title_lemmas, description_lemmatized=desc_lemmas,
                rus_eng_chars_mixed=total_mixed_count, unicode_chars=unicode_chars, 
                digits=digits, english_only=english_only, nouns=nouns,
                english_digits_mix=en_digits, russian_digits_mix=ru_digits)

def tokenization(batch_size=20000):
    t0 = time()

    result_table = avito_utils.get_item_text_table()
    result_table.drop()

    texts = avito_utils.select(avito_utils.item_info, ['title', 'description'], batch_size=batch_size) 
    pool = avito_utils.PoolWrapper()

    for batch in tqdm(texts):
        batch_result = pool.process_parallel(extract_text_features, batch)
        result_table.insert_many(batch_result)

    pool.close()

    print 'done in %0.5fs' % (time() - t0)
    
if __name__ == "__main__":
    import sys
    args = sys.argv
    if args[1] == 'chargrams':
        compute_chargrams()
    elif args[1] == 'tokens':
        tokenization()
    else:
        print 'should be one of the following: chargrams, tokens'