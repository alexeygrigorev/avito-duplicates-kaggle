# -*- encoding: utf-8 -*-

from time import time
from collections import Counter
import gc

from tqdm import tqdm
from glove import Corpus, Glove
import avito_utils

stopwords = set(u'в,на,и,с,по,для,не,от,за,к,до,из,или,у,один,вы,при,так,ваш,как,а,наш,быть,под,б,'
                u'р,мы,эт,же,также,что,это,раз,свой,он,если,но,я,о,ещё,тот,этот,то,они,ни,чем,где,'
                u'бы,оно,там,ж,она,ты'.split(','))


def read_sentences(min_word_count=50):
    word_to_int_idx = {}
    word_counter = Counter()

    gen = avito_utils.select(avito_utils.item_text, columns=['title_lemmatized', 'description_lemmatized'])
    sentences = []

    cnt = 0

    for batch in tqdm(gen):
        for rec in batch:
            all_tokens = rec["title_lemmatized"] + rec["description_lemmatized"]
            for sentence in all_tokens:
                res = []
                for t in sentence.split():
                    if t in stopwords:
                        continue
                        
                    # to save memory
                    if t in word_to_int_idx:
                        idx = word_to_int_idx[t]
                    else:
                        cnt = cnt + 1
                        word_to_int_idx[t] = cnt
                        idx = cnt
                    res.append(idx)
                word_counter.update(res)
                sentences.append(res)

    reverse_idx = {v: k for (k, v) in word_to_int_idx.items()}

    sentences_str = []
    for s in sentences:
        sentence = [reverse_idx[i] for i in s if word_counter[i] >= min_word_count]
        if not sentence:
            continue
        sentences_str.append(sentence)

    return sentences_str


def train_glove(sentences):
    print 'training glove model...'
    t0 = time()
    
    num_features = 300    # Word vector dimensionality
    context = 5          # Context window size
    learning_rate = 0.05
    
    corpus = Corpus()
    corpus.fit(sentences, window=context)

    glove = Glove(no_components=num_features, learning_rate=learning_rate)
    glove.fit(corpus.matrix, epochs=30, no_threads=8, verbose=True)
    glove.add_dictionary(corpus.dictionary)

    print 'took %0.5fs.' % (time() - t0)
    return glove


def run():
    sentences = read_sentences(min_word_count=50)
    gc.collect()

    model = train_glove(sentences)
    model.save('w2v/glove_lemma_stopwords')

if __name__ == "__main__":
    run()