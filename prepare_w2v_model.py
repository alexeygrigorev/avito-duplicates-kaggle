from tqdm import tqdm
from time import time

import avito_utils
from gensim.models import Word2Vec

import gc

stopwords = set(u'в,на,и,с,по,для,не,от,за,к,до,из,или,у,один,вы,при,так,ваш,как,а,наш,быть,под,б,'
                u'р,мы,эт,же,также,что,это,раз,свой,он,если,но,я,о,ещё,тот,этот,то,они,ни,чем,где,'
                u'бы,оно,там,ж,она,ты'.split(','))

def read_sentences():
    word_to_int_idx = {}

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
                sentences.append(res)

    reverse_idx = {v: k for (k, v) in word_to_int_idx.items()}

    sentences_str = []
    for s in sentences:
        sentences_str.append([reverse_idx[i] for i in s])

    return sentences_str

def train_w2v(sentences):
    print 'training w2v model...'
    t0 = time()
    
    num_features = 300    # Word vector dimensionality
    min_word_count = 10   # Minimum word count
    num_workers = 8     # Number of threads to run in parallel
    context = 3          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    model = Word2Vec(sentences_str, workers=num_workers, size=num_features, min_count=min_word_count,
                     window=context, sample=downsampling, seed=1)
    print 'took %0.5fs.' % (time() - t0)
    return model

def run():
    sentences = read_sentences()
    gc.collect()

    model = train_w2v(sentences)
    model.save('w2v/lemma_stopwords')

if __name__ == "__main__":
    run()