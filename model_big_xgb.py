
# coding: utf-8

# In[1]:

import os
import json
import random
from time import time

import pandas as pd
import numpy as np

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import KFold

from sklearn.metrics import roc_auc_score

import xgboost as xgb

t0 = time()

seed = int(time() * 1000)
print 'using seed =', seed

random.seed(seed)
np.random.seed(seed % 4294967295)

n_estimators = 2500

xgb_pars = {
    'eta': 0.05,
    'gamma': 0.07,
    'max_depth': 8,
    'min_child_weight': 1,
    'max_delta_step': 0,
    'subsample': 0.86,
    'colsample_bytree': 0.55,
    'colsample_bylevel': 1,
    'lambda': 1,
    'alpha': 0,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 12,
    'seed': 20,
    'silent': 1
}

# In[2]:

label = pd.read_csv('features_chargrams_train.csv', usecols=['isDuplicate'], dtype=np.uint8)
y_full = label.isDuplicate.values

# In[4]:

print 'reading chargrams...'

chargram_cols = [
 'all_text_cosine', 'all_text_digits_char_diff',
 'all_text_jaccard', 'category',
 'desc_jaccard',
 'description_digits_char_diff', 'description_en_char_diff', 'description_unique_char_diff',
 'distance', 'images_cnt_diff',
 'price_diff', 'same_city', 'same_location', 'same_metro',
 'title_cosine',
 'title_digits_char_diff', 'title_digits_ratio_diff', 'title_en_char_diff',
 'title_en_ratio_diff', 'title_non_char_diff', 'title_non_char_ratio_diff',
 'title_ru_char_diff', 'title_ru_ratio_diff',
 'zip_diff']

df_diffs_chargrams = pd.read_csv('features_chargrams_train.csv', usecols=chargram_cols, 
                                 dtype=np.float32)


# In[10]:

print 'reading fuzzy...'

fuzzy_cols = ['all_text_1_all_text_2_token_set_ratio',
 'description_1_description_2_QRatio',
 'description_1_description_2_UQRatio',
 'description_1_description_2_partial_ratio',
 'description_1_title_2_UWRatio',
 'description_1_title_2_partial_ratio',
 'title_1_description_2_UWRatio',
 'title_1_description_2_partial_ratio',
 'title_1_description_2_token_set_ratio',
 'title_1_title_2_UWRatio',
 'title_1_title_2_WRatio']

df_fuzzy = pd.read_csv('features_fuzzy_train.csv', usecols=fuzzy_cols, dtype=np.float32)


# In[12]:

print 'reading imagemagick...'

imagemagick_cols = ['imagemagick_abs_diff_ellipse_green_amax',
 'imagemagick_abs_diff_ellipse_green_amin',
 'imagemagick_abs_diff_ellipse_overall_p75',
 'imagemagick_abs_diff_ellipse_overall_std',
 'imagemagick_abs_diff_filesize_p25',
 'imagemagick_abs_diff_imstat_blue_max_amin',
 'imagemagick_abs_diff_imstat_blue_max_mean',
 'imagemagick_abs_diff_imstat_blue_std_amin',
 'imagemagick_abs_diff_imstat_green_kurtosis_kurtosis',
 'imagemagick_abs_diff_imstat_green_mean_std',
 'imagemagick_abs_diff_imstat_green_min_mean',
 'imagemagick_abs_diff_imstat_green_min_p25',
 'imagemagick_abs_diff_imstat_green_skewness_amin',
 'imagemagick_abs_diff_imstat_green_std_amin',
 'imagemagick_abs_diff_imstat_overall_entropy_p75',
 'imagemagick_abs_diff_imstat_overall_min_p75',
 'imagemagick_abs_diff_imstat_red_max_std',
 'imagemagick_abs_diff_imstat_red_min_p25',
 'imagemagick_abs_diff_imstat_red_std_amin',
 'imagemagick_geometry_match',
 'imagemagick_no_exact_matches',
 'imagemagick_phash_all_pairs_euclidean_amin',
 'imagemagick_phash_all_pairs_manhattan_amin',
 'imagemagick_phash_chroma_pairs_euclidean_p25',
 'imagemagick_phash_chroma_pairs_manhattan_p25']

df_imagemagick = pd.read_csv('features_imagemagick_train.csv', usecols=imagemagick_cols, dtype=np.float32)


# In[13]:

print 'reading imagehash...'

imagehash_cols = ['ahash_hamming_distance_mean',
 'ahash_symmetric_cross_entropy_25p',
 'ahash_symmetric_cross_entropy_75p',
 'dhash_hamming_distance_min',
 'phash_dot_product_max',
 'phash_dot_product_std',
 'script_dhash_hamming_distance_25p',
 'script_dhash_hamming_distance_min',
 'script_dhash_hamming_distance_std']
df_imagehash = pd.read_csv('features_imagehash_train.csv', usecols=imagehash_cols, dtype=np.float32)


# In[14]:

print 'reading java-image...'

java_image_cols = ['hist_bhattacharyya_min',
 'hist_chi_square_25p',
 'hist_chi_square_min',
 'hist_jaccard_distance_max',
 'hist_jaccard_distance_mean',
 'hist_jaccard_distance_min',
 'hist_sum_square_min']

df_java = pd.read_csv('features_java-image_train.csv', usecols=java_image_cols, dtype=np.float32)


# In[15]:

print 'reading w2v...'

w2v_cols = ['w2v_all_text_1_all_text_2_corr_p75',
 'w2v_all_text_1_all_text_2_euclidean_amin',
 'w2v_all_text_1_all_text_2_euclidean_mean',
 'w2v_description_1_title_2_cosine_amin',
 'w2v_title_1_description_2_cosine_amin',
 'w2v_nouns_1_nouns_2_all_manhattan_all',
 'w2v_title_1_title_2_all_manhattan_all',
 'w2v_title_1_title_2_cokurtosis_kurtosis',
 'w2v_title_1_title_2_corr_amin',
 'w2v_title_1_title_2_cosine_amax',
 'w2v_title_1_title_2_euclidean_amin',
 'w2v_title_1_title_2_euclidean_p25']

df_w2v = pd.read_csv('features_w2v_train.csv', usecols=w2v_cols, dtype=np.float32)



# In[25]:

print 'reading glove...'

glove_cols = ['glove_all_text_1_all_text_2_all_cosine_all',
 'glove_all_text_1_all_text_2_cosine_p75',
 'glove_description_1_title_2_all_cosine_all',
 'glove_description_1_title_2_all_manhattan_all',
 'glove_description_1_title_2_cosine_mean',
 'glove_description_1_title_2_manhattan_wmd_sym',
 'glove_nouns_1_nouns_2_all_cosine_all',
 'glove_title_1_description_2_all_cosine_all',
 'glove_title_1_description_2_all_manhattan_all',
 'glove_title_1_description_2_cosine_mean',
 'glove_title_1_description_2_manhattan_wmd_sym',
 'glove_title_1_title_2_all_cosine_all',
 'glove_title_1_title_2_cosine_amin',
 'glove_title_1_title_2_cosine_kurtosis',
 'glove_title_1_title_2_euclidean_amin',
 'glove_title_1_title_2_manhattan_wmd_mean']

df_glove = pd.read_csv('features_glove_train.csv', usecols=glove_cols, dtype=np.float32)


# In[16]:

print 'reading attributes...'

attrs_cols = ['attrs_pairs_cosine_tfidf_svd',
 'attrs_pairs_dot_tfidf_svd',
 'attrs_pairs_euclidean_tfidf_svd',
 'attrs_pairs_manhattan_tfidf_svd',
 'attrs_values_dot_tfidf',
 'attrs_values_dot_tfidf_svd',
 'attrs_values_jaccard_reg',
 'attrs_values_num_match']

df_attrs = pd.read_csv('features_attrs_train.csv', usecols=attrs_cols, dtype=np.float32)


# In[17]:

print 'reading w2v_wmd...'

w2v_wmd_cols = ['w2v_all_text_1_all_text_2_euclid_wmd_mean',
 'w2v_all_text_1_all_text_2_euclid_wmd_sym',
 'w2v_description_1_description_2_euclid_wmd_mean',
 'w2v_description_1_title_2_manhattan_wmd_sym',
 'w2v_title_1_description_2_manhattan_wmd_sym',
 'w2v_title_1_title_2_euclid_wmd_sym',
 'w2v_title_1_title_2_manhattan_wmd_mean']

df_w2v_wmd = pd.read_csv('features_w2v_wmd_train.csv', usecols=w2v_wmd_cols, dtype=np.float32)


# In[19]:

print 'reading text...'

text_cols = ['text_all_bm25_dot',
 'text_all_bm25_svd_diff_8',
 'text_all_count_dot',
 'text_all_intersect',
 'text_all_tfidf_cosine',
 'text_all_tfidf_dot',
 'text_desc_bm25_cosine_svd',
 'text_desc_bm25_dot_svd',
 'text_desc_bm25_euclidean_svd',
 'text_desc_bm25_svd_diff_3',
 'text_title_bm25_cosine',
 'text_title_bm25_dot',
 'text_title_bm25_dot_svd',
 'text_title_bm25_euclidean_svd',
 'text_title_bm25_svd_diff_0',
 'text_title_bm25_svd_diff_1',
 'text_title_bm25_svd_diff_2',
 'text_title_count_dot_svd',
 'text_title_tfidf_dot',
 'text_title_tfidf_manhattan_svd']

df_text = pd.read_csv('features_text_train.csv', usecols=text_cols, dtype=np.float32)


# In[26]:

print 'reading common_tokens...'

common_tokens_cols = ['neg_common_tokens_sum',
 'pos_diff_tokens_sum',
 'svd_all_text_0', 'svd_all_text_1', 'svd_all_text_2', 'svd_all_text_3', 'svd_all_text_4',
 'svd_all_text_5', 'svd_all_text_6', 'svd_all_text_7',
 'svd_desc_common_0', 'svd_desc_common_1', 'svd_desc_common_2', 'svd_desc_common_4', 'svd_desc_common_5',
 'svd_desc_diff_0', 'svd_desc_diff_1', 'svd_desc_diff_2', 'svd_desc_diff_3', 'svd_desc_diff_4', 'svd_desc_diff_5',
 'svd_title_common_0', 'svd_title_common_1', 'svd_title_common_2', 'svd_title_common_3', 'svd_title_common_4',
 'svd_title_diff_0', 'svd_title_diff_1', 'svd_title_diff_2', 'svd_title_diff_3',
 'svm_all_text_common', 'svm_desc_common', 'svm_desc_diff',
 'svm_title_both', 'svm_title_common', 'svm_title_diff']

df_common_tokens = pd.read_csv('features_common_tokens_train.csv', usecols=common_tokens_cols, dtype=np.float32)


# In[27]:

print 'reading whash...'

whash_cols = ['whash_hamming_75p', 'whash_hamming_max', 'whash_hamming_skew', 'whash_hamming_std']
df_whash = pd.read_csv('features_whash_train.csv', usecols=whash_cols, dtype=np.float32)

# In[27]:

print 'reading dimred...'

df_dimred = pd.read_csv('features_dimred_train.csv', dtype=np.float32)
dimred_cols = list(df_dimred.columns)

# In[27]:

print 'reading ssim...'

df_ssim = pd.read_csv('features_ssim_train.csv', dtype=np.float32)
ssim_cols = list(df_ssim.columns)

# In[12]:

print 'reading misspelling train...'

misspelling_cols = ['all_mistakes_cnt_diff',
 'all_text_CommaWhitespaceRule_union_norm',
 'all_text_MorfologikRussianSpellerRule_union',
 'all_text_MorfologikRussianSpellerRule_union_norm',
 'all_text_MultipleWhitespaceRule_union',
 'all_text_MultipleWhitespaceRule_union_norm',
 'all_text_UppercaseSentenceStartRule_union_norm',
 'all_text_spelling_cosine', 'all_text_spelling_dist', 'all_text_spelling_jaccard',
 'description_CommaWhitespaceRule_dist',
 'description_MorfologikRussianSpellerRule_union_norm',
 'description_spelling_cosine',
 'title_MorfologikRussianSpellerRule_union_norm',
 'title_spelling_cosine']

df_misspellings = pd.read_csv('features_misspellings_train.csv', usecols=misspelling_cols, dtype=np.float32)
df_misspellings.fillna(0, inplace=1)

# In[32]:
print 'reading imp_tokens train...'

imp_tokens_cols = ['all_important_tokens_cos',
 'digits14_cos', 'en+ru_translated_digits_mix_cos',
 'en_digits_mix_cos', 'en_only14_cos']

df_imp_tokens = pd.read_csv('features_important_tokens_train.csv', usecols=imp_tokens_cols, dtype=np.float32)



# In[32]:

df_train = pd.concat([df_diffs_chargrams, df_fuzzy, df_imagemagick, df_imagehash, 
                      df_java, df_w2v, df_glove, df_attrs,
                      df_w2v_wmd, df_text, df_common_tokens, df_whash,
                      df_dimred, df_ssim, df_misspellings, df_imp_tokens], axis=1)

features = list(df_train.columns)

# In[35]:

del df_diffs_chargrams, df_fuzzy, df_imagemagick, df_imagehash
del df_java, df_w2v, df_glove, df_attrs
del df_w2v_wmd, df_text, df_common_tokens, df_whash
del df_dimred, df_ssim, df_misspellings, df_imp_tokens


# In[21]:

X_full = df_train[features].values
del df_train

## 

cv = KFold(len(y_full), n_folds=3, shuffle=True, random_state=42)
train_preds = np.zeros(y_full.shape)

scores = []

for train, val in cv:
    dtrain = xgb.DMatrix(X_full[train], label=y_full[train], feature_names=features, missing=np.nan)
    dval = xgb.DMatrix(X_full[val], label=y_full[val], feature_names=features, missing=np.nan)
    watchlist = [(dtrain, 'train'), (dval, 'val')]

    model = xgb.train(xgb_pars, dtrain, num_boost_round=n_estimators, verbose_eval=1, 
                      evals=watchlist)

    y_pred = model.predict(dval)
    train_preds[val] = y_pred
    score = roc_auc_score(y_full[val], y_pred)
    scores.append(score)
    del dtrain, dval, watchlist, model

print 'mean auc:', np.mean(scores)    

mscore = np.mean(scores)
mscore = int(mscore * 10000)

filename = 'xgb_model_9_top400_%s_%s' % (seed, mscore)
print 'saving train scores to', filename

train_res = pd.DataFrame({'probability': train_preds})
train_res.to_csv(filename + '_train.csv.gz', index=False, compression='gzip')

settings = {
    'chargram_cols': chargram_cols,
    'fuzzy_cols': fuzzy_cols,
    'imagemagick_cols': imagemagick_cols,
    'imagehash_cols': imagehash_cols,
    'java_image_cols': java_image_cols,
    'glove_cols': glove_cols,
    'w2v_cols': w2v_cols,
    'attrs_cols': attrs_cols,
    'w2v_wmd_cols': w2v_wmd_cols,
    'text_cols': text_cols,
    'common_tokens_cols': common_tokens_cols,
    'dimred_cols': dimred_cols,
    'whash_cols': whash_cols,
    'ssim_cols': ssim_cols,
    'imp_tokens_cols': imp_tokens_cols,
    'misspelling_cols': misspelling_cols,
    'score': np.mean(scores),
    'xgb_params': xgb_pars,
    'model': 'xgb',
    'n_estimators': n_estimators,
    'seed': seed,
}


with file(filename + '_settings.json', 'w') as f:
    json.dump(settings, f, indent=2)

del dtrain, dval, watchlist



# In[23]:

dtrainfull = xgb.DMatrix(X_full, label=y_full, feature_names=features, missing=np.nan)
del X_full, y_full


# In[ ]:

model = xgb.train(xgb_pars, dtrainfull, num_boost_round=n_estimators, verbose_eval=10, 
                  evals=[(dtrainfull, 'train')])

del dtrainfull



def create_feature_map(fmap_filename, features):
    outfile = open(fmap_filename, 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()

create_feature_map(filename + '.fmap', features) 
model.dump_model(filename + '_model.dump', fmap=filename + '.fmap', with_stats=True)

# In[27]:
print 'reading chargrams...'
df_diffs_chargrams = pd.read_csv('features_chargrams_test.csv', usecols=chargram_cols, dtype=np.float32)

print 'reading fuzzy...'
df_fuzzy = pd.read_csv('features_fuzzy_test.csv', usecols=fuzzy_cols, dtype=np.float32)

print 'reading imagemagick...'
df_imagemagick = pd.read_csv('features_imagemagick_test.csv', usecols=imagemagick_cols, dtype=np.float32)

print 'reading imagehash...'
df_imagehash = pd.read_csv('features_imagehash_test.csv', usecols=imagehash_cols, dtype=np.float32)

print 'reading java-image...'
df_java = pd.read_csv('features_java-image_test.csv', usecols=java_image_cols, dtype=np.float32)

print 'reading w2v...'
df_w2v = pd.read_csv('features_w2v_test.csv', usecols=w2v_cols, dtype=np.float32)

print 'reading glove...'
df_glove = pd.read_csv('features_glove_test.csv', usecols=glove_cols, dtype=np.float32)

print 'reading attributes...'
df_attrs = pd.read_csv('features_attrs_test.csv', usecols=attrs_cols, dtype=np.float32)

print 'reading w2v_wmd...'
df_w2v_wmd = pd.read_csv('features_w2v_wmd_test.csv', usecols=w2v_wmd_cols, dtype=np.float32)

print 'reading text...'
df_text = pd.read_csv('features_text_test.csv', usecols=text_cols, dtype=np.float32)

print 'reading common_tokens...'
df_common_tokens = pd.read_csv('features_common_tokens_test.csv', usecols=common_tokens_cols, dtype=np.float32)

print 'reading whash...'
df_whash = pd.read_csv('features_whash_test.csv', usecols=whash_cols, dtype=np.float32)

print 'reading dimred...'
df_dimred = pd.read_csv('features_dimred_test.csv', usecols=dimred_cols, dtype=np.float32)

print 'reading ssim...'
df_ssim = pd.read_csv('features_ssim_test.csv', usecols=ssim_cols, dtype=np.float32)

print 'reading misspelling...'
df_misspellings = pd.read_csv('features_misspellings_test.csv', usecols=misspelling_cols, dtype=np.float32)
df_misspellings.fillna(0, inplace=1)

print 'reading imp_tokens...'
df_imp_tokens = pd.read_csv('features_important_tokens_test.csv', usecols=imp_tokens_cols, dtype=np.float32)

df_test  = pd.concat([df_diffs_chargrams, df_fuzzy, df_imagemagick, df_imagehash, 
                      df_java, df_w2v, df_glove, df_attrs,
                      df_w2v_wmd, df_text, df_common_tokens, df_whash,
                      df_dimred, df_ssim, df_misspellings, df_imp_tokens], axis=1)

# In[35]:

del df_diffs_chargrams, df_fuzzy, df_imagemagick, df_imagehash
del df_java, df_w2v, df_glove, df_attrs
del df_w2v_wmd, df_text, df_common_tokens, df_whash
del df_dimred, df_ssim, df_misspellings, df_imp_tokens


# In[41]:
X_test = df_test[features].values
dtest = xgb.DMatrix(X_test, feature_names=features, missing=np.nan)
del df_test


# In[46]:
ids = pd.read_csv('../input/ItemPairs_test.csv', usecols=['id'], dtype=np.int)
test_ids = ids.id.values
y_result = model.predict(dtest)


# In[58]:
submission = pd.DataFrame({'id': test_ids, 'probability': y_result})
submission.to_csv(filename + '_test.csv.gz', index=False, compression='gzip')