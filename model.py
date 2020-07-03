#-*- coding:utf-8 -*-
import os
import dill as pickle
import numpy as np
import pandas as pd
import nltk
import time
import datetime

# nltk.download('stopwords')
from itertools import product
from nltk.corpus import stopwords
from scipy.sparse import hstack

from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.metrics import classification_report, roc_curve, mean_squared_error, auc, confusion_matrix, make_scorer
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, KFold, StratifiedShuffleSplit
from sklearn.feature_extraction import text

from imblearn.combine import *
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from imblearn.pipeline import Pipeline

from utility.utility import *
from utility.processing import processer

from joblib import Parallel, delayed




def main():
    train = pd.read_csv('./data/preprocessed_train.csv')
    test = pd.read_csv('./data/preprocessed_test.csv')
    idx = test.id.values.astype(int)
    y = train.median_relevance.values

    train_query = list(train.apply(lambda x:'%s' % x['query_preprocessed'], axis=1))
    train_title = list(train.apply(lambda x:'%s' % x['product_title_preprocessed'], axis=1))

    test_query = list(test.apply(lambda x:'%s' % x['query_preprocessed'], axis=1))
    test_title = list(test.apply(lambda x:'%s' % x['product_title_preprocessed'], axis=1))

    stop_words = text.ENGLISH_STOP_WORDS.union(['http','www','img','border','color','style','padding','table','font', \
                                                'thi','inch','ha','width','height','0','1','2','3','4','5','6','7','8','9'])
    stop_words = text.ENGLISH_STOP_WORDS.union(set(stopwords.words('english')))

    tfv = text.TfidfVectorizer(min_df=7,  max_features=None, strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}', \
                               ngram_range=(1, 3), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = stop_words)

    full_data = train_query + train_title
    tfv.fit(full_data)

    X_train = hstack([tfv.transform(train_query), tfv.transform(train_title)])
    X_test = hstack([tfv.transform(test_query), tfv.transform(test_title)])

    sim = similarlity_stack()
    svd = TruncatedSVD()
    scl = StandardScaler(with_mean=False)
    sampling = SVMSMOTE(svm_estimator=SVC(), k_neighbors=4)
    svm =  SVC(probability=True)

    clf = Pipeline([('FeatureUnion', FeatureUnion( [('svd', svd), ('sim', sim)] )),\
                    ('scl', scl),\
                    ('sampling', sampling),\
                    ('svm', svm)])
                    
    svd__n_components = [230]
    sampling__sampling_strategy = ['not majority']
    svm__gamma = ['auto']
    svm__kernel = ['rbf']
    svm__class_weight = [None]
    svm__C = [3]
               
    param_grid = {'FeatureUnion__svd__n_components' : svd__n_components,\
                  'sampling__sampling_strategy' : sampling__sampling_strategy,\
                  'svm__gamma' : svm__gamma,\
                  'svm__class_weight' : svm__class_weight,\
                  'svm__kernel' : svm__kernel,\
                  'svm__C': svm__C}

    scoring = {'kappa': make_scorer(metric.quadratic_weighted_kappa, greater_is_better = True), \
               'pr_auc': make_scorer(metric.pr_auc_score, greater_is_better = True, needs_proba=True, average='micro')}
               
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=scoring, refit='kappa', verbose=10, n_jobs=5, iid=True, cv=cv)
    model.fit(X_train, y)

    results = pd.DataFrame(model.cv_results_)
    
    if not os.path.exists("./gridsearch"):
        os.makedirs("./gridsearch")
    results.to_csv("./gridsearch/results.csv", index=False)
    
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters :")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    best_model = model.best_estimator_
    best_model.fit(X_train, y)
    model_preds = best_model.predict(X_test)

    sampled_svm_grid_submission = pd.DataFrame({"id": idx, "prediction": model_preds})
    if not os.path.exists("./submission"):
        os.makedirs("./submission")
    sampled_svm_grid_submission.to_csv("./submission/imbalance_svmsmote.csv", index=False)


def _fit_and_score(tfv, clf, smt, svm, X, Y, train_idx, dev_idx): 
    train, dev = X.iloc[train_idx], X.iloc[dev_idx]
    y, y_dev = Y.iloc[train_idx], Y.iloc[dev_idx]

    train_query = list(train.apply(lambda x:'%s' % x['query_preprocessed'], axis=1))
    train_title = list(train.apply(lambda x:'%s' % x['product_title_preprocessed'], axis=1))

    dev_query = list(dev.apply(lambda x:'%s' % x['query_preprocessed'], axis=1))
    dev_title = list(dev.apply(lambda x:'%s' % x['product_title_preprocessed'], axis=1))

    tfv.fit(train_query + train_title)
    X_train = hstack([tfv.transform(train_query), tfv.transform(train_title)])
    X_dev = hstack([tfv.transform(dev_query), tfv.transform(dev_title)])

    X_scaled_train = clf.fit_transform(X_train)
    X_scaled_dev = clf.transform(X_dev)

    X_samp, y_samp = smt.fit_sample(X_scaled_train, y)

    svm_result = svm.fit(X_samp, y_samp)
    svm_pred_dev = svm_result.predict(X_scaled_dev)
    svm_pred_proba_dev = svm_result.predict_proba(X_scaled_dev)

    average_auc_score, auc_score_per_rating = metric.pr_auc_score(y_dev, svm_pred_proba_dev)
    return metric.quadratic_weighted_kappa(y_dev, svm_pred_dev), average_auc_score, auc_score_per_rating

# def _fit_and_score(tfv, clf, smt, svm, X, Y, train_idx, dev_idx): 
#     return 0.713, 0.2, {'rating 1' : .12, 'rating 2' : .32, 'rating 3' : .52, 'rating 4' : .76}

def parallel_k_fold(data, target, param, stop_words, k, number_of_total_iter):
    '''
    return cv_kappa_scores, cv_pr_auc_scores, param_grid
    '''
    # n_components, C, gamma, class_weight, kernel, min_df
    param_grid = {'n_components' : param[0], 'C' : param[1], \
                  'gamma' : param[2], 'class_weight' : param[3], \
                  'kernel' : param[4], 'min_df' : param[5]}
    
    n_splits = 4
    n_jobs = 2
    start = time.time()
    
    _k_fold = StratifiedShuffleSplit( n_splits=n_splits, test_size=0.1, train_size=0.9)
    parallel = Parallel(n_jobs=n_jobs)
    now = time.localtime()
    print('Start fitting %d models at %02d:%02d:%02d with'%(n_splits, now.tm_hour, now.tm_min, now.tm_sec), param_grid, 'params %d at a time ...'%n_jobs)

    tfv = text.TfidfVectorizer(min_df=param_grid['min_df'],  max_features=None, strip_accents='unicode', analyzer='word',\
                               token_pattern=r'\w{1,}', ngram_range=(1, 3), use_idf=True, smooth_idf=True, \
                               sublinear_tf=True, stop_words = stop_words)
    sim = similarlity_stack()
    svd = TruncatedSVD(n_components = param_grid['n_components'])
    scl = StandardScaler(with_mean=False)
    clf = Pipeline([('FeatureUnion', FeatureUnion( [('svd', svd), ('sim', sim)] )),\
                    ('scl', scl)])
    smt = SVMSMOTE(sampling_strategy='not majority', k_neighbors=4, svm_estimator=SVC(C=param_grid['C'], gamma=param_grid['gamma']))
    svm = SVC(C=param_grid['C'], gamma=param_grid['gamma'], class_weight=param_grid['class_weight'], \
              kernel=param_grid['kernel'], probability=True)
    
    score_list = parallel(
        delayed(_fit_and_score)(clone(tfv), clone(clf), clone(smt), clone(svm), data, target, train_index, test_index) 
        for train_index, test_index in _k_fold.split(data, target)
    )
    
    cv_kappa_scores, cv_pr_auc_scores, cv_auc_score_per_rating = [], [], []
    for kappas, auc, auc_dict in score_list:
        cv_kappa_scores.append(kappas)
        cv_pr_auc_scores.append(auc)
        cv_auc_score_per_rating.append(auc_dict)
    
    r_1, r_2, r_3, r_4 = [], [], [], []
    for auc_score_per_rating in cv_auc_score_per_rating:
        r_1.append(auc_score_per_rating['rating 1'])
        r_2.append(auc_score_per_rating['rating 2'])
        r_3.append(auc_score_per_rating['rating 3'])
        r_4.append(auc_score_per_rating['rating 4'])
    
    print('%d/%d'%(k+1, number_of_total_iter), param_grid, \
          'kappa: %.2f, pr_auc: %.2f,'% (np.mean(cv_kappa_scores), np.mean(cv_pr_auc_scores)), \
          'r1: %.2f, r2: %.2f, r3: %.2f, r4: %.2f'%(np.mean(r_1), np.mean(r_2), np.mean(r_3), np.mean(r_4)), \
          'time: %s'%(str(datetime.timedelta(seconds=int(time.time() - start)))))
    time.sleep(120)
    return np.mean(cv_kappa_scores), np.std(cv_kappa_scores), np.mean(cv_pr_auc_scores), np.std(cv_pr_auc_scores), \
           (np.mean(r_1), np.std(r_1), np.mean(r_2), np.std(r_2), np.mean(r_3), np.std(r_3), np.mean(r_4), np.std(r_4)), param_grid

def gridsearchcv(data, target, parmas):
    parmas = list(product(*parmas))
    stop_words = text.ENGLISH_STOP_WORDS.union(set(stopwords.words('english')))
    
    n_jobs = 2
    parallel = Parallel(n_jobs=n_jobs)
    number_of_total_iter = len(parmas)
    scores_statistic = parallel(
        delayed(parallel_k_fold)(data, target, param, stop_words, i, number_of_total_iter) 
        for i, param in enumerate(parmas)
    )
    
    param_list = []
    cv_kappa_score_mean, cv_kappa_score_std = [], []
    cv_pr_auc_score_mean, cv_pr_auc_score_std = [], []
    auc_rating_1_mean, auc_rating_2_mean, auc_rating_3_mean, auc_rating_4_mean = [], [], [], [] 
    auc_rating_1_std, auc_rating_2_std, auc_rating_3_std, auc_rating_4_std = [], [], [], [] 
    for kappa_mean, kappa_std, auc_mean, auc_std, auc_score_per_ratings, param_grid in scores_statistic:
        param_list.append(param_grid)
        cv_kappa_score_mean.append(kappa_mean)
        cv_kappa_score_std.append(kappa_std)
        cv_pr_auc_score_mean.append(auc_mean)
        cv_pr_auc_score_std.append(auc_std)
        for auc_list, statics in zip([auc_rating_1_mean, auc_rating_1_std, auc_rating_2_mean, auc_rating_2_std, \
                                      auc_rating_3_mean, auc_rating_3_std, auc_rating_4_mean, auc_rating_4_std], auc_score_per_ratings):
            auc_list.append(statics)
          
    param_n_components, param_C, param_gamma, param_class_weight, param_kernel, param_min_df = [], [], [], [], [], []
    for pram_dicts in param_list:
        param_n_components.append(pram_dicts['n_components'])
        param_C.append(pram_dicts['C'])
        param_gamma.append(pram_dicts['gamma'])
        param_class_weight.append(pram_dicts['class_weight'])
        param_kernel.append(pram_dicts['kernel'])
        param_min_df.append(pram_dicts['min_df'])
          
          
    results = pd.DataFrame({'param_n_components' : param_n_components,\
                            'param_C' : param_C,\
                            'param_gamma' : param_gamma,\
                            'param_class_weight' : param_class_weight,\
                            'param_kernel' : param_kernel,\
                            'param_min_df' : param_min_df,\
                            'kappa_score_mean' : cv_kappa_score_mean, \
                            'kappa_score_std' : cv_kappa_score_std, \
                            'pr_auc_score_mean' : cv_pr_auc_score_mean, \
                            'pr_auc_score_std' : cv_pr_auc_score_std, \
                            'auc_rating_1_mean' : auc_rating_1_mean, \
                            'auc_rating_1_std' : auc_rating_1_std, \
                            'auc_rating_2_mean' : auc_rating_2_mean, \
                            'auc_rating_2_std' : auc_rating_2_std, \
                            'auc_rating_3_mean' : auc_rating_3_mean, \
                            'auc_rating_3_std' : auc_rating_3_std, \
                            'auc_rating_4_mean' : auc_rating_4_mean, \
                            'auc_rating_4_std' : auc_rating_4_std})
    
    return results



if __name__=="__main__":
    df_train = pd.read_csv('./data/preprocessed_train.csv')
    test = pd.read_csv('./data/preprocessed_test.csv')
    idx = test.id.values.astype(int)
    
    Y = df_train['median_relevance']
    df_train = df_train[['query_preprocessed', 'product_title_preprocessed']]
    test = test[['query_preprocessed', 'product_title_preprocessed']]
    
    # n_components, C, gamma, class_weight, kernel, min_df
    parmas = [[230, 250], [1, 3, 5, 10], ['auto'], [None], ['rbf'], [3, 5, 7]]
    # parmas = [[230], [5], ['auto'], [None], ['rbf'], [7]]
    result = gridsearchcv(df_train, Y, parmas)
    
    import os
    if not os.path.exists("./gridsearch"):
        os.makedirs("./gridsearch")
    result.to_csv("./gridsearch/results_detail.csv", index=False)

    # main()
    
    
    
    


