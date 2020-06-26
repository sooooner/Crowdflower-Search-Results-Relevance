#-*- coding:utf-8 -*-
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
from sklearn.metrics import classification_report, roc_curve, mean_squared_error, auc, confusion_matrix, make_scorer
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.feature_extraction import text

from imblearn.combine import *
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from imblearn.pipeline import Pipeline

from utility.utility import *
from utility.processing import processer


def kf_cv(X, Y, train_idx, dev_idx, param_grid):
    train, dev = X.iloc[train_idx], X.iloc[dev_idx]
    y, y_dev = Y.iloc[train_idx], Y.iloc[dev_idx]

    train_query = list(train.apply(lambda x:'%s' % x['query_preprocessed'], axis=1))
    train_title = list(train.apply(lambda x:'%s' % x['product_title_preprocessed'], axis=1))

    dev_query = list(dev.apply(lambda x:'%s' % x['query_preprocessed'], axis=1))
    dev_title = list(dev.apply(lambda x:'%s' % x['product_title_preprocessed'], axis=1))

    stop_words = text.ENGLISH_STOP_WORDS.union(['http','www','img','border','color','style','padding','table',\
                                                'font', '', 'thi','inch','ha','width','height',\
                                                '0','1','2','3','4','5','6','7','8','9'])

    stop_words = text.ENGLISH_STOP_WORDS.union(set(stopwords.words('english')))
    tfv = text.TfidfVectorizer(min_df=param_grid['min_df'],  max_features=None, strip_accents='unicode', analyzer='word',\
                               token_pattern=r'\w{1,}', ngram_range=(1, 3), use_idf=True, smooth_idf=True, \
                               sublinear_tf=True, stop_words = stop_words).fit(train_query + train_title)

    X_train = hstack([tfv.transform(train_query), tfv.transform(train_title)])
    X_dev = hstack([tfv.transform(dev_query), tfv.transform(dev_title)])

    sim = similarlity_stack()
    svd = TruncatedSVD(n_components = param_grid['n_components'])
    scl = StandardScaler(with_mean=False)
    smt = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
    svm = SVC(C=param_grid['C'], gamma=param_grid['gamma'], class_weight=param_grid['class_weight'], \
              kernel=param_grid['kernel'], probability=True)

    X_sim_train = sim.fit_transform(X_train, y)
    X_svd_train = svd.fit_transform(X_train)
    X_stacked_train = hstack([X_svd_train, X_sim_train])
    X_scaled_train = scl.fit_transform(X_stacked_train)

    X_sim_dev = sim.transform(X_dev)       
    X_svd_dev = svd.transform(X_dev)
    X_stacked_dev = hstack([X_svd_dev, X_sim_dev])
    X_scaled_dev = scl.transform(X_stacked_dev)

    X_samp, y_samp = smt.fit_sample(X_scaled_train, y)

    svm_result = svm.fit(X_samp, y_samp)
    svm_pred_dev = svm_result.predict(X_scaled_dev)
    svm_pred_proba_dev = svm_result.predict_proba(X_scaled_dev)

    return metric.quadratic_weighted_kappa(y_dev, svm_pred_dev), metric.pr_auc_score(y_dev, svm_pred_proba_dev)

def gridsearchcv(X, Y, parmas):
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    parmas = list(product(*parmas))
    
    param_list = []
    cv_kappa_score_mean, cv_kappa_score_std = [], []
    cv_pr_auc_score_mean, cv_pr_auc_score_std = [], []
    
    for param in parmas:
        start = time.time()
        # n_components, C, gamma, class_weight, kernel, min_df
        param_grid = {'n_components' : param[0], 'C' : param[1], \
                      'gamma' : param[2], 'class_weight' : param[3], \
                      'kernel' : param[4], 'min_df' : param[5]}
        cv_kappa_scores, cv_pr_auc_scores = [], []
        
        for train_idx, dev_idx in kf.split(X, Y):
            kappa, pr_auc = kf_cv(X, Y, train_idx, dev_idx, param_grid)
            cv_kappa_scores.append(kappa)
            cv_pr_auc_scores.append(pr_auc)
            
        param_list.append(param_grid)
        cv_kappa_score_mean.append(np.mean(cv_kappa_scores))
        cv_kappa_score_std.append(np.std(cv_kappa_scores))
        cv_pr_auc_score_mean.append(np.mean(cv_pr_auc_scores))
        cv_pr_auc_score_std.append(np.std(cv_pr_auc_scores))

        print(param_grid, "kappa_score : %.2f, pr_auc_scores : %.2f"% (np.mean(cv_kappa_scores), np.mean(cv_pr_auc_scores)))
        print("time : %s"%(str(datetime.timedelta(seconds=int(time.time() - start)))))  
        
    results = pd.DataFrame({'param' : param_list, \
                            'kappa_score_mean' : cv_kappa_score_mean, \
                            'kappa_score_std' : cv_kappa_score_std, \
                            'pr_auc_score_mean' : cv_pr_auc_score_mean, \
                            'pr_auc_score_std' : cv_pr_auc_score_std})
    
    return results

def main():
    train = pd.read_csv('./data/preprocessed_train.csv')
    test = pd.read_csv('./data/preprocessed_test.csv')
    idx = test.id.values.astype(int)
    y = train.median_relevance.values

    train_query = list(train.apply(lambda x:'%s' % x['query_preprocessed'], axis=1))
    train_title = list(train.apply(lambda x:'%s' % x['product_title_preprocessed'], axis=1))

    test_query = list(test.apply(lambda x:'%s' % x['query_preprocessed'], axis=1))
    test_title = list(test.apply(lambda x:'%s' % x['product_title_preprocessed'], axis=1))

    slicer = int(len(train) * 0.9)
    y_dev = y[slicer:]
    y = y[:slicer]

    dev_query = train_query[slicer:]
    train_query = train_query[:slicer]
    dev_title = train_title[slicer:]
    train_title = train_title[:slicer]

    stop_words = text.ENGLISH_STOP_WORDS.union(['http','www','img','border','color','style','padding','table','font', \
                                                'thi','inch','ha','width','height','0','1','2','3','4','5','6','7','8','9'])
    stop_words = text.ENGLISH_STOP_WORDS.union(set(stopwords.words('english')))

    tfv = text.TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}', \
                          ngram_range=(1, 3), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words = stop_words)

    full_data = train_query + train_title
    tfv.fit(full_data)

    X_train = hstack([tfv.transform(train_query), tfv.transform(train_title)])
    X_dev = hstack([tfv.transform(dev_query), tfv.transform(dev_title)])
    X_test = hstack([tfv.transform(test_query), tfv.transform(test_title)])

    sim = similarlity_stack()
    svd = TruncatedSVD()
    scl = StandardScaler(with_mean=False)
    sampling = OneSidedSelection()
    # sampling = SMOTE()
    svm =  SVC(probability=True)

    clf = Pipeline([('FeatureUnion', FeatureUnion( [('svd', svd), ('sim', sim)] )),\
                    ('scl', scl),\
                    ('sampling', sampling),\
                    ('svm', svm)])
                    
    svd__n_components = [200, 230, 250]
    sampling__sampling_strategy = ['auto', 'majority']
    # sampling__sampling_strategy = ['minority ', 'auto']
    svm__gamma = ['auto', 'scale']
    svm__class_weight = ['balanced', None]
    svm__C = [10, 100]

    scoring = {'kappa': make_scorer(metric.quadratic_weighted_kappa, greater_is_better = True), \
               'pr_auc': make_scorer(metric.pr_auc_score, greater_is_better = True, needs_proba=True, average='micro')}
               
    param_grid = {'FeatureUnion__svd__n_components' : svd__n_components,\
                  'sampling__sampling_strategy' : sampling__sampling_strategy,\
                  'svm__gamma' : svm__gamma,\
                  'svm__class_weight' : svm__class_weight,\
                  'svm__C': svm__C}
               
    import os
    cv = os.cpu_count()
    if cv > 5:
        cv = 5
    model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=scoring, refit='kappa', verbose=10, n_jobs=-1, iid=True, cv=cv)
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
    dev_preds = best_model.predict(X_dev)
    model_preds = best_model.predict(X_test)

    sampled_svm_grid_submission = pd.DataFrame({"id": idx, "prediction": model_preds})
    if not os.path.exists("./submission"):
        os.makedirs("./submission")
    sampled_svm_grid_submission.to_csv("./submission/imbalance_svm.csv", index=False)
    
    print(confusion_matrix(y_dev, dev_preds))
    print(classification_report(y_dev, dev_preds))
    plot_multiclass_roc_prc(best_model, X_dev, y_dev, save=True)


if __name__=="__main__":
    df_train = pd.read_csv('./data/preprocessed_train.csv')
    Y = df_train['median_relevance']
    test = pd.read_csv('./data/preprocessed_test.csv')
    idx = test.id.values.astype(int)

    # n_components, C, gamma, class_weight, kernel, min_df
    # parmas = [[230], [100, 1000], ['auto'], ['balanced', None], ['rbf', 'poly'], [3, 5, 7]]
    parmas = [[230], [100], ['auto'], [None], ['rbf'], [5]]

    result = gridsearchcv(df_train, Y, parmas)
        
    result.to_csv("./gridsearch/resultss.csv", index=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


