#-*- coding:utf-8 -*-
import os
import time
import datetime
import dill as pickle
import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from os.path import dirname, basename, join
from itertools import product
from nltk.corpus import stopwords
from scipy.sparse import hstack

from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction import text

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SVMSMOTE

from utility.utility import metric, similarlity_stack, plot_multiclass_roc_prc
from utility.processing import processer



class gridsearchCV(object):
    def __init__(self, data, target, params, filename, sample=False):
        self.data = data
        self.target = target
        self.params = list(product(*params))
        self.filename = filename
        self.sample = sample
        self.output_filename = join(dirname(self.filename), 'sample_%s_'%str(self.sample) + basename(self.filename))
        self.number_of_total_iter = len(self.params)
        self.stop_words = text.ENGLISH_STOP_WORDS.union(set(stopwords.words('english')))

    def _fit_and_score(self, tfv, clf, smt, svm, train_idx, dev_idx): 
        '''
        fitting data to pipeline and scoring
        return : tuple, 
                 quadratic weighted kappa, average auc score, auc score per rating
        '''
        train_idx = self.data.index.unique()[train_idx]
        dev_idx = self.data.index.unique()[dev_idx]
    
        train, dev = self.data.loc[train_idx], self.data.loc[dev_idx]
        y, y_dev = self.target.loc[train_idx], self.target.loc[dev_idx]
    
        dev = dev.loc[~dev.index.duplicated(keep='first')]
        y_dev = y_dev.loc[~y_dev.index.duplicated(keep='first')]
    
        rand_idx = np.random.permutation(np.arange(train.shape[0]))
        train = train.iloc[rand_idx]
        y = y.iloc[rand_idx]
    
        train_query = list(train.apply(lambda x:'%s' % x['query_preprocessed'], axis=1))
        train_title = list(train.apply(lambda x:'%s' % x['product_title_preprocessed'], axis=1))
    
        dev_query = list(dev.apply(lambda x:'%s' % x['query_preprocessed'], axis=1))
        dev_title = list(dev.apply(lambda x:'%s' % x['product_title_preprocessed'], axis=1))
    
        tfv.fit(train_query + train_title)
        X_train = hstack([tfv.transform(train_query), tfv.transform(train_title)])
        X_dev = hstack([tfv.transform(dev_query), tfv.transform(dev_title)])
        del train_query, train_title, dev_query, dev_title
        
        X_scaled_train = clf.fit_transform(X_train)
        X_scaled_dev = clf.transform(X_dev)
    
        if self.sample:
            X_samp, y_samp = smt.fit_sample(X_scaled_train, y)
        else:
            X_samp, y_samp = X_scaled_train, y
    
        svm_result = svm.fit(X_samp, y_samp)
        svm_pred_dev = svm_result.predict(X_scaled_dev)
        svm_pred_proba_dev = svm_result.predict_proba(X_scaled_dev)
        
        average_auc_score, auc_score_per_rating = metric.pr_auc_score(y_dev, svm_pred_proba_dev)
        return metric.quadratic_weighted_kappa(y_dev, svm_pred_dev), average_auc_score, auc_score_per_rating
    
    # def _fit_and_score(self, tfv, clf, smt, svm, train_idx, dev_idx): 
    #     return 0.713, 0.2, {'rating 1' : .12, 'rating 2' : .32, 'rating 3' : .52, 'rating 4' : .76}        
        
    def parallel_k_fold(self, n_jobs, n_splits, param, k=0):
        '''
        write cv_kappa_scores, cv_pr_auc_scores, param_grid
        '''
        # n_components, C, gamma, class_weight, kernel, min_df
        param_grid = {'n_components' : param[0], 'C' : param[1], \
                      'gamma' : param[2], 'class_weight' : param[3], \
                      'kernel' : param[4], 'min_df' : param[5]}
        
        _k_fold = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.1, train_size=0.9)
        parallel = Parallel(n_jobs=n_jobs)
        
        start = time.time()
        now = time.localtime()
        print('Start fitting %d sampling %s models at %02d:%02d:%02d with'%(n_splits, str(self.sample), now.tm_hour, now.tm_min, now.tm_sec), param_grid, 'params %d at a time ...'%n_jobs)
    
        tfv = text.TfidfVectorizer(min_df=param_grid['min_df'],  max_features=None, strip_accents='unicode', analyzer='word',\
                                   token_pattern=r'\w{1,}', ngram_range=(1, 3), use_idf=True, smooth_idf=True, \
                                   sublinear_tf=True, stop_words=self.stop_words)
        sim = similarlity_stack()
        svd = TruncatedSVD(n_components = param_grid['n_components'])
        scl = StandardScaler(with_mean=False)
        clf = Pipeline([('FeatureUnion', FeatureUnion( [('svd', svd), ('sim', sim)] )),\
                        ('scl', scl)])
        svm = SVC(C=param_grid['C'], gamma=param_grid['gamma'], class_weight=param_grid['class_weight'], \
                  kernel=param_grid['kernel'], probability=True)
        smt = SVMSMOTE(sampling_strategy='not majority', k_neighbors=4, svm_estimator=svm)

        score_list = parallel(
            delayed(self._fit_and_score)(clone(tfv), clone(clf), clone(smt), clone(svm), train_index, test_index) 
            for train_index, test_index in _k_fold.split(self.data.index.unique(), self.target.loc[~self.target.index.duplicated(keep='first')])
        )
    
        cv_kappa_scores, cv_pr_auc_scores, cv_auc_score_per_rating = [], [], []
        for kappas, auc, auc_dict in score_list:
            cv_kappa_scores.append(kappas)
            cv_pr_auc_scores.append(auc)
            cv_auc_score_per_rating.append(auc_dict)
    
        r_static = [[] for _ in range(4)]
        for auc_score_per_rating in cv_auc_score_per_rating:
            for i in range(4):
                r_static[i].append(auc_score_per_rating['rating %d'%(i+1)])
    
        w_line = str(param)[1:-1] + ', %.4f, %.4f, %.4f, %.4f, '%(np.mean(cv_kappa_scores), np.std(cv_kappa_scores), np.mean(cv_pr_auc_scores), np.std(cv_pr_auc_scores)) + "%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f"%(np.mean(r_static[0]), np.std(r_static[0]), np.mean(r_static[1]), np.std(r_static[1]), np.mean(r_static[2]), np.std(r_static[2]), np.mean(r_static[3]), np.std(r_static[3])) + "\n"
        
        with open(join(dirname(self.filename), 'result_%d_%s_temp.txt'%(k+1, str(self.sample))), 'w', encoding="utf-8") as f:
            f.write(w_line)
            
        print('%d/%dsample %s'%(k+1, self.number_of_total_iter, str(self.sample)), param_grid, \
              'kappa:%.2f, auc:%.2f,'% (np.mean(cv_kappa_scores), np.mean(cv_pr_auc_scores)), \
              'r1:%.2f, r2:%.2f, r3:%.2f, r4:%.2f'%(np.mean(r_static[0]), np.mean(r_static[1]), np.mean(r_static[2]), np.mean(r_static[3])), \
              'time: %s'%(str(datetime.timedelta(seconds=int(time.time() - start)))))
              
        # time.sleep(120)
    
    def gridsearchcv_writer(self, n_jobs):
        parallel = Parallel(n_jobs=n_jobs)
        parallel(
            delayed(self.parallel_k_fold)(n_jobs=-1, n_splits=2, param=param, k=i) 
            for i, param in enumerate(self.params)
        )
            
        auc_static = ['auc_rating_%d_mean,auc_rating_%d_std' % (i, i) for i in range(1, 5)]
        w_list = "n_components,C,gamma,class_weight,kernel,min_df,kappa_score_mean,kappa_score_std,pr_auc_score_mean,pr_auc_score_std," + ','.join(auc_static) +'\n'
        
        with open(self.output_filename, 'w', encoding='utf-8') as fw:
            fw.write(w_list)
            for k in range(self.number_of_total_iter):
                with open(join(dirname(self.filename), 'result_%d_%s_temp.txt'%(k+1, str(self.sample))), 'r', encoding="utf-8") as f:
                    line = f.readline()
                    fw.write(line)
                os.remove(join(dirname(self.filename), 'result_%d_%s_temp.txt'%(k+1, str(self.sample))))

if __name__=="__main__":
    train = pd.read_csv('./data/preprocessed_eda_train.csv')
    train.set_index('id', inplace=True)
    
    target = train['median_relevance']
    train = train[['query_preprocessed', 'product_title_preprocessed']]
    
    # n_components, C, gamma, class_weight, kernel, min_df
    params = [[100, 200, 300], [1, 10], ['auto'], [None], ['rbf'], [3, 7]]
    # params = [[230], [10], ['auto'], [None], ['rbf'], [3, 7]]
    
    n_jobs = 6
    filename = './gridsearch/result.txt'
    for sample in [False, True]:
        gridsearchcv = gridsearchCV(train, target, params, filename, sample)
        gridsearchcv.gridsearchcv_writer(n_jobs)
    
    

























