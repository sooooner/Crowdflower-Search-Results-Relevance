#-*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from scipy.sparse import hstack

from sklearn.svm import SVC
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import text

from imblearn.over_sampling import SVMSMOTE
from imblearn.pipeline import Pipeline

from utility.utility import  similarlity_stack

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, type=str, help="mode select:'eda' or 'sampling'")
ap.add_argument("--ensemble", required=False, type=bool, help="Whether to ensemble or not")
args = ap.parse_args()

def Predict(mode='eda'):
    '''
    mode : string, 'eda' or 'sampling'
    '''
    if mode=='eda':
        train = pd.read_csv('./data/preprocessed_eda_train.csv')
    elif mode=='sampling':
        train = pd.read_csv('./data/preprocessed_train.csv')
    else:
        raise  ReeborgError("select only one of two modes: 'eda' and 'sampling'")
    train = train.drop_duplicates(['query_preprocessed', 'product_title_preprocessed'])
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
                               ngram_range=(1, 3), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words=stop_words)

    tfv.fit(train_query + train_title)
    X_train = hstack([tfv.transform(train_query), tfv.transform(train_title)])
    X_test = hstack([tfv.transform(test_query), tfv.transform(test_title)])

    sim = similarlity_stack()
    if mode=='eda':
        svd = TruncatedSVD(n_components=200)
        scl = StandardScaler(with_mean=False)
        svm =  SVC(C=10, gamma="auto", kernel="rbf", class_weight=None ,probability=True)
        clf = Pipeline([('FeatureUnion', FeatureUnion( [('svd', svd), ('sim', sim)] )),\
                            ('scl', scl),\
                            ('svm', svm)])
    elif mode=='sampling':
        svd = TruncatedSVD(n_components=200)
        scl = StandardScaler(with_mean=False)
        svm =  SVC(C=10, gamma="auto", kernel="rbf", class_weight=None ,probability=True)
        sampling = SVMSMOTE(svm_estimator=svm, k_neighbors=4)
        clf = Pipeline([('FeatureUnion', FeatureUnion( [('svd', svd), ('sim', sim)] )),\
                                  ('scl', scl),\
                                  ('sampling', sampling),\
                                  ('svm', svm)])      

    clf.fit(X_train, y)
    preds = clf.predict(X_test)
    pred_probas = clf.predict_proba(X_test)
    
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission_probas = pd.DataFrame(pred_probas, index=idx)
    
    return submission, submission_probas

def Ensemble(*pred_probas):
    submission = 0 
    for pred_proba in pred_probas:
        submission += pred_proba
    submission /= len(pred_probas)

    submission = submission.reset_index().rename(columns={"index": "id"})
    
    submission['prediction'] = submission[[0, 1, 2, 3]].idxmax(axis=1)+1
    submission.drop([0, 1, 2, 3], axis=1, inplace=True)
    
    submission.to_csv("./submission/submission_ensemble123.csv", index=False)
    
if __name__=="__main__":
    if not os.path.exists("./submission"):
        os.makedirs("./submission") 
    if args.ensemble:
        _, submission_probas_eda = Predict(mode='eda')
        _, submission_probas_sampling = Predict(mode='sampling')
        Ensemble(submission_probas_eda, submission_probas_sampling)
    else:
        submission, submission_proba = Predict(mode=args.mode)
        submission.to_csv("./submission/submission.csv", index=False)
        submission_proba.to_csv("./submission/submission_proba.csv", index=False)
    










