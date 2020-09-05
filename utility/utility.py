#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_curve, auc, roc_auc_score
from imblearn.combine import *
from imblearn.over_sampling import *
from imblearn.under_sampling import * 
from sklearn.preprocessing import label_binarize
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score
from utility.processing import processer
from os import makedirs
from os.path import dirname, basename, join, exists

class metric():
    def __init__(self):
        pass
        
    # https://www.kaggle.com/triskelion/kappa-intuition
    @staticmethod
    def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
        assert(len(rater_a) == len(rater_b))
        if min_rating is None:
            min_rating = min(rater_a + rater_b)
        if max_rating is None:
            max_rating = max(rater_a + rater_b)
        num_ratings = int(max_rating - min_rating + 1)
        conf_mat = [[0 for i in range(num_ratings)]
                    for j in range(num_ratings)]
        for a, b in zip(rater_a, rater_b):
            conf_mat[a - min_rating][b - min_rating] += 1
        return conf_mat

    @staticmethod
    def histogram(ratings, min_rating=None, max_rating=None):
        if min_rating is None:
            min_rating = min(ratings)
        if max_rating is None:
            max_rating = max(ratings)
        num_ratings = int(max_rating - min_rating + 1)
        hist_ratings = [0 for x in range(num_ratings)]
        for r in ratings:
            hist_ratings[r - min_rating] += 1
        return hist_ratings

    @staticmethod
    def quadratic_weighted_kappa(y, y_pred):
        rater_a = y
        rater_b = y_pred
        min_rating=None
        max_rating=None
        rater_a = np.array(rater_a, dtype=int)
        rater_b = np.array(rater_b, dtype=int)
        assert(len(rater_a) == len(rater_b))
        if min_rating is None:
            min_rating = min(min(rater_a), min(rater_b))
        if max_rating is None:
            max_rating = max(max(rater_a), max(rater_b))
        conf_mat = metric.confusion_matrix(rater_a, rater_b, min_rating, max_rating)
        num_ratings = len(conf_mat)
        num_scored_items = float(len(rater_a))

        hist_rater_a = metric.histogram(rater_a, min_rating, max_rating)
        hist_rater_b = metric.histogram(rater_b, min_rating, max_rating)

        numerator = 0.0
        denominator = 0.0

        for i in range(num_ratings):
            for j in range(num_ratings):
                expected_count = (hist_rater_a[i] * hist_rater_b[j]
                                  / num_scored_items)
                d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
                numerator += d * conf_mat[i][j] / num_scored_items
                denominator += d * expected_count / num_scored_items

        return (1.0 - numerator / denominator)        
        
    @staticmethod
    def pr_auc_score(y, y_score, average='micro'):
        Y_bin = label_binarize(y, classes=[1, 2, 3, 4])
        average_precision = average_precision_score(Y_bin, y_score, average=average)
    #     # 특정 점수에 대한 rating을 목표로 할때
        ap={}
        for i in range(1, 5):
            ap['rating %d'%i] = average_precision_score(Y_bin[:, i-1], y_score[:, i-1])
        return average_precision, ap    
    
    @staticmethod
    def Euclidean_distance(A, B):
        return np.linalg.norm(A-B)
        
    @staticmethod
    def cos_sim(A, B):
        numerator = np.dot(A, B)
        denominator = np.linalg.norm(A)*np.linalg.norm(B)
        if denominator == 0:
            return 0.0
        return numerator/denominator
    
    @staticmethod
    def cos_distance(A, B):
        return 1 - metric.cos_sim(A, B)
        
    @staticmethod
    def text_sim(A, B):
        A_idx = np.where(A>0)[0]
        B_idx = np.where(B>0)[0]
        numerator = len(np.intersect1d(A_idx, B_idx))
        denominator = np.log(len(A_idx)) + np.log(len(B_idx))
        if denominator == 0:
            return 0.0
        return numerator/denominator
        
    @staticmethod
    def jaccard_sim(A, B):
        A_idx = np.where(A>0)[0]
        B_idx = np.where(B>0)[0]
        numerator = len(np.intersect1d(A_idx, B_idx))
        denominator = len(np.union1d(A_idx, B_idx))
        if denominator == 0:
            return 0.0
        return numerator/denominator
        
    @staticmethod
    def jaccard_distance(A, B):
        return 1 - metric.jaccard_sim(A, B)

# https://cloud.google.com/ai-platform/prediction/docs/custom-pipeline?hl=ko
class similarlity_stack(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.average = None
        self.boundary = None

    def fit(self, X ,y):
        return self
        
    def fit_transform(self, X, y):
        self.fit(X, y)
        sim_matrix = self.transform(X)

        self.average = np.zeros((len(np.unique(y)), 2))
        self.boundary = np.zeros((len(np.unique(y))-1, 2))
        for i, targe in enumerate(np.sort(np.unique(y))):
            self.average[i] = sim_matrix.toarray()[np.where(y == targe)].sum(axis=0)/len(np.where(y == targe)[0])
            # self.average[i] = sim_matrix[np.where(y == targe)].sum(axis=0)/len(np.where(y == targe)[0])
        for i in range(len(np.unique(y)))[:-1]:
            self.boundary[i] = (self.average[i] + self.average[i+1])/2
        return sim_matrix
        
    def transform(self, X, y=None):
        cos_sims = []
        jaccard_sims = []
        text_sims = []
        for row in X.tocsr():
            front, end = processer.split_array(row)
            cos_sims.append(metric.cos_sim(front, end))
            # jaccard_sims.append(metric.jaccard_sim(front, end))
        # return sparse.csr_matrix(np.matrix([x for x in zip(cos_sims, jaccard_sims)]))
        return sparse.csr_matrix(np.matrix([x for x in cos_sims])).reshape(-1, 1)
        
    def predict(self, X):
        if type(self.average) == type(None):
            raise ValueError('Fitting X before predict y')

        if X.shape[1] != 2:
            X = self.transform(X)
            
        X = X[:, 0]
        self.average = self.average[:, 0]
        self.boundary = self.boundary[:, 0]
            
        y = np.ones(X.shape[0])
        for i in range(3):
            y[(X > self.boundary[i]).toarray().reshape(-1)] = i+2
        return y
        
        
class tf_weight_stack(BaseEstimator, TransformerMixin):
    def __init__(self, tfv):
        self.tfv = tfv
    
    def fit(self, query, title):
        return self
        
    def fit_transform(self, query, title):
        self.fit(query, title)
        return self.transform(query, title)
        
    def transform(self, query, title):
        title_tfv = self.tfv.transform(title)
        tf_weight = [np.sum([title_tfv[row, self.tfv.vocabulary_[w]] for w in query[row].split(' ')]) for row in range(len(title))]
        return sparse.csr_matrix(tf_weight).reshape(-1, 1)
        
        
# class sampling_for_piepline(BaseEstimator, TransformerMixin):     
#     def fit(self, X, y):
#         return self
# 
#     # piepline이 fit(train)할때 가장먼저 호출
#     # fit_transform이 없다면 fit()과 transform()을 각각 호출함
#     def fit_transform(self, X, y):
#         X_samp, y_samp = OneSidedSelection().fit_sample(X, y)
#         return X_samp, y_samp
# 
#     # piepline의 fit이 끝나고 test time에서는 transform()을 호출함
#     def transform(self, X):
#         return X


def plot_multiclass_roc_prc(clf, X, y, file_name=None):
    Y_bin = label_binarize(y, classes=[1, 2, 3, 4])
    try:
        y_score = clf.decision_function(X)
    except:
        print('Apply fit to the clf before drawing the curve')
    
    fpr, tpr, roc_auc = {}, {}, {}
    y_dummies = pd.get_dummies(y, drop_first=False).values
    for i in range(1, 5):
        fpr[i], tpr[i], _ = roc_curve(y_dummies[:, i-1], y_score[:, i-1])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    precision, recall, average_precision = {}, {}, {}
    for i in range(1, 5):
        precision[i], recall[i], _ = precision_recall_curve(Y_bin[:, i-1], y_score[:, i-1])
        average_precision[i] = average_precision_score(Y_bin[:, i-1], y_score[:, i-1])

    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_bin.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_bin, y_score, average="micro")
    
    fig, axes = plt.subplots(1,2, figsize =(14, 7))
    colors = ['k', 'r', 'g', 'b', 'teal']
    
    axes[0].plot([0, 1], [0, 1], 'k--', label='area average = %0.2f' % (sum(roc_auc.values())/len(roc_auc.values())))
    for i in range(1, 5):
        axes[0].plot(fpr[i], tpr[i], color=colors[i], label='rating %i area = %0.2f' % (i, roc_auc[i]))
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('FP')
    axes[0].set_ylabel('TP')
    axes[0].set_title('roc curve')
    axes[0].legend(loc="lower right")
    
    axes[1].plot(recall['micro'], precision['micro'], 'k--', label='area average = %0.2f' % average_precision["micro"])
    for i in range(1, 5):
        axes[1].plot(recall[i], precision[i], color=colors[i], label = 'rating %i area = %0.2f' % (i, average_precision[i]))
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('precision recall curve')
    axes[1].legend(loc="lower left")
    
    if file_name!=None:
        plt.savefig(join(dirname(file_name), 'img', basename(file_name)))
    else:
        plt.show()
        














