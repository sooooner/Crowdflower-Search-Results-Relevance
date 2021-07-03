#-*- coding:utf-8 -*-
import os
import pandas as pd
from utility.predict import Predict
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, type=str, help="mode select:'eda' or 'sampling'")
args = ap.parse_args()

def load_data(mode):
    '''
    mode : string, 'eda' or 'sampling'
    '''
    if mode=='eda':
        train = pd.read_csv('./data/preprocessed_eda_train.csv')
    elif mode=='sampling':
        train = pd.read_csv('./data/preprocessed_train.csv')
    else:
        raise Exception("select only one of two modes: 'eda' and 'sampling'")

    train = train.drop_duplicates(['query_preprocessed', 'product_title_preprocessed'])
    test = pd.read_csv('./data/preprocessed_test.csv')
    return train, test

if __name__=="__main__":
    if not os.path.exists("./submission"):
        os.makedirs("./submission") 

    data = load_data(mode=args.mode)
    submission, submission_proba = Predict(data, mode=args.mode)
    submission.to_csv("./submission/submission.csv", index=False)
    submission_proba.to_csv("./submission/submission_proba.csv", index=False)
    










