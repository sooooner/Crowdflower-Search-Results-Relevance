#-*- coding:utf-8 -*-
import re
import string
import numpy as np
import pandas as pd

from scipy import sparse
from bs4 import BeautifulSoup 

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

from os.path import dirname, basename, join

class processer:
    def __init__(self):
        pass
    
    def string_lower(self, x):
        return x.lower()
    
    def remove_pattern(self, x):
        patterns = {'e-mail' : r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',\
                    'URL' : r'(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',\
                    'HTML' : r'<[^>]*>',\
                    'ko' : r'([ㄱ-ㅎㅏ-ㅣ가-힣]+)',\
                    'ch' : '[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]+'}
        for pattern in patterns:
            x = re.sub(pattern=patterns[pattern], repl='', string=x)
        return x
    
    def punct(self, x):
        punct = string.punctuation
        punct_re = re.compile('[{}]'.format(re.escape(punct)))
        x = punct_re.sub(' ', x)
        token_x = []
        for w in x.split(' '):
            w = w.replace(',', '')
            token_x.append(w.strip())
        x = ' '.join(token_x) 
        return x
    
    def remove_sw(self, x):
        sw = ['http','www','img','border','color','style','padding','table','font', \
              'thi','inch','ha','width','height','0','1','2','3','4','5','6','7','8','9']
        # sklearn.feature_extraction
        sw = text.ENGLISH_STOP_WORDS.union(sw)
        # nltk.corpus
        sw = text.ENGLISH_STOP_WORDS.union(set(stopwords.words('english')))
        token_x = []
        for w in x.split(' '):
            if w not in sw:
                token_x.append(w)
        x = ' '.join(token_x)        
        return x
    
    def tokenizer(self, x):
        token_x = word_tokenize(x)
        x = ' '.join(token_x)
        return x
        
    
    def P_stemmer(self, x):
        # PorterStemmer
        token_x = []
        stemmer = PorterStemmer()
        for w in x.split(' '):
            token_x.append(stemmer.stem(w))
        x = ' '.join(token_x)
        return x
    
    def lemmatizer(self, x):
        # WordNetLemmatizer
        lemma = WordNetLemmatizer()
        token_x = []
        for w in x.split(' '):
            token_x.append(lemma.lemmatize(w))
        x = ' '.join(token_x)
        return x
    
    def df_apply_func(self, df, func, first_run=False):    
        if first_run:
            df['query_preprocessed'] = df.apply(lambda x: func(x['query']), axis=1)
            df['product_title_preprocessed'] = df.apply(lambda x: func(x['product_title']), axis=1)
            df['product_description_preprocessed'] = df.apply(lambda x: func(x['product_description']), axis=1)
        else :
            df['query_preprocessed'] = df.apply(lambda x: func(x['query_preprocessed']), axis=1)
            df['product_title_preprocessed'] = df.apply(lambda x: func(x['product_title_preprocessed']), axis=1)
            df['product_description_preprocessed'] = df.apply(lambda x: func(x['product_description_preprocessed']), axis=1)
        return df
        
    def words_freq_sort(df):
        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform(df)
        words_freqs = bow.sum(axis=0) 
        words_freq = [(w, words_freqs[0, i]) for w, i in vectorizer.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq

    def split_array(x):
        x = x.toarray().flatten()
        half = len(x)//2
        half_of_front = x[:half].reshape(-1)
        half_of_end = x[half:].reshape(-1)
        return half_of_front, half_of_end
     
def preprocessing(inputs, output, before_eda=False):
    
    if basename(inputs) == 'eda_train.csv':
        df = pd.concat([pd.read_csv('./data/eda/eda_train_1.csv').fillna(''), pd.read_csv('./data/eda/eda_train_2.csv').fillna(''), \
                        pd.read_csv('./data/eda/eda_train_3.csv').fillna(''), pd.read_csv('./data/eda/eda_train_4.csv').fillna('')])
        df = df.sample(frac=1).reset_index(drop=True)
    else:
        df = pd.read_csv(inputs).fillna('')
        
    process = processer()
    df_apply_func = process.df_apply_func
    
    if not 'eda' in inputs:
        df = df_apply_func(df, process.string_lower, first_run=True)
    else : 
        df = df_apply_func(df, process.string_lower, first_run=False)
    df = df_apply_func(df, process.remove_pattern)
    df = df_apply_func(df, process.punct)
    df = df_apply_func(df, process.tokenizer)
    df = df_apply_func(df, process.remove_sw)
    df = df_apply_func(df, process.lemmatizer)
    if before_eda:
        print('finish processing for ' + inputs)
        output = join(dirname(args.input), 'for_eda_' + basename(args.input))
        df.to_csv(output, index=False)
    else:
        df = df_apply_func(df, process.P_stemmer)
        df = df_apply_func(df, process.remove_sw)
        df.drop_duplicates(keep='first', inplace=True)
        print('finish processing for ' + inputs)
        df.to_csv(output, index=False)

    
if __name__=="__main__":

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str, help="input file of preprocessing data")
    ap.add_argument("--output", required=False, type=str, help="output file of preprocessing data")
    ap.add_argument("--eda", required=False, type=bool, help="preprocessing before eda")
    args = ap.parse_args()

    output = None
    if args.output:
        output = args.output
    else:
        output = join(dirname(args.input), 'preprocessed_' + basename(args.input))
        
    before_eda = False
    if args.eda:
        before_eda = args.eda

    preprocessing(args.input, output, before_eda=before_eda)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    