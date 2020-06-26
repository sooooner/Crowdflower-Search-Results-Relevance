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

class processer:
    def __init__(self):
        pass
    
    def string_lower(self, x):
        return x.lower()
    
    def remove_pattern(self, x):
        patterns = {'e-mail' : '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',\
                    'URL' : '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',\
                    'HTML' : '<[^>]*>',\
                    'ko' : '([ㄱ-ㅎㅏ-ㅣ가-힣]+)',\
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
    
    def df_apply_func(df, func, output_name='_preprocessed', first_run=False):    
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

if __name__=="__main__":
    # load data
    train = pd.read_csv('./data/train.csv').fillna('')
    test = pd.read_csv('./data/test.csv').fillna('')
    process = processer()
    df_apply_func = processer.df_apply_func

    train = df_apply_func(train, process.string_lower, first_run=True)
    train = df_apply_func(train, process.remove_pattern)
    train = df_apply_func(train, process.punct)
    train = df_apply_func(train, process.tokenizer)
    train = df_apply_func(train, process.remove_sw)
    train = df_apply_func(train, process.P_stemmer)
    train = df_apply_func(train, process.lemmatizer)
    train = df_apply_func(train, process.remove_sw)
    
    test = df_apply_func(test, process.string_lower, first_run=True)
    test = df_apply_func(test, process.remove_pattern)
    test = df_apply_func(test, process.punct)
    test = df_apply_func(test, process.tokenizer)
    test = df_apply_func(test, process.remove_sw)
    test = df_apply_func(test, process.P_stemmer)
    test = df_apply_func(test, process.lemmatizer)
    test = df_apply_func(test, process.remove_sw)
    
    #save
    import os
    if not os.path.exists("./data"):
        os.makedirs("./data")
    train.to_csv('./data/preprocessed_train.csv', index=False)
    test.to_csv('./data/preprocessed_test.csv', index=False)