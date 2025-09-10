# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:14:26 2023

@author: paaes
"""

## for data
import json
import pandas as pd
import numpy as np## for plotting
import matplotlib.pyplot as plt
import seaborn as sns## for processing
import re
import nltk## for bag-of-words

from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing## for explainer
#from lime import lime_text## for word embedding
#import gensim
#import gensim.downloader as gensim_api## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K## for bert language model
import transformers

colnames=['tweetld','tweetText','userld','imageld(s)','username','timestamp','label']
train = pd.read_csv('C:/DataSets/Tweets/Train Data/tweets.txt', names=colnames, delimiter='\t', header='infer',index_col=['tweetld'],)
print(train.head())

## filter categories
dtf = train[ train["label"].isin(['fake','real']) ][["tweetText","label"]]
## rename columns
dtf = dtf.rename(columns={"label":"y", "tweetText":"text"})
## print 5 random rows
## print 5 random rows
dtf.sample(5)

'''
fig, ax = plt.subplots()
fig.suptitle("y", fontsize=12)
dtf["y"].reset_index().groupby("y").count().sort_values(by= 
       "index").plot(kind="barh", legend=False, 
        ax=ax).grid(axis='x')
plt.show()
'''

'''
Preprocess a string.
:parameter
    :param text: string - name of column containing text
    :param lst_stopwords: list - list of stopwords to remove
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''
def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and   
    #characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text


# function to remove stop words with nltk

lst_stopwords = nltk.corpus.stopwords.words("english")
print(lst_stopwords)


# function to "clean text"
dtf["text_clean"] = dtf["text"].apply(lambda x: 
          utils_preprocess_text(x, flg_stemm=False, flg_lemm=True, 
          lst_stopwords=lst_stopwords))
dtf.head()










