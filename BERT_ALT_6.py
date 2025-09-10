# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:12:42 2023

@author: paaes
"""


import numpy as np
import pandas as pd
import tokenization
import tensorflow as tf
import tensorflow_hub as hub
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import re
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras import utils 
from sklearn.preprocessing import LabelEncoder 
from sklearn.utils import shuffle
import transformers 
import matplotlib.pyplot as plt
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
import os
from tensorflow.keras import models, layers, preprocessing as kprocessing


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

dtf['y'] = dtf['y'].map({'real':1, 'fake':0})
## split dataset


X_train, X_test = model_selection.train_test_split(dtf, test_size=0.3)## get target
y_train = X_train["y"].values
y_test = X_test["y"].values

'''
label = preprocessing.LabelEncoder()
y = label.fit_transform(X_train['y'])
y = to_categorical(y)
print(y[:5])
'''

bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

#m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
#bert_layer = hub.KerasLayer(m_url, trainable=True)

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

#vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
#do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

#outputs = bert_layer(text_input)

l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

model = tf.keras.Model(inputs=[text_input], outputs = [l])
model.summary()

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall')
]

model.compile(optimizer='adam',
 loss='binary_crossentropy',
 metrics=METRICS)


#evaluating the model
import numpy as np
y_predicted = model.predict(X_test)
y_predicted = y_predicted.flatten()
y_predicted = np.where(y_predicted > 0.5, 1, 0)
y_predicted

'''

sample_dataset = [
 'You can win a lot of money, register in the link below,
 'You have an iPhone 10, spin the image below to claim your prize and it will be delivered in your door step',
 'You have an offer, the company will give you 50% off on every item purchased.',
 'Hey Bravin, don't be late for the meeting tomorrow will start lot exactly 10:30 am,
 "See you monday, we have alot to talk about the future of this company ."
]
'''











