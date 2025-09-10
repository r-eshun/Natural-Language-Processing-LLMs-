# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:12:42 2023

@author: paaes
"""


import numpy as np
import pandas as pd
import tokenization.FullTokenizer as tfff
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


## split dataset
train

train_data, test_data= model_selection.train_test_split(dtf, test_size=0.3)## get target
y_train = train_data["y"].values
y_test = test_data["y"].values


label = preprocessing.LabelEncoder()
y = label.fit_transform(train_data['y'])
y = to_categorical(y)
print(y[:5])

m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(m_url, trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tfff(vocab_file, do_lower_case)

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
        
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len-len(input_sequence)
        
        
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
        
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def build_model(bert_layer, max_len=512):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    
    clf_output = sequence_output[:, 0, :]
    
    lay = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    lay = tf.keras.layers.Dense(32, activation='relu')(lay)
    lay = tf.keras.layers.Dropout(0.2)(lay)
    out = tf.keras.layers.Dense(5, activation='softmax')(lay)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

#Here We check only the first 250 characters of each text, and also we set train-test input and train labels
max_len = 250
train_input = bert_encode(train_data.OriginalTweet.values, tokenizer, max_len=max_len)
test_input = bert_encode(test_data.OriginalTweet.values, tokenizer, max_len=max_len)
train_labels = y

model = build_model(bert_layer, max_len=max_len)
model.summary()

# Run model
checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

train_sh = model.fit(
    train_input, train_labels,
    validation_split=0.2,
    epochs=3,
    callbacks=[checkpoint, earlystopping],
    batch_size=32,
    verbose=1
)




























