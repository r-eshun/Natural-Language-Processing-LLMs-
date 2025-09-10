# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 17:17:33 2023

@author: paaes
"""

import pandas as pd
import numpy as np
import csv

train=[]
colnames=['tweetld','tweetText','userld','imageld(s)','username','timestamp','label']
train = pd.read_csv('C:/DataSets/Tweets/Train Data/tweets.txt', names=colnames, delimiter='\t', header='infer',index_col=['tweetld'],)
print(train.head())

df_fake = train[train['label']=='fake']
print(df_fake.head())
print(df_fake.shape)
df_true = train[train['label']=='real']
print(df_true.head())
print(df_true.shape)

#df_fake["Label"] = "Fake"
#df_true["Label"] = "True"
df=[]
df = pd.concat([df_fake,df_true])
print(df.head())
print(df.shape)
df1 = df.sample(frac=1)

print(df1.iloc[:,0].head())

#check null values
df1.isnull().sum()
df21 = df1

df2 = df21.iloc[:500,:]

df2['label'] = df2['label'].map({'real':1, 'fake':0})
# plot for tweet categories (non-existent here)
import matplotlib.pyplot as plt
import seaborn as sns

#Creating Figure
fig, axes = plt.subplots(1,2, figsize = (15,6))

'''
#Adding the histogram1 - Fake News
sns.histplot(df_fake.subject, palette = 'Set1', alpha = 0.5, ax = axes[0])
axes[0].tick_params(axis = 'x', rotation = 90)
axes[0].set_title('Fake News Subject')


#Adding the histogram2 - True News
sns.histplot(df1.subject, palette = 'Set1', alpha = 0.5, ax = axes[1])
axes[1].tick_params(axis = 'x', rotation = 90)
axes[1].set_title('True  News Subject')

#Printing the count of Subject
print("Fake News Subject : ",dict(df_fake.subject.value_counts()))
print("True News Subject : ",dict(df_true.subject.value_counts()))
'''

sns.distplot(df2.label)

plt.tick_params(axis = 'x', rotation = 90)

plt.title('True VS Fake News')

df1.label.value_counts()


#df["text"] = df["title"]+df["text"] #considering text and title as X

#df1['Label'] = df1['Label'].map({'True':1, 'Fake':0})
#cols = 'tweetld tweetText userld imageld(s) username timestamp'
X = df2.iloc[:,0].values

y = df2['label'].values

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.5, random_state = 10)

from transformers import AutoTokenizer
def tokenize(X):
    
    X = tokenizer(
        text = list(X),
        add_special_tokens = True,
        max_length = 100,
        truncation = True,
        padding = 'max_length',
        return_tensors = 'tf',
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True
        )
    return X

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

X_train_tokens = tokenize(X_train)
X_test_tokens = tokenize(X_test)

print(X_train_tokens)

import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Embedding
from tensorflow.keras.optimizers import Adam
from transformers import TFBertModel

Length = 100
def get_model():

    dropout_rate = 0.2

    input_ids = Input(shape = (Length,), dtype = tf.int32, name = 'input_ids')
    input_mask = Input(shape = (Length,), dtype = tf.int32, name = 'input_mask')
    embeddings = bert([input_ids, input_mask])[1] #pooler output
    print(embeddings)
    out = Dropout(0.2)(embeddings)
    #64 units dense layer
    out = Dense(64,activation = 'relu')(out)
    out = Dropout(0.2)(out)
    y = Dense(1,activation = 'sigmoid')(out)

    model = Model(inputs=[input_ids, input_mask], outputs=y)
    model.layers[2].trainable = True
    #define optimizer
    optimizer = Adam(learning_rate=1e-05, epsilon=1e-08, decay=0.01,clipnorm=1.0)

    #complile the model
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = 'accuracy')
    return model
#load the BERT model

bert = TFBertModel.from_pretrained('bert-base-uncased')
#plot BERT model created
model = get_model()
tf.keras.utils.plot_model(model)

# train the model

from keras.callbacks import EarlyStopping

history = model.fit(x = {'input_ids':X_train_tokens['input_ids'],'input_mask':X_train_tokens['attention_mask']}, y = y_train, epochs=3, validation_split = 0.5, batch_size = 64 )

#callbacks=[EarlyStopping( monitor='val_accuracy' ,mode='max', patience=3,verbose=False,restore_best_weights=True)]
#!pip install transformers

'''
#evaluate the model
yhat = np.where(model.predict({ 'input_ids' : X_test_seq['input_ids'] , 'input_mask' : X_test_seq['attention_mask']}) >=0.5,1,0)

print(classification_report(y_test,yhat))

'''




