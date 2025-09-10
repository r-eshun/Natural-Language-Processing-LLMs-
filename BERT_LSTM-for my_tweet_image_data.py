# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 16:16:24 2023

@author: paaes
"""

# Import all necessary libraries
'''
try:
    %tensorflow_version 2.x
except Exception:
    pass
'''
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras import utils
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import bert
import os
import numpy as np
import re
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing




colnames=['tweetld','tweetText','userld','imageld(s)','username','timestamp','label']
data = pd.read_csv('C:/DataSets/Tweets/Train Data/tweets.txt', names=colnames, delimiter='\t', header='infer',index_col=['tweetld'],)
print(data.head())
print(data.shape)




## filter categories
data1 = data[ data["label"].isin(['fake','real']) ][["tweetText","label"]]

print(data1.shape)


## rename columns
df = data1.rename(columns={"tweetText":"text"})


print(df.head())
print(df.shape)
df = df.sample(frac=1)

print(df.sample(5))
#print(df.head())

train, test= model_selection.train_test_split(df, test_size=0.3)## get target




# Sort values by 'image_path'

# Check the shapes
print("train samples:",train.shape[0])
print("test samples:",test.shape[0])

# Cleaning text function
def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    sentence = sentence.lower()

    return sentence

def remove_tags(text):
    return TAG_RE.sub('', text)

TAG_RE = re.compile(r'<[^>]+>')
vec_preprocess_text = np.vectorize(preprocess_text)

# Check number of classes
nClasses = train.label.nunique()

encoder = LabelEncoder()
processed_train = vec_preprocess_text(train.text.values)
processed_test = vec_preprocess_text(test.text.values)


encoded_labels_train = encoder.fit_transform(train.label.values)
labels_train = utils.to_categorical(encoded_labels_train, nClasses)

encoded_labels_test = encoder.fit_transform(test.label.values)
labels_test = utils.to_categorical(encoded_labels_test, nClasses)

print("Processed text sample:", processed_train[0])
print("Shape of train labels:", labels_train.shape)

# Import the BERT BASE model from Tensorflow HUB (layer, vocab_file and tokenizer)
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)

# Preprocessing of texts according to BERT
def get_masks(text, max_length):
    """Mask for padding"""
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    length = len(tokens)
    if length > max_length:
        tokens = tokens[:max_length]

    return np.asarray([1]*len(tokens) + [0] * (max_length - len(tokens)))
vec_get_masks = np.vectorize(get_masks, signature = '(),()->(n)')

def get_segments(text, max_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    length = len(tokens)
    if length > max_length:
        tokens = tokens[:max_length]
    
    segments = []
    current_segment_id = 0
    with_tags = ["[CLS]"] + tokens + ["[SEP]"]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return np.asarray(segments + [0] * (max_length - len(tokens)))
vec_get_segments = np.vectorize(get_segments, signature = '(),()->(n)')

def get_ids(text, tokenizer, max_length):
    """Token ids from Tokenizer vocab"""
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    length = len(tokens)
    if length > max_length:
        tokens = tokens[:max_length]

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = np.asarray(token_ids + [0] * (max_length-length))
    return input_ids
vec_get_ids = np.vectorize(get_ids, signature = '(),(),()->(n)')


def prepare(text_array, tokenizer, max_length = 128):
    
    ids = vec_get_ids(text_array, 
                      tokenizer, 
                      max_length).squeeze()
    masks = vec_get_masks(text_array,
                      max_length).squeeze()
    segments = vec_get_segments(text_array,
                      max_length).squeeze()

    return ids, segments, masks

max_length = 40 # that must be set according to your dataset
ids_train, segments_train, masks_train = prepare(processed_train,
                                                 tokenizer,
                                                 max_length)
ids_test, segments_test, masks_test = prepare(processed_test, 
                                               tokenizer,
                                               max_length)



# Classification Model
input_word_ids = layers.Input(shape=(max_length,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = layers.Input(shape=(max_length,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = layers.Input(shape=(max_length,), dtype=tf.int32,
                                    name="segment_ids")
den_out, seq_out = bert_layer([input_word_ids, input_mask, segment_ids])

X = layers.LSTM(128)(seq_out)
X = layers.Dropout(0.5)(X)
X = layers.Dense(256, activation="relu")(X)
X = layers.Dropout(0.5)(X)
output = layers.Dense(nClasses, activation = 'softmax')(X)

model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[output])

# Adam optimizer
opt = optimizers.Adam(lr=.01)

# Compile model
model.compile(loss = 'categorical_crossentropy',
              optimizer = opt,
              metrics = ['accuracy'])

model.summary()

es = callbacks.EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)

# Setup callbacks, logs and early stopping condition
checkpoint_path = "BERT_LSTM/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
cp = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy',save_best_only=True,verbose=1, mode='max')
csv_logger = callbacks.CSVLogger('BERT_LSTM/BERT_LSTM.log')
es = callbacks.EarlyStopping(patience = 3, restore_best_weights=True)

# Reduce learning rate if no improvement is observed
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', factor=0.1, patience=1, min_lr=0.00001)


history = model.fit([ids_train, masks_train, segments_train], 
          labels_train,
          epochs = 16,
          batch_size = 512,
          validation_split = 0.3,
          #callbacks = [csv_logger, reduce_lr])
          callbacks = [reduce_lr])

# Load the log file
#df = pd.read_csv('BERT_LSTM/BERT_LSTM.log')
'''
# Visualize Training and Test accuracy
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['epoch'], y=df['accuracy'],
                    mode='lines',
                    name='training'))

fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_accuracy'],
                    mode='lines',
                    name='test'))

fig.update_layout(
    font_size = 20,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
)

# Visualize Training and Test loss
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['epoch'], y=df['loss'],
                    mode='lines',
                    name='training'))

fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_loss'],
                    mode='lines',
                    name='test'))

fig.update_layout(
    font_size = 20,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
)

fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='Gray')
fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='Gray')

fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='Gray')
fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='Gray')

# Evaluate model performance
'''
model.evaluate([ids_test, masks_test, segments_test],
               labels_test, 
               batch_size = 512)
     

