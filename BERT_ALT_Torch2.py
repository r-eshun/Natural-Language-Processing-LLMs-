# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 19:01:09 2023

@author: paaes
"""

import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import pandas as pd


from transformers import BertForSequenceClassification

colnames=['tweetld','tweetText','userld','imageld(s)','username','timestamp','label']
train = pd.read_csv('C:/DataSets/Tweets/Train Data/tweets.txt', names=colnames, delimiter='\t', header='infer',index_col=['tweetld'],)
print(train.head())

## filter categories
df = train[ train["label"].isin(['fake','real']) ][["tweetText","label"]]
## rename columns
df = df.rename(columns={"label":"y", "tweetText":"text"})
## print 5 random rows
## print 5 random rows
df.sample(5)

df['y'] = df['y'].map({'real':1, 'fake':0})
## split dataset

#encode labels
#possible_labels = df.Conference.unique()
'''
label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index
label_dict
'''

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df.index.values, 
                                                  df.y.values, 
                                                  test_size=0.15, 
                                                  random_state=42, 
                                                  stratify=df.y.values)

df['data_type'] = ['not_set']*df.shape[0]

df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'

df.groupby(['text', 'y', 'data_type']).count()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)
                                          
encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].text.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].text.values, 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)


input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].y.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].y.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(df),
                                                      output_attentions=False,
                                                      output_hidden_states=False)










