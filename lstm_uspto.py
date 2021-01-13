# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 17:28:12 2021
@author: Vo Duc Thuan
"""

import  sys
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

vocab_size = 3000 # size of vocabulary
embedding_dim = 64
training_portion = .85 # set ratio of train (85%) and validation (15%)
list_of_patents = []
labels = []

# Default parameters
label_para=4
content_para=2
category_para=9
max_length = 20

# those parameters based on uspto.csv
# Col 0,1: Invention no. and Date
# Col 2: Invention title
# Col 3: Main category (A, B, C, D, E, F, G, H)
# Col 4: Sub-categories 
# Col 5, 6, 7: Sub-categories (for further purposes)
# Col 8: Abstract context
# Col 9: Body context
if sys.argv[1]=='cat1':
    label_para=3
    category_para=9
if sys.argv[1]=='cat2':
    label_para=4
    category_para=479
if sys.argv[2]=='inv':
    content_para=2
    max_length = 20
if sys.argv[2]=='abs':
    content_para=8
    max_length = 100
    
# Read data and remove stopword
with open("data/uspto.csv", 'r', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[label_para])
        patent = row[content_para]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            patent = patent.replace(token, ' ')
            patent = patent.replace(' ', ' ')
        list_of_patents.append(patent)
print(len(labels))
print(len(list_of_patents))

train_size = int(len(list_of_patents) * training_portion)
train_patents = list_of_patents[0: train_size]
train_labels = labels[0: train_size]
validation_patents = list_of_patents[train_size:]
validation_labels = labels[train_size:]

oov_tok = '<OOV>'
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_patents)
word_index = tokenizer.word_index

dict(list(word_index.items())[0:20]) ## print out first 20 index of vocabulary

train_sequences = tokenizer.texts_to_sequences(train_patents)
# First of 50 records in token form
for i in range(50):
    print(train_sequences[i])

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
# First of 50 records after padding to size 20
for i in range(50):
    print(train_padded[i])

validation_sequences = tokenizer.texts_to_sequences(validation_patents)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding='post', truncating='post')

# set of lables
print(set(labels))

# label to token
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

# First of 20 labels (token form)
for i in range(50):
    print(training_label_seq[i])

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# Checking encode and original
def decode_patent(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
print('------------------------')
print(decode_patent(train_padded[20]))
print(train_patents[20])
print('------------------------')

# Use tf.keras.layers.Bidirectional(tf.keras.layers.LSTM()).
# Use ReLU in place of tanh function.
# Add a Dense layer with 7 units and softmax activation.
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    tf.keras.layers.Dense(category_para, activation='softmax')
])
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Traing model with 15 epochs
num_epochs = 15
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)

# Predict input text
patent_input = ["Apparatus and method for determining a physiological condition"]
seq = tokenizer.texts_to_sequences(patent_input)
padded = pad_sequences(seq, maxlen=max_length)
prediction = model.predict(padded)
print(prediction)
print(labels[np.argmax(prediction)])