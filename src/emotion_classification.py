import re
import pandas as pd
import tensorflow as tf
import os
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping

#with Long short-term memory (LSTM)-RNN

df=pd.read_csv('data/train_data.csv')
#df.sentiment.value_counts()

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) 
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text

df['content'] = df['content'].apply(clean_text)
df['content'] = df['content'].str.replace('\d+', '') #remove digits


MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100

#to vectorize a text corpus
#num_words: the maximum number of words to keep, based on word frequency. 
#filters: more characters to exclude
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='=!"#$%&*+-.:<=>?\^_`~')
#This method creates the vocabulary index based on word frequency,lower integer means more frequent word
tokenizer.fit_on_texts(df['content'].values)
word_index = tokenizer.word_index 
#print('%s unique tokens' % len(word_index))

#Transform each text in a sequence of integers
X = tokenizer.texts_to_sequences(df['content'].values)
#to produce batches for training/validation
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
#print('Shape of data tensor:', X.shape)
Y = pd.get_dummies(df['sentiment']).values #from categorical to dummy
#print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 100)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(13, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))