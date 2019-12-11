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
import matplotlib.pylab as plt

df=pd.read_csv('data/train_data.csv')
#df.sentiment.value_counts() #unbalanced data, 13 categories

STOPWORDS = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
    return text

df['content'] = df['content'].apply(clean_text)
df['content'] = df['content'].str.replace('\d+', '') #remove digits

#convert each word into an index
tokenizer = Tokenizer(filters='\n\t!"#$%&()*+,-./:;<=>?[\]^_`{|}~ ')
tokenizer.fit_on_texts(df['content'].values)
word_index = tokenizer.word_index 

#Transform each tweet in a sequence of integers/indexes of words
a = tokenizer.texts_to_sequences(df['content'].values)
b=list()
for i in a:
    b.append(len(i)) 
MAX_SEQUENCE_LENGTH = max(b) #largest length of filtered tweets
#arrays with zeros and/or words indexes:
X = pad_sequences(a, maxlen=MAX_SEQUENCE_LENGTH)
#emotions categories classified with arrays of zeros and ones
Y = pd.get_dummies(df['sentiment']).values

#test data= 20% of total data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 100)

#size of the vocabulary, maximum integer index + 1 : 
INPUT_DIM=len(word_index)
#output_dim for embedding layer:
EMBEDDING_DIM = 100 ######

model = Sequential()
#input_length=MAX_SEQUENCE_LENGTH
model.add(Embedding(INPUT_DIM, EMBEDDING_DIM, input_length=X.shape[1]))
#how many units do i turn off:
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(13, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 3
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1)

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
