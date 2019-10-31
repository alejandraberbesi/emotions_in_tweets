import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer

df = pd.read_csv('data/train_data.csv')
stemming=PorterStemmer()

# cleaning steps:
def filter(text):
    text2 = text.split()
    text2 = [stemming.stem(word) for word in text2 if word.isalpha()]
    text2 = ' '.join(text2)
    return text2

for index, row in df.iterrows():
    df.loc[index,'content']  = filter(df.loc[index,'content'])

tfidfconverter = TfidfVectorizer(max_features=1500,stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(df['content']).toarray()
y=np.array(df['sentiment'])

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,random_state = 10)
clf = RandomForestClassifier(n_estimators=500, random_state=100, class_weight = "balanced")
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
