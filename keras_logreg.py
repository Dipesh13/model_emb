#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
from get_embedding import sent_embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.utils import to_categorical


df = pd.read_csv('train.csv')

df = df[df['labels'] != 'test_data_samples']
# Pre-Processing : remove punctations.
# digits lower case in sent_embediing.
# stop words left. use nltk for that
# decode left
df['data']= df['data'].apply(lambda x: x.translate(None, string.punctuation))
df['data']= df['data'].apply(lambda x: x.translate(None, string.digits))
# df['data']= df['data'].apply(lambda x: x.decode('utf-8'))
sentences = df['data'].tolist()
y = df['labels'].tolist()
# target = to_categorical(y)
y  = pd.get_dummies(y)

X = []
for sentence in sentences:
    sent_emb = sent_embedding(sentence)
    X.append(sent_emb)

# for a,b in zip(X,y):
#     print(len(a),b)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=2,stratify=y)

# lr = LogisticRegression()
# knn = KNN()
# dt = DecisionTreeClassifier()
#
# classifiers = [('Logistic Regression', lr),('K Nearest Neighbours', knn),('Classification Tree', dt)]
#
# for clf_name, clf in classifiers:
#     clf.fit(X_train, y_train)
#     preds = clf.predict(X_train)
#     print(clf_name + " train accuracy: ", accuracy_score(y_train, preds))
#     preds_test = clf.predict(X_test)
#     print(clf_name + " test accuracy: ", accuracy_score(y_test, preds_test))
#
#     with open(clf_name+'.pickle', 'wb') as fo:
#         pickle.dump(clf,fo)

model = Sequential()
# model.add(Dense(1,input_dim=50,activation='relu'))
model.add(Dense(1,input_dim=50,activation='sigmoid'))
model.add(Dense(3,activation='softmax'))

model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')
# model.fit(np.array(X_train),np.array(y_train))
model.fit(np.array(X_train),np.array(y_train),epochs=100)

# model.save('keras1.h5')
model.save('keras_logreg.h5')

# load model
clf = load_model('keras_logreg.h5')

# print(clf.summary())
print(clf.predict(np.array(X_test)))