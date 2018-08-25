#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from operations import df

# df = pd.read_csv('train_operators.csv')
X = df[['f1','f2']]
y = df['label']
y  = pd.get_dummies(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=2,stratify=y)

model = Sequential()
model.add(Dense(10,input_dim=2,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')
# model.fit(np.array(X),np.array(y),nb_epoch=10,validation_data=(X_test,y_test))
model.fit(np.array(X_train),np.array(y_train),nb_epoch=10)

model.save('keras_operators.h5')

# load model
clf = load_model('keras_operators.h5')

# print(clf.summary())
print(clf.predict(np.array(X_test)))