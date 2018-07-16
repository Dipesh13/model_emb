#!/usr/bin/python
# -*- coding: utf-8 -*-
from flask import Flask, request
from flask_cors import CORS
import requests
import json
import pickle
from pred import prediction

app = Flask(__name__)
CORS(app)

# model_file = './K Nearest Neighbours.pickle'
# with open(model_file, 'rb') as f:
#     model = pickle.load(f)

@app.route('/predict',methods=['POST'])
def predict_tfidf():
    data = json.loads(request.data.decode('utf8'))
    preds = []
    for article in data['articles']:
        preds.append(prediction(article))
    return json.dumps({'label' : preds})

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000, debug = True)