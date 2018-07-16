#!/usr/bin/python
# -*- coding: utf-8 -*-
import pickle
from get_embedding import sent_embedding

with open('K Nearest Neighbours.pickle', 'rb') as fi:
    model = pickle.load(fi)

def prediction(data):
    for sentence in data:
        sent_emb = sent_embedding(sentence)
        label = model.predict(sent_emb.reshape(1, -1))
        return label[0]

# data = """Match of the tournament Russia v Croatia had it all: a spectacular goal from Denis Cheryshev, unbearable tension during a see-sawing extra-time period, raucous celebrations when Mário Fernandes equalised and then shootout heartache for the hosts – followed quickly by an acknowledgment that they should hold their heads high.
#
# Player of the tournament Luka Modric. He was influential from the first game but by extra time of the Croatia v England game it was impossible to take your eyes off him. Modric has played more fluently but the way he dragged himself and his team through against the odds was breathtaking.
#
# Goal of the tournament It seems a long time ago now but Ricardo Quaresma’s trademark “trivela” for Portugal against Iran was a luscious piece of skill that bears watching time and again.
#
# Personal highlight It sounds a bit mawkish but watching the sun rise over Kazan Arena from my apartment, only a few hours after seeing France beat Argentina 4-3 there in a stunning game of football, gave rise to a feeling of immense gladness simply to be here. Nobody would take covering this tournament for granted.
#
# Biggest disappointment That we did not see a more diverse tournament from the quarter-finals onwards. Europe’s primacy was disconcerting and, while we may have to wait to see if it is a trend, not entirely surprising. It was a regret off the pitch, too, because the vibrancy that tens of thousands of Latin American fans brought to the early stages was one of the aspects that made this month special."""
#
# pr = prediction(data.decode('utf-8'))
# print(pr)