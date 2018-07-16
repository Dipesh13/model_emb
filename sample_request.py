#!/usr/bin/python
# -*- coding: utf-8 -*-
import requests
import json

dict_data = {"articles":["""Novak Djokovic seals fourth Wimbledon title in final stroll over Kevin Anderson. It took Novak Djokovic five minutes to break Kevin Anderson’s serve, but he could not break his admirable spirit over two hours and 19 minutes in a Wimbledon final memorable only for the South African’s dogged but doomed fightback.

Anderson, troubled early in the match by a sore right elbow, was forced to endure one of the most gruelling afternoons of his career but Djokovic suffered too, swearing at the crowd as his frustrations consumed him before he secured his 13th grand slam title, his fourth here, winning 6-2, 6-2, 7-6 (3).

The sun-bathed Centre Court audience can hardly have imagined when they bought their tickets that they would not be seeing either or both of Roger Federer and Rafael Nadal contest the final. Anderson beat Federer in an excellent quarter-final, Djokovic saw off Nadal over two days in the second semi-final.Shattered after surviving six hours and 35 minutes against John Isner in the first semi-final on Friday, Anderson – the 2017 US Open runner-up – refused to surrender in his second major final and dug deep to make a fight of it in the third set, although it was still a poor spectacle. After last year’s final, when Marin Cilic’s blistered feet did for him against Federer, the closing Sunday needed something special, and this was not it.

The longest rally of the match lasted 15 shots, as Anderson strove to hold serve at 0-2 in the second set. There were the usual sympathetic cheers when he managed it, but pointlessness and inevitability hung heavily in the suffocating air.

Of the 950 points he had served for in the championships, Anderson, a serving behemoth and decent athlete, chose to remain on the baseline for 920 of them. That is either unshakable faith in his ability to hit opponents off the court, tactical ineptitude, exhaustion, or a combination of all three. Not once in the first hour of this match did he come in behind his serve. Neither did Djokovic – but he had no need to; he won through with patience rather than inspiration.

After an hour and 10 minutes, Anderson had the crowd cheering him for all the right reasons when he forced his first break point of the final, but he could not convert and two minutes later the second set was gone, too."""]}


res = requests.post('http://0.0.0.0:5000/predict', data=json.dumps(dict_data))
# print(res.json())
preds = res.json()['label']
print(preds)