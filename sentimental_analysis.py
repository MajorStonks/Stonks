# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:53:03 2022

@author: user
"""

import requests
import flair
import regex as re
import pandas as pd


def get_data(tweet):
    data = {
        'id': tweet['id_str'],
        'created_at': tweet['created_at'],
        'text': tweet['full_text']
    }
    return data



def Senti_analyze(company):
    params = {
    'q': company,
    'tweet_mode': 'extended',
    'lang': 'en',
    'count':'10000'
}


    BEARER_TOKEN='AAAAAAAAAAAAAAAAAAAAAH3BbAEAAAAANhrInokh03CpxLwZNoSP1qfpQSc%3Dn4eeAxqx51TqnbPu2w60mJ0B51tswZH9l6kX7Nn4phyO8DoLrp'


    response=requests.get(
        'https://api.twitter.com/1.1/search/tweets.json',
        params=params,
        headers={'authorization': 'Bearer '+BEARER_TOKEN}
        )



    df = pd.DataFrame()
    for tweet in response.json()['statuses']:
        row = get_data(tweet)
        df = df.append(row, ignore_index=True)
    
    sentiment_model = flair.models.TextClassifier.load('en-sentiment')
    tweets=df['text']

    probs = []
    sentiments = []

# use regex expressions (in clean function) to clean tweets

    for tweet in tweets.to_list():
        # make prediction
        sentence = flair.data.Sentence(tweet)
        sentiment_model.predict(sentence)
        # extract sentiment prediction
        probs.append(sentence.labels[0].score)  # numerical score 0-1
        sentiments.append(sentence.labels[0].value)  # 'POSITIVE' or 'NEGATIVE'

# add probability and sentiment predictions to tweets dataframe
    tweets['probability'] = probs
    tweets['sentiment'] = sentiments


    senti=pd.DataFrame()
    senti['date']=df['created_at']

    senti['sentiment']=sentiments[0:]

    senti['Probability']=probs[0:]
    
    return senti





