# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:47:55 2021

@author: Team Stonks
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from stonks_open import *
from pathlib import PurePath
import os
#data preprocess
def Close_predict(csv_file):
    df=pd.read_csv(csv_file)
    df['Index']=df.index
    df.replace(0,np.nan)
    df.dropna(how='all',axis=0)
    df=df[['Index','Open','Close','Volume']]
    
    

    X=df.loc[:,['Open']].values
    Y=df.loc[:,['Close']].values

    scaler = MinMaxScaler(feature_range=(0,1))
    X1 = scaler.fit_transform(X)
    Y1=scaler.fit_transform(Y)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X1, Y1,
                                                        test_size = 0.1)
    X_train= np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test=np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1))

    print('Stonks_open running')
    value,S_open,graph=process_data(csv_file)    
    
#Model train
    print('Stonks_Close running')
    from keras.models import Sequential
    from keras.layers import Dense, LSTM,Activation,Dropout

    model = Sequential()
    model.add(LSTM(units=40, return_sequences=True, input_shape=(X_train.shape[1],1)))
    model.add(LSTM(units=40))
    model.add(Dense(1))

    model.compile(optimizer='adam',loss='mse')
    model.fit(X_train, y_train, batch_size=1,epochs=1 ,verbose=2)



#Predict
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    Y_test=scaler.inverse_transform(y_test)
    value=np.reshape(value,(value.shape[0], value.shape[0], 1))
    v=model.predict(value)
    
    S_close=scaler.inverse_transform(v)
    return S_open[0],S_close[0]
    

