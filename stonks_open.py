# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:56:13 2021

@author: Team Stonks
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
#data preprocess
def process_data(csv_file):
    df=pd.read_csv(csv_file)
    df['Index']=df.index
    df.replace(0,np.nan)
    df.dropna(how='all',axis=0)  
    df=df[['Index','Prev Close','Open','Close','Volume']]
    
    

    X=df.loc[:,['Prev Close']].values
    Y=df.loc[:,['Open']].values
    Z=df.loc[:,['Close']].values
    scaler = MinMaxScaler(feature_range=(0,1))
    X1 = scaler.fit_transform(X)
    Y1=scaler.fit_transform(Y)
    Z1=scaler.fit_transform(Z)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X1, Y1,
                                                        test_size = 0.1)
    X_train= np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test=np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1))

#Model train
    from keras.models import Sequential
    from keras.layers import Dense, LSTM,Activation,Dropout
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer='adam',loss='mse')
    model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=2)


    Value=Z1[-1]
   
    Value=np.reshape(Value,(Value.shape[0], Value.shape[0], 1))
#Predict
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    Y_test=scaler.inverse_transform(y_test)

    v=model.predict(Value)
    value = scaler.inverse_transform(v)
    
    return v,value,predictions



