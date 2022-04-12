# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 18:13:41 2021

@author: Team Stonks
"""

import streamlit as st
from Stonks_Complete import *
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM,Activation,Dropout
import base64
from sentimental_analysis import *

def lineplot(df,value,title):   
    fig = px.line(        
        df, #Data Frame
        x = df['Date'], #Columns from the data frame
        y = df[value],
        title=title
           
    )
    fig.update_layout(
            width=1400,
            height=650,
            paper_bgcolor="white",)
    fig.update_traces(line_color = "Coral")
    st.plotly_chart(fig)
   
def sentiplot(df,title):
   fig = px.histogram(       
        df, #Data Frame
        x = df['date'], #Columns from the data frame
        y = df['Probability'],
        color=df['sentiment'],
        title=title
           
    )
   fig.update_layout(
            width=1400,
            height=650,
            paper_bgcolor="white",)
   
   fig.update_layout(barmode='group')
   st.plotly_chart(fig)
    
def main():       
    # front end elements of the web page
    st.set_page_config('STONKS','LOGO_stonks.png','wide')
    main_bg = "stonk.jpg"
    main_bg_ext = "jpg"


    st.markdown(
    f"""
   <style>
	.reportview-container{{
		background-image:url(./stonk.jpg)
	}}
	}}
    </style>
    """,
    unsafe_allow_html=True
    )
       
    st.image('stonks_logo.png',width=250)
  
    menu = ['CIPLA',                                                           #companies names for prediction
'GAIL','BPCL','COALINDIA','HCLTECH','ITC','ONGC',
'TATAMOTORS','HINDUNILVR','HDFCBANK','CENTRALBK','ADANIPORTS','ADANIPOWER','BANKBARODA','ABBOTINDIA',
'WIPRO','PERSISTENT','UCOBANK','AXISBANK','KOTAKBANK','SBIN','NTPC','HTC','BANKINDIA'
]
    menu.sort()
    choice = st.sidebar.selectbox("List of Companies",menu)
    
    if st.sidebar.button("Predict"):
       
        Open,Close=Close_predict(choice+'.csv')
        
        st.write("The Open Price of Stock is:",Open[0])
        st.write("The Close Price Of Stock is:",Close[0])
        
        df=pd.read_csv(choice+'.csv')
        df.replace(0,np.nan)
        df.dropna(how='all',axis=0)
        df=df[['Date','Open','Close','Volume',]]        
        st.dataframe(df.tail(10))
        st.write(" ")
        st.write("Open graph of ",choice,':')
        open_title="Open graph of "+choice+':'
        lineplot(df,'Open',open_title)
        st.write(" ")
        st.write("Close graph of ",choice,':')
        close_title='Close Graph of '+choice+':'
        
        lineplot(df,'Close',close_title)
        
        st.write("Sentimental analysis of ",choice,':')
        analysis_title='sentimental analysis of '+choice+':'
        
        DF=Senti_analyze(choice)
        
        sentiplot(DF, analysis_title)
        
        
 
if __name__=='__main__': 
    main()