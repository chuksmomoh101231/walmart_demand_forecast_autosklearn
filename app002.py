#!/usr/bin/env python
# coding: utf-8

# In[11]:


import streamlit as st
import pandas as pd
import numpy as np
import autosklearn
from autosklearn.regression import AutoSklearnRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score,r2_score,accuracy_score
import joblib


# In[12]:


model = joblib.load('final_model.pkl')


file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])


if file_upload is not None:
    data = pd.read_csv(file_upload)
    #data = data.fillna(0)
    #data['cum_moving_average']=data['sales'].expanding().mean()
    #data['exp_weighted_moving_average']=data['sales'].ewm(span=28).mean()
    #data['total_price'] = data['sales'] * data['sell_price']
    #data['Moving_Average']= data['Weekly_Sales'].rolling(window=7, min_periods=1).mean()
    #data = h2o.H2OFrame(data)
    predictions = model.predict(data)
    #predictions = automl.predict(data).as_data_frame()
    predictions = pd.DataFrame(predictions, columns = ['prediction'])
    predictions = data.join(predictions)
    
    st.write(predictions)
    
    
    @st.cache
    
    def convert_df(df):
        return df.to_csv(index = False, header=True).encode('utf-8')
    csv = convert_df(predictions)
    
    st.download_button(label="Download data as CSV",data=csv,
                file_name='demand_forecast.csv',mime='text/csv')
    

        
            
            
            


# In[5]:


# FOR SINGLE PREDICTIONS, TRY ST.WRITE INSTEAD OF ST.SUCCESS


# In[ ]:




