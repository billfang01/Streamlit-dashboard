# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:12:20 2024

@author: python2
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import requests
import plotly.subplots as sp
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from dotenv import dotenv_values
#import creds

today = datetime.date.today()

###

yf = yf.download("^GSPC GC=F HG=F ^VIX ^TWII", start="1980-01-01", end=today)
yf = yf['Close'].copy()
yf.columns = ['Gold', 'Copper', 'S&P500','加權指數','VIX恐慌指數']

yf['金銅比'] = yf['Gold'] / yf['Copper']

df = yf.copy()


def get_data(series_id):
    fred_api_key = os.environ.get('FRED_API_KEY')
    if fred_api_key is None:
        raise ValueError("FRED API key not found in environment variables")

    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={fred_api_key}&file_type=json"
    
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['observations']).replace('.', np.nan).dropna()
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = df['value'].astype(float)
    df = df.set_index('date')
    df[series_id] = df['value']
    return df[[series_id]]



indicators_df = pd.DataFrame(index=pd.date_range('1980-01-02', 'today', freq='MS'))


indicators_df['yield_curve'] = get_data('T10Y2YM')



indicators_df['CPI'] = get_data('CPIAUCSL')

def calculate_percent_change_from_year_ago(data):
    
    percent_change = data.pct_change(periods=12) * 100
    return percent_change


percent_change_from_year_ago = calculate_percent_change_from_year_ago(indicators_df['CPI'])


st.set_option('deprecation.showPyplotGlobalUse', False)


indicators_df_resampled = indicators_df.resample('D').ffill()  


df = pd.merge(df, indicators_df_resampled, left_index=True, right_index=True, how='left')

percent_change_from_year_ago_resampled = percent_change_from_year_ago.resample('D').ffill()

df = pd.merge(df, percent_change_from_year_ago_resampled, left_index=True, right_index=True, how='left')

del df['CPI_x']

df.columns = ['Gold', 'Copper', 'S&P500','加權指數','VIX恐慌指數','金銅比','殖利率倒掛','CPI年增率']

df  = round(df, 2)

###

st.title("量化地平線")


selected_products_left = st.sidebar.selectbox('顯示左軸產品數據', df_data.columns, key='left')
selected_products_right = st.sidebar.selectbox('顯示右軸產品數據', df_data.columns, key='right')


chart_type_left = st.sidebar.selectbox('選擇類型(左)', ['折線圖', '柱狀圖'], key='left_chart_type')
chart_type_right = st.sidebar.selectbox('選擇類型(右)', ['折線圖', '柱狀圖'], key='right_chart_type')


latest_data_left = df_data[selected_products_left].iloc[-1]
latest_data_right = df_data[selected_products_right].iloc[-1]


start_column, end_column = st.sidebar.columns(2)


start_date = start_column.date_input('開始日期', min_value=datetime.date(1980, 1, 2), max_value=datetime.date.today())


end_date = end_column.date_input('結束日期', min_value=datetime.date(1980, 1, 2), max_value=datetime.date.today(), value=datetime.date.today())


filtered_df = df_data.loc[start_date:end_date]




fig = make_subplots(specs=[[{"secondary_y": True}]])


if chart_type_right == '折線圖':
    fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df[selected_products_right], mode='lines', name=selected_products_right), secondary_y=True)
else:
    fig.add_trace(go.Bar(x=filtered_df.index, y=filtered_df[selected_products_right], name=selected_products_right, opacity=0.3), secondary_y=True)

fig.add_trace(go.Scatter(x=filtered_df.index, y=filtered_df[selected_products_left], mode='lines', name=selected_products_left, line=dict(color='red')), secondary_y=False)


fig.update_layout(title='走勢圖',
                  xaxis_title='日期',
                  yaxis_title='左軸產品數值' if selected_products_left else '',
                  yaxis2_title='右軸產品數值' if selected_products_right else '',
                  width=1000, height=400)

st.plotly_chart(fig)














