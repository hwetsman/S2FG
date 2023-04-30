import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def Make_Log(base, df, col):
    new_col = np.log(df[col]) / np.log(base)
    return new_col


def Limit_df(df, start_date):
    df['date'] = pd.to_datetime(df['date'])
    df = df[df.date > pd.to_datetime(start_date)]
    df['date'] = df.date.astype(int)//(10**9)
    st.write(df)
    _, constant = df.tail(1)['date'].to_dict().popitem()
    st.write(constant)
    return df, constant


base = st.sidebar.selectbox(
    'pick a base', [2, 2.71828, 3, 3.1416, 4, 5, 6, 7, 8, 9, 10, 20], index=0)
price_file = 'monthly_btc_price.csv'
orig = pd.read_csv(price_file)
orig['price'] = orig['price'].astype(float)
start_date = '1-1-2014'
orig, constant = Limit_df(orig, start_date)
orig['date'] = orig.date - constant
st.write(orig)
year = 31536000
ticks = [constant+year*x for x in [0, 1, 2, 3, 4, 5, 6, 7, 8]]
# st.write(orig)
fig = go.Figure()
# for base in [2, 2.71828, 3, 3.1416, 4, 5, 6, 7, 8, 9, 10, 20]:
# for base in [2]:
orig[f'Price_Base_{base}'] = Make_Log(base, orig, 'price')
# st.write(orig)
fig.add_trace(go.Scatter(x=orig['date'],
              y=orig[f'Price_Base_{base}'], mode='lines', hovertext=base))
# fig = px.line(orig, x='date', y=f'Price_Base_{base}', title=f'Price_Base_{base}')
# fig.update_yaxes(title_text=f'Price_Base_{base}', type="log")
# fig.update_xaxes(type='log')
# pd.to_datetime(52, 444, 800, source='unix')
fig.update_xaxes(tickvals=ticks)  # , ticklabels=[
# '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022'])
# fig.add_vline(x=constant)
# fig.add_vline(x=constant+year)
# for x in [1+year*y for y in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]:
#     fig.add_vline(x=x)
st.plotly_chart(fig)
