import math
from scipy.optimize import fsolve
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_btc_emitted(block, block_last_month):
    if block_last_month == 0:
        epoch = 1
        btc_per_block = halving_dict[1][2]
        btc_emitted = block_emission_df.loc[epoch, 'btc_per_block']*block
    else:
        x = block_emission_df[block_emission_df.starting_block < block]
        current_epoch = x.index.max()
        y = block_emission_df[block_emission_df.starting_block < block_last_month]
        last_epoch = y.index.max()
        if last_epoch == current_epoch:
            epoch = current_epoch
            btc_emitted = block_emission_df.loc[epoch, 'btc_per_block']*(block-block_last_month)
        else:
            btc_per_block_1 = block_emission_df.loc[last_epoch, 'btc_per_block']
            blocks_1 = block_emission_df.loc[last_epoch, 'ending_block'] - block_last_month
            btc_per_block_2 = block_emission_df.loc[current_epoch, 'btc_per_block']
            blocks_2 = block - block_emission_df.loc[last_epoch, 'ending_block']
            btc_emitted = (btc_per_block_1*blocks_1) + (btc_per_block_2*blocks_2)
    return btc_emitted


st.set_page_config(layout="wide")
price_file = 'monthly_btc_price.csv'
block_file = 'blocks_by_month.csv'
adj_file = 'btc_adj_circ_supply.csv'

file = 'btc_emission.csv'
price_file = 'monthly_btc_price.csv'
block_file = 'blocks_by_month.csv'
adj_file = 'btc_adj_circ_supply.csv'
halving_dict = {1: (0, 210000, 50), 2: (210001, 420000, 25), 3: (420001, 630000, 12.5), 4: (
    630001, 840000, 6.25), 5: (840001, 1050000, 3.125), 6: (1050001, 1260000, 1.5625), 7: (1260001, 1470000, .78125)}
block_emission_df = pd.DataFrame.from_dict(halving_dict, orient='index', columns=[
    'starting_block', 'ending_block', 'btc_per_block'])


# dis_mode = st.sidebar.selectbox('Choose Stock Mode', ['Nominal', 'Adjusted for Lost Coins'])
adj_df = pd.read_csv(adj_file)
adj_df['date'] = adj_df['timestamp'].str[0:10]
adj_df['eom'] = pd.to_datetime(adj_df['date'], format="%Y-%m-%d")+MonthEnd(0)
adj_df['timestamp'] = adj_df['timestamp'].str[0:-2]
adj_df = adj_df[adj_df['timestamp'] == adj_df['eom']]
adj_df = adj_df.rename(columns={'value': 'adj_stock'})
adj_df = adj_df.drop(['date', 'eom'], axis=1)
adj_df = adj_df.rename(columns={'timestamp': 'date'})
adj_df.date = pd.to_datetime(adj_df.date)


block_df = pd.read_csv(block_file)
for i, r in block_df.iterrows():
    if i == 0:
        block_df.loc[i, 'month_flow'] = get_btc_emitted(block_df.loc[i, 'block'], 0)
    else:
        block_df.loc[i, 'month_flow'] = get_btc_emitted(
            block_df.loc[i, 'block'], block_df.loc[i-1, 'block'])
block_df.loc[0, 'stock'] = 0
for i, r in block_df.iterrows():
    if i == 0:
        pass
    else:
        block_df.loc[i, 'stock'] = block_df.loc[i-1, 'month_flow']+block_df.loc[i-1, 'stock']
block_df.date = pd.to_datetime(block_df.date)

price_df = pd.read_csv(price_file)
price_df.date = pd.to_datetime(price_df.date)
price_df.sort_values('date', inplace=True)
price_df.reset_index(inplace=True, drop=True)
# st.write(price_df)
#
block_price_df = pd.merge(price_df, block_df, on='date', how='left')
block_price_df['flow'] = 12*block_price_df['month_flow']
block_price_df['s2f'] = block_price_df['stock']/block_price_df['flow']

switch_df = pd.merge(adj_df, block_price_df, on='date', how='right')
df = switch_df.copy()
st.write(df)

fig = px.line(df, 'date', 'price',
              title="Bitcoin Price")
fig.data[0].line.color = "gray"
# org S2F
org = st.sidebar.radio('Original S2F', ['Yes', 'No'])
if org == 'Yes':
    df['S2F'] = .4*((df['stock']/df['flow'])**3)
    fig.add_trace(go.Scatter(x=df.date.values, y=df.S2F, mode='markers',
                  name='Org_S2F', marker={'color': 'pink'}))
else:
    pass

# S2F-G
G = st.sidebar.radio('S2F-G Model', ['Yes', 'No'])
if G == 'Yes':
    df['S2FG'] = (2.8294*pow(10, -21))*(df['stock']**5.0046/df['flow']**2.0669)
    fig.add_trace(go.Scatter(x=df.date.values, y=df.S2FG, mode='markers',
                  name='S2F-G', marker={'color': 'blue'}))
else:
    pass
# fig.add_trace(go.Scatter(x=df.date.values, y=df.price, mode='markers', name='Obs_price'))

radio = st.sidebar.radio('S2FG-Lost_Coins', ['Yes', 'No'])
if radio == 'No':
    pass
else:
    df['adj_S2FG'] = .00000000000001 * (df['stock']**4.6079)/(df['flow']**2.6608)
    fig.add_trace(go.Scatter(x=df.date.values, y=df.adj_S2FG,
                  mode='markers', name='Adj_calc', marker={'color': 'red'}))

partial = st.sidebar.radio('S2FG-Lost_Coins_Through_2018', ['Yes', 'No'])
if partial == 'No':
    pass
else:
    df['adj_S2FG_P'] = 1.0568*10**-18 * (df['stock']**4.9343)/(df['flow']**2.3872)
    fig.add_trace(go.Scatter(x=df.date.values, y=df.adj_S2FG_P,
                  mode='markers', name='Adj_calc_part', marker={'color': 'green'}))


fig.update_yaxes(title_text='Prices', type='log')
fig.update_xaxes(title_text='Date')


# moving average
have_ma = st.sidebar.radio('Shall we show a moving average?', ['Yes', 'No'])
if have_ma == 'Yes':
    ma = st.sidebar.slider('Choose a value for the moving average',
                           min_value=2, max_value=200, value=12)

    df['ma'] = df['price'].rolling(window=ma).mean()
    fig.add_trace(go.Line(x=df.date.values, y=df.ma, name=f'{ma} month moving average', line=dict(
        color='white',
        width=1)))
else:
    pass
st.plotly_chart(fig, use_container_width=True)

# st.write(
#     f'Ratio of Cum Error between adj_S2FG and S2FG is {df.adj_S2FG_err.sum()/df.S2FG_err.sum()}')

x_values = df[['stock', 'flow']].values
y_values = df['price'].values

# log(price) = log(169873.1612)-0.0087*log(adj_stock)-0.0028*log(flow)
# price = 5.23*adj_stock^-.0087/flow^-.0028
