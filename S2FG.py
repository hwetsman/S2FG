import requests
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime


def get_btc_price():
    response = requests.get('https://api.coindesk.com/v1/bpi/currentprice.json')
    data = response.json()
    btc_price = data["bpi"]["USD"]["rate"]
    return btc_price


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


file = 'btc_emission.csv'
price_file = 'monthly_btc_price.csv'
block_file = 'blocks_by_month.csv'
halving_dict = {1: (0, 210000, 50), 2: (210001, 420000, 25), 3: (420001, 630000, 12.5), 4: (
    630001, 840000, 6.25), 5: (840001, 1050000, 3.125), 6: (1050001, 1260000, 1.5625), 7: (1260001, 1470000, .78125)}
block_emission_df = pd.DataFrame.from_dict(halving_dict, orient='index', columns=[
    'starting_block', 'ending_block', 'btc_per_block'])
st.set_page_config(layout="wide")

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
# st.write(block_df)
#
price_df = pd.read_csv(price_file)
price_df.date = pd.to_datetime(price_df.date)
price_df.sort_values('date', inplace=True)
price_df.reset_index(inplace=True, drop=True)
# st.write(price_df)
#
block_price_df = pd.merge(price_df, block_df, on='date', how='left')
block_price_df['flow'] = 12*block_price_df['month_flow']
block_price_df['s2f'] = block_price_df['stock']/block_price_df['flow']
c = 2.8294*(10 ** (-21))
m = 5.0046
n = 2.0669
block_price_df['s2fg_price'] = c*block_price_df['stock']**m/block_price_df['flow']**n
st.write(block_price_df)
#
# today = datetime.datetime.today().date()
# year = str(today.year)
# month = str(today.month)
# eom = pd.to_datetime(year+month, format="%Y%m") + MonthEnd(1)
# #
# #
# # get new price
# if eom.date() == today:
#     if datetime.datetime.now().hour == 23:
#         if datetime.datetime.now().minute == 59:
#             if today not in price_df.date.tolist():
#                 last_index = price_df.index.tolist()[-1]
#                 price_df.loc[last_index+1, 'price'] = get_btc_price()
#                 price_df.loc[last_index+1, 'date'] = today
#                 price_df.to_csv(price_file, index=False)
#             else:
#                 pass
#         else:
#             pass
#     else:
#         pass
# else:
#     pass
# #
# #
# df = pd.read_csv(file)
# df.columns = ['date', 'block', 'epoch', 'subsidy', 'year',
#               'starting', 'added', 'end', 'waste1', 'waste2']
#
# df['year'] = pd.to_datetime(df['year'])
# df = df.drop(['waste1', 'waste2', 'date'], axis=1)
# df.set_index('year', inplace=True, drop=True)
# df.loc[pd.to_datetime('12/31/2028'), 'block'] = 892500
# df.loc[pd.to_datetime('12/31/2028'), 'epoch'] = 5
# df.loc[pd.to_datetime('12/31/2028'), 'subsidy'] = 3.125
# df.loc[pd.to_datetime('12/31/2028'), 'starting'] = df.loc[pd.to_datetime('12/31/27'), 'end']
# df.loc[pd.to_datetime('12/31/2028'), 'added'] = 164062.5
# df.loc[pd.to_datetime('12/31/2028'), 'end'] = df.loc[pd.to_datetime('12/31/28'),
#                                                      'starting'] + df.loc[pd.to_datetime('12/31/2028'), 'added']
# for i in [2029, 2030, 2031, 2032]:
#     df.loc[pd.to_datetime(f'12/31/{i}'), 'epoch'] = 6
#     df.loc[pd.to_datetime(f'12/31/{i}'), 'subsidy'] = 3.125/2
#     df.loc[pd.to_datetime(f'12/31/{i}'), 'block'] = 52500 + \
#         df.loc[pd.to_datetime(f'12/31/{i-1}'), 'block']
#     df.loc[pd.to_datetime(f'12/31/{i}'), 'added'] = 52500 * \
#         df.loc[pd.to_datetime(f'12/31/{i}'), 'subsidy']
#     df.loc[pd.to_datetime(
#         f'12/31/{i}'), 'end'] = df.loc[pd.to_datetime(f'12/31/{i}'), 'added'] + df.loc[pd.to_datetime(f'12/31/{i-1}'), 'end']
#     df.loc[pd.to_datetime(f'12/31/{i}'), 'starting'] = df.loc[pd.to_datetime(f'12/31/{i-1}'), 'end']
# for i in [2033, 2034, 2035, 2036]:
#     df.loc[pd.to_datetime(f'12/31/{i}'), 'epoch'] = 7
#     df.loc[pd.to_datetime(f'12/31/{i}'), 'subsidy'] = 3.125/2/2
#     df.loc[pd.to_datetime(f'12/31/{i}'), 'block'] = 52500 + \
#         df.loc[pd.to_datetime(f'12/31/{i-1}'), 'block']
#     df.loc[pd.to_datetime(f'12/31/{i}'), 'added'] = 52500 * \
#         df.loc[pd.to_datetime(f'12/31/{i}'), 'subsidy']
#     df.loc[pd.to_datetime(
#         f'12/31/{i}'), 'end'] = df.loc[pd.to_datetime(f'12/31/{i}'), 'added'] + df.loc[pd.to_datetime(f'12/31/{i-1}'), 'end']
#     df.loc[pd.to_datetime(f'12/31/{i}'), 'starting'] = df.loc[pd.to_datetime(f'12/31/{i-1}'), 'end']
# for i in [2037, 2038, 2039, 2040]:
#     df.loc[pd.to_datetime(f'12/31/{i}'), 'epoch'] = 8
#     df.loc[pd.to_datetime(f'12/31/{i}'), 'subsidy'] = 3.125/2/2/2
#     df.loc[pd.to_datetime(f'12/31/{i}'), 'block'] = 52500 + \
#         df.loc[pd.to_datetime(f'12/31/{i-1}'), 'block']
#     df.loc[pd.to_datetime(f'12/31/{i}'), 'added'] = 52500 * \
#         df.loc[pd.to_datetime(f'12/31/{i}'), 'subsidy']
#     df.loc[pd.to_datetime(
#         f'12/31/{i}'), 'end'] = df.loc[pd.to_datetime(f'12/31/{i}'), 'added'] + df.loc[pd.to_datetime(f'12/31/{i-1}'), 'end']
#     df.loc[pd.to_datetime(f'12/31/{i}'), 'starting'] = df.loc[pd.to_datetime(f'12/31/{i-1}'), 'end']
# i = '2010'
# df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 0.23
# i = '2011'
# df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 3.06
# i = '2012'
# df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 12.56
# i = '2013'
# df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 946.92
# i = '2014'
# df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 378.64
# i = '2015'
# df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 362.73
# i = '2016'
# df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 753.26
# i = '2017'
# df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 10859.56
# i = '2018'
# df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 4165.61
# i = '2019'
# df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 7420.84
# i = '2020'
# df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 18795.20
# i = '2021'
# df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 57238.62
# i = '2022'
# df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 19780
#
#
# c = 2.8294*(10 ** (-21))
# m = 5.0046
# n = 2.0669
# df['S2F'] = df['end']/df['added']
# df['S2F_Price'] = c*df['end']**m/df['added']**n
# st.write(df)
# df['sd'] = df.S2F_Price.std()
# df.reset_index(drop=False, inplace=True)
# df = df[df.year.dt.date > today]
# st.write(df)
# fig = make_subplots(specs=[[{'secondary_y': False}]])
# fig.add_trace(go.Scatter(x=block_price_df['date'], y=block_price_df['s2fg_price'],
#               name='S2FG', mode='lines'), secondary_y=False)
# fig.add_trace(go.Scatter(x=block_price_df['date'],
#               y=block_price_df.price, name='Price', mode='markers'))
# fig.add_trace(go.Scatter(x=df.year, y=df.S2F_Price,
#                          name='Projected', mode='lines+markers'))
# fig.update_yaxes(title_text="y-axis in logarithmic scale", type="log")
# st.plotly_chart(fig, use_container_width=True)
# # st.plotly_chart(fig)
