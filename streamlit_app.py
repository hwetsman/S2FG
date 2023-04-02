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


def get_btc_height():
    url = 'https://api.blockcypher.com/v1/btc/main'
    response = requests.get(url)
    data = response.json()
    height = data["height"]
    return height


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
# st.write(block_df)
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
# st.write(block_price_df)
#
today = datetime.datetime.today().date()
year = str(today.year)
month = str(today.month)
eom = pd.to_datetime(year+month, format="%Y%m") + MonthEnd(1)
# #
# #
# get new price
if eom.date() == today:
    if datetime.datetime.now().hour == 23:
        if datetime.datetime.now().minute == 59:
            if today not in price_df.date.tolist():
                last_index = price_df.index.tolist()[-1]
                price_df.loc[last_index+1, 'price'] = get_btc_price()
                price_df.loc[last_index+1, 'date'] = today
                price_df.to_csv(price_file, index=False)
            else:
                pass
        else:
            pass
    else:
        pass
else:
    pass
# #

# get block height
if eom.date() == today:
    if datetime.datetime.now().hour == 23:
        if datetime.datetime.now().minute == 59:
            if today not in block_df.date.tolist():
                last_index = block_df.index.tolist()[-1]
                block_df.loc[last_index+1, 'block'] = get_btc_height()
                block_df.loc[last_index+1, 'date'] = today
                block_df.to_csv(block_file, index=False)
            else:
                pass
        else:
            pass
    else:
        pass
else:
    pass

# #
projected_df = pd.read_csv(file)
projected_df.columns = ['date', 'block', 'epoch', 'subsidy', 'year',
                        'starting', 'added', 'end', 'waste1', 'waste2']

projected_df['year'] = pd.to_datetime(projected_df['year'])
projected_df = projected_df.drop(['waste1', 'waste2', 'date'], axis=1)
projected_df.set_index('year', inplace=True, drop=True)
projected_df.loc[pd.to_datetime('12/31/2028'), 'block'] = 892500
projected_df.loc[pd.to_datetime('12/31/2028'), 'epoch'] = 5
projected_df.loc[pd.to_datetime('12/31/2028'), 'subsidy'] = 3.125
projected_df.loc[pd.to_datetime('12/31/2028'),
                 'starting'] = projected_df.loc[pd.to_datetime('12/31/27'), 'end']
projected_df.loc[pd.to_datetime('12/31/2028'), 'added'] = 164062.5
projected_df.loc[pd.to_datetime('12/31/2028'), 'end'] = projected_df.loc[pd.to_datetime('12/31/28'),
                                                                         'starting'] + projected_df.loc[pd.to_datetime('12/31/2028'), 'added']
for i in [2029, 2030, 2031, 2032]:
    projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'epoch'] = 6
    projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'subsidy'] = 3.125/2
    projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'block'] = 52500 + \
        projected_df.loc[pd.to_datetime(f'12/31/{i-1}'), 'block']
    projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'added'] = 52500 * \
        projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'subsidy']
    projected_df.loc[pd.to_datetime(
        f'12/31/{i}'), 'end'] = projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'added'] + projected_df.loc[pd.to_datetime(f'12/31/{i-1}'), 'end']
    projected_df.loc[pd.to_datetime(
        f'12/31/{i}'), 'starting'] = projected_df.loc[pd.to_datetime(f'12/31/{i-1}'), 'end']
for i in [2033, 2034, 2035, 2036]:
    projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'epoch'] = 7
    projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'subsidy'] = 3.125/2/2
    projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'block'] = 52500 + \
        projected_df.loc[pd.to_datetime(f'12/31/{i-1}'), 'block']
    projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'added'] = 52500 * \
        projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'subsidy']
    projected_df.loc[pd.to_datetime(
        f'12/31/{i}'), 'end'] = projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'added'] + projected_df.loc[pd.to_datetime(f'12/31/{i-1}'), 'end']
    projected_df.loc[pd.to_datetime(
        f'12/31/{i}'), 'starting'] = projected_df.loc[pd.to_datetime(f'12/31/{i-1}'), 'end']
for i in [2037, 2038, 2039, 2040]:
    projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'epoch'] = 8
    projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'subsidy'] = 3.125/2/2/2
    projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'block'] = 52500 + \
        projected_df.loc[pd.to_datetime(f'12/31/{i-1}'), 'block']
    projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'added'] = 52500 * \
        projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'subsidy']
    projected_df.loc[pd.to_datetime(
        f'12/31/{i}'), 'end'] = projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'added'] + projected_df.loc[pd.to_datetime(f'12/31/{i-1}'), 'end']
    projected_df.loc[pd.to_datetime(
        f'12/31/{i}'), 'starting'] = projected_df.loc[pd.to_datetime(f'12/31/{i-1}'), 'end']
i = '2010'
projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 0.23
i = '2011'
projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 3.06
i = '2012'
projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 12.56
i = '2013'
projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 946.92
i = '2014'
projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 378.64
i = '2015'
projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 362.73
i = '2016'
projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 753.26
i = '2017'
projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 10859.56
i = '2018'
projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 4165.61
i = '2019'
projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 7420.84
i = '2020'
projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 18795.20
i = '2021'
projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 57238.62
i = '2022'
projected_df.loc[pd.to_datetime(f'12/31/{i}'), 'price'] = 19780
# st.write(projected_df)
#
c = 2.8294*(10 ** (-21))
m = 5.0046
n = 2.0669
projected_df['S2F'] = projected_df['end']/projected_df['added']
projected_df['S2F_Price'] = c*projected_df['end']**m/projected_df['added']**n
# st.write(projected_df)
projected_df.reset_index(drop=False, inplace=True)

projected_df = projected_df[projected_df.year.dt.year > int(year)-2]

tops = st.sidebar.selectbox('Choose multiple to measure top', [1, 1.5, 2, 2.5, 3, 5], index=2)
bottoms = st.sidebar.selectbox('Choose a fraction to measure bottom',
                               [.9, .8, .7, .6, .5, .4, .3], index=4)

block_price_df.price = block_price_df.price.astype(float)
block_price_df['range'] = (tops)*block_price_df['s2fg_price']
block_price_df['bottom_range'] = bottoms*block_price_df['s2fg_price']


projected_mult = st.sidebar.selectbox('Choose multiple to project top', [
                                      1, 1.5, 2, 2.5, 3, 3.5], index=2)
projected_lows = st.sidebar.selectbox(
    'Choose a fraction to project bottoms', [.9, .8, .7, .6, .5, .4, .3], index=4)
projected_df['range'] = (projected_mult)*projected_df.S2F_Price
projected_df['bottoms'] = projected_lows*projected_df.S2F_Price


fig = px.line(block_price_df, x='date', y='s2fg_price', width=800, height=700,
              title='Price = C * Stock^m/Flow^n   c= 2.8294*(10^(-21)); m = 5.0046; n = 2.0669')
fig.add_trace(go.Scatter(x=block_price_df['date'],
              y=block_price_df.price, name='Price', mode='markers'))
fig.add_trace(go.Scatter(x=projected_df.year, y=projected_df.S2F_Price,
                         name='Projected', mode='lines+markers'))
fig.add_trace(go.Scatter(x=block_price_df.date, y=block_price_df.range,
              name="Above S2FG Price", mode='lines'))
fig.add_trace(go.Scatter(x=block_price_df.date, y=block_price_df.bottom_range,
                         name='Below S2FG Price', mode='lines'))

fig.add_trace(go.Scatter(x=projected_df.year, y=projected_df.range,
              name='Above S2FG Price', mode='lines'))
fig.add_trace(go.Scatter(x=projected_df.year, y=projected_df.bottoms,
                         name='Below S2FG Price', mode='lines'))

fig.update_yaxes(title_text="Value in US$", type="log")
fig.update_xaxes(title_text='Date')
st.plotly_chart(fig, use_container_width=True)
# st.plotly_chart(fig)
