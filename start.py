import os
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from mplfinance.original_flavor import candlestick_ohlc
from matplotlib import pyplot as patches
import pandas as pd
from forexconnect import fxcorepy, ForexConnect, Common
import pandas_ta as ta
from scipy.stats import linregress
import math
#import close

#### All hours in GMT

graph_back_test=False
live = True
mail = True
Dict = {
    'FXCM': {
        'str_user_i_d': '71591187',
        'str_password': 's8ykzvw',
        'str_connection': 'Demo',
        'str_account': '71591187',
        # 'str_user_i_d': '87053959',
        # 'str_password': 'S4Tpj3P!zz.Mm2p',
        # 'str_connection': 'Real',
        # 'str_account': '87053959',
        'str_url': "http://www.fxcorporate.com/Hosts.jsp",
        'str_session_id': None,
        'str_pin': None,
        'str_table': 'orders',
    },
    'indicators': {
        'sd': datetime.now() - relativedelta(weeks=4),
        'ed': datetime.now(),
    },
    'channel_length': 27 * 3,
    'amount': 1,
    'instrument':
        {

             4: {
                 'hour_open': 0,  # opening time in UTC (for midnight put 24)
                 'hour_close': 21,  # closing time in UTC
                 'day_open': 6,  # 6 is Sunday
                 'day_close': 4,  # 4 is Friday
                 'FX': ['UK100', 'AUD/CAD', 'AUD/CHF', 'AUD/JPY', 'AUD/NZD', 'AUD/USD',
                        'CAD/CHF', 'CAD/JPY', 'CHF/JPY', 'GBP/AUD', 'GBP/CAD',
                        'GBP/CHF', 'GBP/JPY', 'GBP/NZD', 'GBP/USD', 'NZD/CAD',
                        'NZD/CHF', 'NZD/JPY', 'NZD/USD', 'TRY/JPY', 'USD/CAD',
                        'USD/CHF', 'USD/CNH', 'USD/HKD', 'USD/JPY', 'USD/MXN',
                        'USD/NOK', 'USD/SEK', 'USD/ZAR', 'XAG/USD', 'XAU/USD',
                        'USD/ILS', 'BTC/USD', 'BCH/USD', 'ETH/USD', 'LTC/USD',
                        'JPN225', 'NAS100', 'NGAS', 'SPX500', 'US30', 'VOLX',
                        'US2000', 'AUS200', 'UKOil', 'USOil', 'USOilSpot', 'UKOilSpot',
                        'EMBasket', 'USDOLLAR', 'JPYBasket', 'CryptoMajor']},


             5: {
                 'hour_open': 1,  # opening time in UTC (for midnight put 24)
                 'hour_close': 19,  # closing time in UTC
                 'FX': ['GER30', 'HKG33', 'CHN50', 'UK100']},

             6: {
                 'hour_open': 0,  # opening time in UTC (for midnight put 24)
                 'hour_close': 18,  # closing time in UTC
                 'FX': ['SOYF', 'WHEATF', 'CORNF']},

             7: {
                 'hour_open': 14,  # opening time in UTC (for midnight put 24)
                 'hour_close': 20,  # closing time in UTC
                 'FX': ['ESPORTS', 'BIOTECH', 'FAANG',
                        'CHN.TECH', 'CHN.ECOMM',
                        'AIRLINES', 'CASINOS',
                        'TRAVEL', 'US.ECOMM',
                        'US.BANKS', 'US.AUTO',
                        'WFH', 'URANIUM']},
            1: {
                'hour_open': 0,  # opening time in UTC (for midnight put 24)
                'hour_close': 21,  # closing time in UTC
                'day_open': 6,  # 6 is Sunday
                'day_close': 4,  # 4 is Friday
                'FX': ['EUR/AUD', 'EUR/CAD',
                       'EUR/CHF', 'EUR/GBP', 'EUR/JPY', 'EUR/NOK', 'EUR/NZD',
                       'EUR/SEK', 'EUR/TRY', 'EUR/USD']},
            2: {
                'hour_open': 6,  # opening time in UTC (for midnight put 24)
                'hour_close': 20,  # closing time in UTC
                'FX': ['EUSTX50', 'FRA40']},
            3: {
                'hour_open': 6,  # opening time in UTC (for midnight put 24)
                'hour_close': 18,  # closing time in UTC
                'FX': ['ESP35', 'Bund']},

        },
    }
backtest_result=[]
# Set how floating-point errors are handled
np.seterr('ignore')

def should_open_buy_trade(df,idx):
    candle_m2 = (df.iloc[idx]['AskClose'] - df.iloc[idx]['AskOpen']) / (
                df.iloc[idx]['AskHigh'] - df.iloc[idx]['AskLow'])
    temp_macd = df['peaks_macd'][:idx].dropna()
    temp_delta = df['peaks_delta'][:idx].dropna()
    result=None
    if temp_macd.size != 0 and temp_delta.size != 0 :
        idx_last_macd=temp_macd.index[-1]
        idx_last_delta=temp_delta.index[-1]
        if (df.iloc[idx - 2:idx]['AskClose'].mean() > max(df.iloc[idx - 2:idx]['senkou_a'].mean(),df.iloc[idx - 2:idx]['senkou_b'].mean()) and
            df.iloc[idx - 2:idx]['tenkan_avg'].mean() > df.iloc[idx - 2:idx]['kijun_avg'].mean() and
            df.iloc[idx - 2:idx]['AskClose'].mean() > df.iloc[idx - 2:idx]['tenkan_avg'].mean() and
            df.iloc[idx]['macd'] > df.iloc[idx_last_macd]['macd'] and
            (df.iloc[idx]['AskClose'] - df.iloc[idx]['kijun_avg'])>(df.iloc[idx]['kijun_avg'] - df.iloc[idx - 27:idx]['AskClose'].min()) and
            df.iloc[idx-27:idx]['rsi'][df['tenkan_avg']>df['kijun_avg']].mean() < 60 and\
            df.iloc[idx-30:idx-27]['chikou'].mean() > max(df.iloc[idx-30:idx-27]['senkou_a'].mean(),df.iloc[idx-30:idx-27]['senkou_b'].mean(),
                                                          df.iloc[idx-30:idx-27]['kijun_avg'].mean(),df.iloc[idx-30:idx-27]['tenkan_avg'].mean())):#and
            #abs(df.iloc[idx]['macd']) > 0.1*(max(df['macd'])+abs(min(df['macd'])))): #and
            #df.iloc[idx - 2:idx]['delta'].mean() > df.iloc[idx - 3:idx - 1]['delta'].mean()):
            result = 'Open Buy'
    return(result)

def should_open_sell_trade(df,idx):
    candle_m2 = (df.iloc[idx]['AskClose'] - df.iloc[idx]['AskOpen']) / (
                df.iloc[idx]['AskHigh'] - df.iloc[idx]['AskLow'])
    temp_macd = df['peaks_macd'][:idx].dropna()
    temp_delta = df['peaks_delta'][:idx].dropna()
    result=None
    if temp_macd.size != 0 and temp_delta.size != 0 :
        idx_last_macd=temp_macd.index[-1]
        idx_last_delta=temp_delta.index[-1]
        if (df.iloc[idx - 2:idx]['AskClose'].mean() < min(df.iloc[idx - 2:idx]['senkou_a'].mean(),df.iloc[idx - 2:idx]['senkou_b'].mean()) and
            df.iloc[idx - 2:idx]['tenkan_avg'].mean() < df.iloc[idx - 2:idx]['kijun_avg'].mean() and
            df.iloc[idx - 2:idx]['AskClose'].mean() < df.iloc[idx - 2:idx]['tenkan_avg'].mean() and
            df.iloc[idx]['macd'] < df.iloc[idx_last_macd]['macd'] and
            df.iloc[idx - 27:idx]['rsi'][df['tenkan_avg']<df['kijun_avg']].mean() > 40 and
            (df.iloc[idx]['kijun_avg'] - df.iloc[idx]['AskClose']) > (df.iloc[idx - 27:idx]['AskClose'].max()-df.iloc[idx]['kijun_avg']) and\
            df.iloc[idx-30:idx-27]['chikou'].mean() < min(df.iloc[idx-30:idx-27]['senkou_a'].mean(),df.iloc[idx-30:idx-27]['senkou_b'].mean(),
                                                          df.iloc[idx-30:idx-27]['kijun_avg'].mean(),df.iloc[idx-30:idx-27]['tenkan_avg'].mean())):# and
            #abs(df.iloc[idx]['macd']) > 0.1*(max(df['macd'])+abs(min(df['macd'])))):# and
            #df.iloc[idx - 2:idx]['delta'].mean() < df.iloc[idx - 3:idx - 1]['delta'].mean()):
            result = 'Open Sell'
    return(result)

def should_close_buy_trade(df,idx,idx_open):
    candle_m2 = (df.iloc[idx]['BidClose'] - df.iloc[idx]['BidOpen']) / (df.iloc[idx]['BidHigh'] - df.iloc[idx]['BidLow'])
    candle_m3 = (df.iloc[idx-1]['BidClose'] - df.iloc[idx-1]['BidOpen']) / (df.iloc[idx-1]['BidHigh'] - df.iloc[idx-1]['BidLow'])
    candle_m4 = (df.iloc[idx - 2]['BidClose'] - df.iloc[idx - 2]['BidOpen']) / (
                df.iloc[idx - 2]['BidHigh'] - df.iloc[idx - 2]['BidLow'])
    #
    if df.iloc[idx]['BidClose'] < max(df.iloc[idx]['senkou_a'],df.iloc[idx]['senkou_b']) and\
        df.iloc[idx]['signal'] < df.iloc[idx]['macd'] and\
        df.iloc[idx-7:idx]['rsi'].mean() > 60 :
        result = 'Kill for in Kumo'
    elif df.iloc[idx-3:idx]['tenkan_avg'].mean() < df.iloc[idx-3:idx]['kijun_avg'].mean() and \
        df.iloc[idx-2:idx]['kijun_avg'].mean() < df.iloc[idx-3:idx-1]['kijun_avg'].mean() and \
        df.iloc[idx]['macd'] < df.iloc[idx - 1]['macd'] and\
        df.iloc[idx-27:idx]['rsi'][df['tenkan_avg']>df['kijun_avg']].mean() > 50 and \
        df.iloc[idx-30:idx-27]['chikou'].mean() < max(df.iloc[idx-30:idx-27]['AskClose'].mean(),df.iloc[idx-30:idx-27]['kijun_avg'].mean(),df.iloc[idx-30:idx-27]['tenkan_avg'].mean()):
        result = 'Kill for Tenkan crossing Kijun'
    elif df.iloc[idx-3:idx]['AskClose'].mean() < df.iloc[idx-3:idx]['tenkan_avg'].mean() and \
        df.iloc[idx]['signal'] > df.iloc[idx]['macd'] and \
        df.iloc[idx-2:idx]['tenkan_avg'].mean()<df.iloc[idx-3:idx-1]['tenkan_avg'].mean() and \
        df.iloc[idx-30:idx-27]['chikou'] .mean()< max(df.iloc[idx-30:idx-27]['AskClose'].mean(),df.iloc[idx-30:idx-27]['kijun_avg'].mean(),df.iloc[idx-30:idx-27]['tenkan_avg'].mean()) and\
        df.iloc[idx-7:idx]['rsi'][df['AskClose']>df['tenkan_avg']].mean() > 60 :
        result = 'Kill for below Tenkans'
    elif df.iloc[idx]['AskClose'] < df.iloc[idx]['kijun_avg'] and \
        df.iloc[idx]['macd'] < df.iloc[idx - 1]['macd'] and \
        df.iloc[idx]['signal'] > df.iloc[idx]['macd'] and\
        df.iloc[idx-7:idx]['rsi'][df['AskClose']>df['kijun_avg']].mean() > 60 and \
        df.iloc[idx-30:idx-27]['chikou'].mean() < max(df.iloc[idx-30:idx-27]['AskClose'].mean(),df.iloc[idx-30:idx-27]['kijun_avg'].mean(),df.iloc[idx-30:idx-27]['tenkan_avg'].mean()):
        result = 'Kill for crossing Kijun'
    elif df.iloc[idx-3:idx]['AskClose'].mean() < df.iloc[idx_open-27:idx_open]['AskLow'].min() and \
        df.iloc[idx-3:idx]['AskClose'].mean() < df.iloc[idx-3:idx]['tenkan_avg'].mean() and\
        df.iloc[idx-3:idx]['tenkan_avg'].mean() < df.iloc[idx-3:idx]['kijun_avg'].mean() and\
        df.iloc[idx-30:idx-27]['chikou'].mean() < min(df.iloc[idx-30:idx-27]['kijun_avg'].mean(),df.iloc[idx-30:idx-27]['tenkan_avg'].mean()) :
        result = 'Kill for wrong direction'
    elif (df.iloc[idx]['AskClose'] - df.iloc[idx]['AskOpen']) < 0 and \
        abs(df.iloc[idx]['AskClose'] - df.iloc[idx]['AskOpen'])>abs(df.iloc[idx-7:idx]['AskHigh'].max() - df.iloc[idx-7:idx]['AskLow'].min()) and\
        df.iloc[idx]['AskClose'] < df.iloc[idx-3:idx]['tenkan_avg'].mean():
        result = 'Kill for high tendance change'
    else:
        result = None

    return result

def should_close_sell_trade(df,idx,idx_open):
    candle_m2 = (df.iloc[idx]['BidClose'] - df.iloc[idx]['BidOpen']) / (df.iloc[idx]['BidHigh'] - df.iloc[idx]['BidLow'])
    candle_m3 = (df.iloc[idx-1]['BidClose'] - df.iloc[idx-1]['BidOpen']) / (df.iloc[idx-1]['BidHigh'] - df.iloc[idx-1]['BidLow'])
    candle_m4 = (df.iloc[idx - 2]['BidClose'] - df.iloc[idx - 2]['BidOpen']) / (
                df.iloc[idx - 2]['BidHigh'] - df.iloc[idx - 2]['BidLow'])

    if df.iloc[idx]['AskClose'] > min(df.iloc[idx]['senkou_a'],df.iloc[idx]['senkou_b']) \
        and df.iloc[idx]['signal'] > df.iloc[idx]['macd']  and\
        df.iloc[idx-7:idx]['rsi'].mean() < 40:
        result = 'Kill for in Kumo'
    elif df.iloc[idx]['tenkan_avg'] > df.iloc[idx]['kijun_avg'] and \
        df.iloc[idx-2:idx]['kijun_avg'].mean() > df.iloc[idx-3:idx-1]['kijun_avg'].mean()and \
        df.iloc[idx]['macd'] > df.iloc[idx]['signal']  and\
        df.iloc[idx-27:idx]['rsi'][df['tenkan_avg']<df['kijun_avg']].mean() < 50 and \
        df.iloc[idx-30:idx-27]['chikou'].mean() > min(df.iloc[idx-30:idx-27]['AskClose'].mean(),df.iloc[idx-30:idx-27]['kijun_avg'].mean(),df.iloc[idx-30:idx-27]['tenkan_avg'].mean()) :
        result = 'Kill for Tenkan crossing Kijun'
    elif df.iloc[idx-3:idx]['AskClose'].mean() > df.iloc[idx-3:idx]['tenkan_avg'].mean() and \
        df.iloc[idx]['signal'] < df.iloc[idx]['macd'] and\
        df.iloc[idx-7:idx]['rsi'][df['AskClose']<df['tenkan_avg']].mean() < 40 and \
        df.iloc[idx-2:idx]['tenkan_avg'].mean()>df.iloc[idx-3:idx-1]['tenkan_avg'].mean() and \
        df.iloc[idx-30:idx-27]['chikou'].mean() > min(df.iloc[idx-30:idx-27]['AskClose'].mean(),df.iloc[idx-30:idx-27]['kijun_avg'].mean(),df.iloc[idx-30:idx-27]['tenkan_avg'].mean()):
        result = 'Kill for below Tenkans'
    elif df.iloc[idx]['AskClose'] > df.iloc[idx]['kijun_avg'] and \
        df.iloc[idx]['kijun_avg'] > df.iloc[idx-1]['kijun_avg'] and \
        df.iloc[idx]['macd'] > df.iloc[idx-1]['macd'] and\
        df.iloc[idx-7:idx]['rsi'][df['AskClose']<df['kijun_avg']].mean() < 40 and \
        df.iloc[idx-30:idx-27]['chikou'].mean() > min(df.iloc[idx-30:idx-27]['AskClose'].mean(),df.iloc[idx-30:idx-27]['kijun_avg'].mean(),df.iloc[idx-27]['tenkan_avg'].mean()):
        result = 'Kill for crossing Kijun'
    elif df.iloc[idx-3:idx]['AskClose'].mean() > df.iloc[idx_open-10:idx_open]['AskHigh'].max() and \
        df.iloc[idx-3:idx]['AskClose'].mean() > df.iloc[idx-3:idx]['tenkan_avg'].mean() and \
        df.iloc[idx-3:idx]['tenkan_avg'].mean() > df.iloc[idx-3:idx]['kijun_avg'].mean() and\
        df.iloc[idx-30:idx-27]['chikou'].mean() > max(df.iloc[idx-30:idx-27]['kijun_avg'].mean(),df.iloc[idx-30:idx-27]['tenkan_avg'].mean()):
        result = 'Kill for wrong direction'
    elif (df.iloc[idx]['AskClose'] - df.iloc[idx]['AskOpen']) > 0 and \
        abs(df.iloc[idx]['AskClose'] - df.iloc[idx]['AskOpen'])>abs(df.iloc[idx-7:idx]['AskHigh'].max() - df.iloc[idx-7:idx]['AskLow'].min()) and\
        df.iloc[idx]['AskClose'] > df.iloc[idx-3:idx]['tenkan_avg'].mean():
        result = 'Kill for high tendance change'
    else:
        result = None

    return result

def open_trade(df, fx, tick, trading_settings_provider, dj, idx):
    def set_amount(lots, dj):
        account = Common.get_account(fx, Dict['FXCM']['str_account'])
        base_unit_size = trading_settings_provider.get_base_unit_size(tick, account)
        amount = base_unit_size * Dict['amount']
        return amount

    box_def = False
    high_box = 0
    low_box = 0
    type_signal = 'No'
    tp = 0
    sl = 0
    index_peak = 0
    #df = analysis(df, idx, tick)
    result_buy = should_open_buy_trade(df, idx)
    result_sell = should_open_sell_trade(df, idx)
    if result_buy != None:
        # min_entry = round((max(df.iloc[-27:-2]['kijun_avg']) - min(df.iloc[-27:-2]['AskLow'])) / (
        #     abs(df.iloc[-2]['BidClose'] - df.iloc[-2]['AskClose'])), 2)
        # if min_entry >= 2:
        try:
            amount = (set_amount(Dict['amount'], dj))
            type_signal = ' BUY Amount:' + str(amount) # + ' Bid/Ask:' + str(min_entry)
            request = fx.create_order_request(
                order_type=fxcorepy.Constants.Orders.TRUE_MARKET_OPEN,
                ACCOUNT_ID=Dict['FXCM']['str_account'],
                BUY_SELL=fxcorepy.Constants.BUY,
                AMOUNT=round(amount, 2),
                SYMBOL=tick,
            )
            fx.send_request(request)
        except Exception as e:
            type_signal = type_signal + ' not working for ' + str(e)
            pass
    elif result_sell != None:

        # min_entry = round((max(df.iloc[-27:-2]['AskHigh']) - min(df.iloc[-27:-2]['kijun_avg'])) / (
        #     abs(df.iloc[-2]['BidClose'] - df.iloc[-2]['AskClose'])), 2)
        # if min_entry >= 2:
        try:
            amount = (set_amount(Dict['amount'], dj))
            type_signal = ' Sell Amount: ' + str(amount) # + ' Bid/Ask: ' + str(min_entry)
            request = fx.create_order_request(
                order_type=fxcorepy.Constants.Orders.TRUE_MARKET_OPEN,
                ACCOUNT_ID=Dict['FXCM']['str_account'],
                BUY_SELL=fxcorepy.Constants.SELL,
                AMOUNT=round(amount, 2),
                SYMBOL=tick,
            )
            fx.send_request(request)
        except Exception as e:
            type_signal = type_signal + ' not working for ' + str(e)

    return df, type_signal, idx, box_def, high_box, low_box, tp, sl, index_peak

def close_trade(df, fx, tick, dj, idx):
    try:
        open_rev_index = [df.index[df['Date'].dt.strftime("%m%d%Y%H") == dj.loc[0, 'tick_time'].strftime("%m%d%Y%H")]][0][0]
    except:
        open_rev_index = 1
    type_signal = 'No'
    open_price = dj.loc[0, 'tick_open_price']
    price = dj.loc[0, 'tick_price']
    box_def = False
    high_box = 0
    low_box = 0
    index_peak = None
    #df = analysis(df, open_rev_index, tick)
    tp = dj.loc[0, 'tick_limit']
    sl = dj.loc[0, 'tick_stop']
    offer = Common.get_offer(fx, tick)
    order = Common.get_offer(fx, tick)
    buy = fxcorepy.Constants.BUY
    sell = fxcorepy.Constants.SELL
    buy_sell = sell if dj.loc[0, 'tick_type'] == buy else buy
    candle_2 = (df.iloc[idx]['BidClose'] - df.iloc[idx]['BidOpen']) / \
               (df.iloc[idx]['BidHigh'] - df.iloc[idx]['BidLow'])
    window_of_interest = 27
    margin = abs(0.1 * (np.nanmax(df.iloc[idx-window_of_interest:idx]['BidHigh']) - np.nanmin(
        df.iloc[idx-window_of_interest:idx]['BidLow'])))

    if open_rev_index <= 1:
        print('open_rev_index too small')
    else:
        # BUY CONDIDITIONS
        if dj.loc[0, 'tick_type'] == 'B':
            current_ratio = (price - open_price) / (open_price - df.iloc[open_rev_index-7:open_rev_index]['BidLow'].min())
            result = should_close_buy_trade(df,idx,open_rev_index)
            if result != None:
                if price > open_price:
                    try:
                        type_signal = ' Buy : ' + str(result)
                        request = fx.create_order_request(
                            order_type=fxcorepy.Constants.Orders.TRUE_MARKET_CLOSE,
                            OFFER_ID=offer.offer_id,
                            ACCOUNT_ID=Dict['FXCM']['str_account'],
                            BUY_SELL=buy_sell,
                            AMOUNT=int(dj.loc[0, 'tick_amount']),
                            TRADE_ID=dj.loc[0, 'tick_id']
                        )
                        resp = fx.send_request(request)
                    except Exception as e:
                        type_signal = type_signal + ' not working for ' + str(e)
                        pass
                else:
                    try:
                        sl = price - (df.iloc[idx-3:idx+1]['BidHigh'].max()-df.iloc[idx-3:idx+1]['BidLow'].min())/2
                        tp = price + (df.iloc[idx-3:idx+1]['BidHigh'].max()-df.iloc[idx-3:idx+1]['BidLow'].min())
                        type_signal = ' Buy : Adjust ' + str(result)
                        request = fx.create_order_request(
                                    order_type=fxcorepy.Constants.Orders.LIMIT,
                                    command=fxcorepy.Constants.Commands.CREATE_ORDER,
                                    OFFER_ID=offer.offer_id,
                                    ACCOUNT_ID=Dict['FXCM']['str_account'],
                                    BUY_SELL=buy_sell,
                                    AMOUNT=int(dj.loc[0, 'tick_amount']),
                                    TRADE_ID=dj.loc[0, 'tick_id'],
                                    RATE=tp,
                        )
                        resp = fx.send_request(request)
                        request = fx.create_order_request(
                                    order_type=fxcorepy.Constants.Orders.STOP,
                                    command=fxcorepy.Constants.Commands.CREATE_ORDER,
                                    OFFER_ID=offer.offer_id,
                                    ACCOUNT_ID=Dict['FXCM']['str_account'],
                                    BUY_SELL=buy_sell,
                                    AMOUNT=int(dj.loc[0, 'tick_amount']),
                                    TRADE_ID=dj.loc[0, 'tick_id'],
                                    RATE=sl,
                        )
                        resp = fx.send_request(request)
                    except Exception as e:
                        type_signal = type_signal + ' not working for ' + str(e)
                        pass
        # if was sell
        if dj.loc[0, 'tick_type'] == 'S':
            current_ratio = (open_price - price) / (df.iloc[open_rev_index-7:open_rev_index]['BidHigh'].max() - open_price)
            result = should_close_sell_trade (df,idx,open_rev_index)
            if result != None:
                if price < open_price:
                    try:
                        type_signal = ' Sell : ' + str(result)
                        request = fx.create_order_request(
                            order_type=fxcorepy.Constants.Orders.TRUE_MARKET_CLOSE,
                            OFFER_ID=offer.offer_id,
                            ACCOUNT_ID=Dict['FXCM']['str_account'],
                            BUY_SELL=buy_sell,
                            AMOUNT=int(dj.loc[0, 'tick_amount']),
                            TRADE_ID=dj.loc[0, 'tick_id']
                        )
                        resp = fx.send_request(request)
                    except Exception as e:
                        type_signal = type_signal + ' not working for ' + str(e)
                        pass
                else:
                    try:
                        sl = price + (df.iloc[idx-3:idx+1]['BidHigh'].max()-df.iloc[idx-3:idx+1]['BidLow'].min())/2
                        tp = price - (df.iloc[idx-3:idx+1]['BidHigh'].max()-df.iloc[idx-3:idx+1]['BidLow'].min())
                        type_signal = ' Sell : Adjust ' + str(result)
                        request = fx.create_order_request(
                                    order_type=fxcorepy.Constants.Orders.LIMIT,
                                    command=fxcorepy.Constants.Commands.CREATE_ORDER,
                                    OFFER_ID=offer.offer_id,
                                    ACCOUNT_ID=Dict['FXCM']['str_account'],
                                    BUY_SELL=buy_sell,
                                    AMOUNT=int(dj.loc[0, 'tick_amount']),
                                    TRADE_ID=dj.loc[0, 'tick_id'],
                                    RATE=tp,
                        )
                        resp = fx.send_request(request)
                        request = fx.create_order_request(
                                    order_type=fxcorepy.Constants.Orders.STOP,
                                    command=fxcorepy.Constants.Commands.CREATE_ORDER,
                                    OFFER_ID=offer.offer_id,
                                    ACCOUNT_ID=Dict['FXCM']['str_account'],
                                    BUY_SELL=buy_sell,
                                    AMOUNT=int(dj.loc[0, 'tick_amount']),
                                    TRADE_ID=dj.loc[0, 'tick_id'],
                                    RATE=sl,
                        )
                        resp = fx.send_request(request)
                    except Exception as e:
                        type_signal = type_signal + ' not working for ' + str(e)
                        pass

    return df, type_signal, open_rev_index, box_def, high_box, low_box, tp, sl, index_peak

def backtest_strategy(df,fx,trading_settings_provider, tick):
    trades = []
    result=0

    for i in range(7, len(df)):
        result_buy = should_open_buy_trade(df, i)
        result_sell = should_open_sell_trade(df, i)

        if result_buy!=None:
            if trades:
                last_trade_date, trade_type, _, _,_ = trades[-1]
                if trade_type != 'Buy' and trade_type != 'Sell':
                    trades.append((df.iloc[i]['Date'], 'Buy',i,0,None))
            else:
                trades.append((df.iloc[i]['Date'], 'Buy', i,0,None))
        elif result_sell != None:

            if trades:
                last_trade_date, trade_type, _, _,_ = trades[-1]
                if trade_type != 'Sell' and trade_type != 'Buy':
                    trades.append((df.iloc[i]['Date'], 'Sell',i,0,None))
            else:
                trades.append((df.iloc[i]['Date'], 'Sell', i,0,None))

        # Assume closing a trade after a certain condition is met
        if i > 7 and trades:
            last_trade_date, trade_type, id_open,_,_ = trades[-1]
            result_close_buy = should_close_buy_trade(df,i,id_open)
            result_close_sell = should_close_sell_trade(df,i,id_open)
            if trade_type == 'Buy' and df.iloc[i]['Date'] - last_trade_date >= pd.Timedelta(hours=5) and result_close_buy!=None:
                trades.append((df.iloc[i]['Date'], 'Close Buy', i,(df.iloc[i]['BidClose']-df.iloc[id_open]['AskOpen']),result_close_buy))
                result=result+(df.iloc[i]['BidClose']-df.iloc[id_open]['AskOpen'])
            elif trade_type == 'Sell' and df.iloc[i]['Date'] - last_trade_date >= pd.Timedelta(hours=5) and result_close_sell!=None:
                trades.append((df.iloc[i]['Date'], 'Close Sell', i,(df.iloc[id_open]['AskOpen']-df.iloc[i]['BidClose']),result_close_sell))
                result = result + (df.iloc[id_open]['AskOpen'] - df.iloc[i]['BidClose'])
    print(trades)
    print(result)
    return result, trades

def indicators(df):
    def ichimoku(df):
        # Tenkan Sen
        tenkan_max = df['BidHigh'].rolling(window=9).max()
        tenkan_min = df['BidLow'].rolling(window=9).min()
        df['tenkan_avg'] = (tenkan_max + tenkan_min) / 2

        # Kijun Sen
        kijun_max = df['BidHigh'].rolling(window=26).max()
        kijun_min = df['BidLow'].rolling(window=26).min()
        df['kijun_avg'] = (kijun_max + kijun_min) / 2

        # Senkou Span A
        # (Kijun + Tenkan) / 2 Shifted ahead by 26 periods
        df['senkou_a'] = ((df['kijun_avg'] + df['tenkan_avg']) / 2).shift(26)

        # Senkou Span B
        # 52 period High + Low / 2
        senkou_b_max = df['BidHigh'].rolling(window=52).max()
        senkou_b_min = df['BidLow'].rolling(window=52).min()
        df['senkou_b'] = ((senkou_b_max + senkou_b_min) / 2).shift(26)

        # Chikou Span
        df['chikou'] = (df['BidClose']).shift(-26)
        return df

    def macd(df):
        dm = df[['AskClose', 'AskOpen', 'AskLow', 'AskHigh']]
        dm.reset_index(level=0, inplace=True)
        dm.columns = ['ds', 'y', 'open', 'low', 'high']
        # MACD
        exp1 = dm.y.ewm(span=12, adjust=False).mean()
        exp2 = dm.y.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        # Signal Line
        exp3 = macd.ewm(span=9, adjust=False).mean()
        # Cross-Over
        df['macd'] = np.array(macd)
        df['signal'] = np.array(exp3)
        df['delta'] = np.array(macd - exp3)
        return df

    def rsi(df, periods=14, ema=True):
        """
        Returns a pd.Series with the relative strength index.
        """
        close_delta = df['AskClose'].diff()

        # Make two series: one for lower closes and one for higher closes
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)

        if ema == True:
            # Use exponential moving average
            ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
            ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
        else:
            # Use simple moving average
            ma_up = up.rolling(window=periods, adjust=False).mean()
            ma_down = down.rolling(window=periods, adjust=False).mean()

        rsi = ma_up / ma_down
        rsi = 100 - (100 / (1 + rsi))
        return rsi

    def get_ci(high, low, close, lookback):
        # https://medium.com/codex/detecting-ranging-and-trending-markets-with-choppiness-index-in-python-1942e6450b58
        # IF CHOPPINESS INDEX >= 61.8 --> MARKET IS CONSOLIDATING
        # IF CHOPPINESS INDEX <= 38.2 --> MARKET IS TRENDING
        tr1 = pd.DataFrame(high - low).rename(columns={0: 'tr1'})
        tr2 = pd.DataFrame(abs(high - close.shift(1))).rename(columns={0: 'tr2'})
        tr3 = pd.DataFrame(abs(low - close.shift(1))).rename(columns={0: 'tr3'})
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').dropna().max(axis=1)
        atr = tr.rolling(1).mean()
        highh = high.rolling(lookback).max()
        lowl = low.rolling(lookback).min()
        ci = 100 * np.log10((atr.rolling(lookback).sum()) / (highh - lowl)) / np.log10(lookback)
        return ci

    def find_last_peaks(df, ind):
        n = len(df)
        df['peaks'] = np.nan
        df['peaks_macd'] = np.nan
        df['peaks_delta'] = np.nan
        df['slope'] = np.nan
        df['slope_macd'] = np.nan
        for i in range(-ind - 1, -n + 3, -1):
            #if abs(df.iloc[i]['macd']) > 0.1*(max(df['macd'])+abs(min(df['macd']))): #abs(df.iloc[i]['macd']) >= abs(df.iloc[i]['signal']) and
            if (i == -2) and \
                    abs(df.iloc[i]['macd']) >= abs(df.iloc[i - 1]['macd']) and \
                    abs(df.iloc[i]['macd']) >= abs(df.iloc[i - 2]['macd']) and \
                    abs(df.iloc[i]['macd']) >= abs(df.iloc[i - 3]['macd']) and \
                    abs(df.iloc[i]['macd']) >= abs(df.iloc[i + 1]['macd']) :
                df.loc[n + i, 'peaks_macd'] = df.iloc[i]['macd']
                df.loc[n + i, 'peaks'] = df.iloc[i]['AskClose']
            elif (i < -2) and \
                    abs(df.iloc[i]['macd']) >= abs(df.iloc[i - 1]['macd']) and \
                    abs(df.iloc[i]['macd']) >= abs(df.iloc[i - 2]['macd']) and \
                    abs(df.iloc[i]['macd']) >= abs(df.iloc[i - 3]['macd']) and \
                    abs(df.iloc[i]['macd']) >= abs(df.iloc[i + 1]['macd']) and \
                    abs(df.iloc[i]['macd']) >= abs(df.iloc[i + 2]['macd']) and \
                    abs(df.iloc[i]['macd']) >= abs(df.iloc[i + 3]['macd']):
                df.loc[n + i, 'peaks_macd'] = df.iloc[i]['macd']
                df.loc[n + i, 'peaks'] = df.iloc[i]['AskClose']
            if abs(df.iloc[i]['delta']) > 0.1*(max(df['delta'])+abs(min(df['delta']))):
                if (i == -2) and \
                        abs(df.iloc[i]['delta']) >= abs(df.iloc[i - 1]['delta']) and \
                        abs(df.iloc[i]['delta']) >= abs(df.iloc[i - 2]['delta']) and \
                        abs(df.iloc[i]['delta']) >= abs(df.iloc[i - 3]['delta']) and \
                        abs(df.iloc[i]['delta']) >= abs(df.iloc[i + 1]['delta']) :
                    df.loc[n + i, 'peaks_delta'] = df.iloc[i]['delta']
                elif (i < -2) and \
                        abs(df.iloc[i]['delta']) >= abs(df.iloc[i - 1]['delta']) and \
                        abs(df.iloc[i]['delta']) >= abs(df.iloc[i - 2]['delta']) and \
                        abs(df.iloc[i]['delta']) >= abs(df.iloc[i - 3]['delta']) and \
                        abs(df.iloc[i]['delta']) >= abs(df.iloc[i + 1]['delta']) and \
                        abs(df.iloc[i]['delta']) >= abs(df.iloc[i + 2]['delta']) and \
                        abs(df.iloc[i]['delta']) >= abs(df.iloc[i + 3]['delta']):
                    df.loc[n + i, 'peaks_delta'] = df.iloc[i]['delta']
        return df

    df = ichimoku(df)
    df = macd(df)
    df["doji_signal"]=ta.cdl_doji(df['AskOpen'],df['AskHigh'],df['AskLow'],df['AskClose'])
    df['rsi'] = rsi(df, 14, True)
    df['ci'] = get_ci(df['AskHigh'], df['AskLow'], df['AskClose'], 28)
    df = find_last_peaks(df,1)

    return (df)

def df_plot(df, tick, trades, type_signal="", index=0, box_def=False, high_box=0, low_box=0, tp=0, sl=0, index_peak=0):
    def sendemail(attach, subject_mail, body_mail):
        fromaddr = 'sanpexos@hotmail.com'
        toaddr = 'paul.pelluchon.doc@gmail.com'
        password = '@c<Md5&$gzGNU<('

        # instance of MIMEMultipart
        msg = MIMEMultipart()
        # storing the senders email address
        msg['From'] = fromaddr
        # storing the receivers email address
        msg['To'] = toaddr
        # storing the subject
        msg['Subject'] = subject_mail
        # string to store the body of the mail
        body = body_mail
        # attach the body with the msg instance
        msg.attach(MIMEText(body, 'plain'))
        # open the file to be sent
        filename = attach
        attachment = open(attach, 'rb')
        # instance of MIMEBase and named as p
        p = MIMEBase('application', 'octet-stream')
        # To change the payload into encoded form
        p.set_payload((attachment).read())
        # encode into base64
        encoders.encode_base64(p)
        p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
        # attach the instance 'p' to instance 'msg'
        msg.attach(p)
        # creates SMTP session
        s = smtplib.SMTP("smtp.office365.com", 587)
        # start TLS for security
        s.starttls()
        # Authentication
        s.login(fromaddr, password)
        # Converts the Multipart msg into a string
        text = msg.as_string()
        # sending the mail
        s.sendmail(fromaddr, toaddr, text)
        # terminating the session
        s.quit()

    if type_signal != 'No':
        df['index'] = df.index
        # if index > 100:
        #     df = df.iloc[-2*index-1:-1]
        # else:
        #     df = df.iloc[-100:-1]
        print(str(tick) + " " + str(type_signal))
        my_dpi = 120
        min_x = len(df)#27 * 10
        fig = plt.figure(figsize=(2190 / my_dpi, 1200 / my_dpi), dpi=my_dpi)
        fig.suptitle(tick + type_signal, fontsize=12)
        ax1 = plt.subplot2grid((9, 1), (0, 0), rowspan=4)
        ax2 = plt.subplot2grid((9, 1), (4, 0), rowspan=2, sharex=ax1)
        ax3 = plt.subplot2grid((9, 1), (6, 0), rowspan=1, sharex=ax1)
        ax4 = plt.subplot2grid((9, 1), (7, 0), rowspan=2, sharex=ax1)

        ###AX1
        # Where enter
        ax1.plot(df.index[-min_x:], df['tenkan_avg'][-min_x:], linewidth=2, color='red')
        ax1.plot(df.index[-min_x:], df['kijun_avg'][-min_x:], linewidth=2, color='blue')
        ax1.plot(df.index[-min_x:], df['senkou_a'][-min_x:], linewidth=0.5, color='black')
        ax1.plot(df.index[-min_x:], df['senkou_b'][-min_x:], linewidth=0.5, color='black')
        ax1.plot(df.index[-min_x:], df['chikou'][-min_x:], linewidth=2, color='brown')
        ax1.plot(df.index[-min_x:], df['peaks'][-min_x:], color='orange', marker='s')

        if type_signal != "":
            ax1.axhline(y=float(df.iloc[index]['AskClose']), color='black', linewidth=1, linestyle='-.')
            if tp != 0:
                 ax1.axhline(y=float(tp), color='blue', linewidth=1, linestyle='-.')
            if sl != 0:
                 ax1.axhline(y=float(sl), color='red', linewidth=1, linestyle='-.')
            ax1.plot(df.iloc[index]['index'], df.iloc[index]['AskClose'], 'black', marker='s')
            ax1.axvline(x=df.iloc[index]['index'], color='black', linewidth=1, linestyle='-.')
        if 'slope' in df.columns:
            ax1.plot([df.loc[3, 'slope'], df.loc[4, 'slope']], [df.loc[1, 'slope'], df.loc[2, 'slope']], linewidth=2,
                     color='yellow', marker='s')
        ax1.fill_between(df.index[-min_x:], df['senkou_a'][-min_x:], df['senkou_b'][-min_x:],
                         where=df['senkou_a'][-min_x:] >= df['senkou_b'][-min_x:],
                         color='lightgreen')
        ax1.fill_between(df.index[-min_x:], df['senkou_a'][-min_x:], df['senkou_b'][-min_x:],
                         where=df['senkou_a'][-min_x:] < df['senkou_b'][-min_x:],
                         color='lightcoral')
        quotes = [tuple(x) for x in df[['index', 'AskOpen', 'AskHigh', 'AskLow', 'AskClose']].values]
        candlestick_ohlc(ax1, quotes, width=0.2, colorup='g', colordown='r')
        ax1.grid()
        low_limit = np.nanmin(df['AskLow'][-min_x:])
        high_limit = np.nanmax(df['AskHigh'][-min_x:])
        ax1.set_ylim(low_limit - 0.1 * (high_limit - low_limit), high_limit + 0.1 * (high_limit - low_limit))
        ax1.set_xlim(np.nanmin(df['index'][-min_x:]), np.nanmax(df['index'][-min_x:]))
        ax1.set(xlabel=None)

        ###AX2
        ax2.bar(df.index[-min_x:], df['macd'][-min_x:], color='grey')
        ax2.plot(df.index[-min_x:], df['signal'][-min_x:], color='red')
        if 'slope_macd' in df.columns:
            ax2.plot([df.loc[3, 'slope_macd'], df.loc[4, 'slope_macd']], [df.loc[1, 'slope_macd'], df.loc[2, 'slope_macd']],
                     linewidth=2,
                     color='yellow', marker='s')
        ax2.axvline(x=df.iloc[index]['index'], color='black', linewidth=1, linestyle='-.')
        ax2.plot(df.index[-min_x:], df['peaks_macd'][-min_x:], color='orange', marker='s')
        ax2.set_ylim(np.nanmin(df['macd'][-min_x:]), np.nanmax(df['macd'][-min_x:]))
        ax2.grid()
        ax2.set(xlabel=None)

        ###AX3
        ax3.bar(df.index[-min_x:], df['delta'][-min_x:], color='black')
        ax3.axvline(x=df.iloc[index]['index'], color='black', linewidth=1, linestyle='-.')
        ax3.plot(df.index[-min_x:], df['peaks_delta'][-min_x:], color='orange', marker='s')
        ax3.set_ylim(np.nanmin(df['delta'][-min_x:]), np.nanmax(df['delta'][-min_x:]))
        ax3_0=ax3.twinx()
        ax3_0.plot(df.index[-min_x:], df['Volume'][-min_x:], color='orange')
        ax3_0.axhline(y=0.4*(df['Volume'].max()-df['Volume'].min()), color='blue', linewidth=1, linestyle='-.')
        ax3.grid()
        ax3.set(xlabel=None)

        ###AX4
        ax4.plot(df.index[-min_x:], df['rsi'][-min_x:], color='black')
        ax4.axhline(y=30, color='grey', linestyle='-.')
        ax4.axhline(y=70, color='grey', linestyle='-.')
        ax4.plot(df.index[-min_x:], df['ci'][-min_x:], color='orange')
        ax4.axhline(y=38.2, color='yellow', linestyle='-.')
        ax4.axhline(y=61.8, color='yellow', linestyle='-.')
        ax4.axvline(x=df.iloc[index]['index'], color='black', linewidth=1, linestyle='-.')
        ax4.set_ylim((0, 100))
        ax4.grid()

        if live==False:
            for i in range(0,len(trades)):
                last_trade_date, trade_type, id, delta, type_close = trades[i]
                if trade_type=='Buy':
                    col='green'
                elif trade_type=='Sell':
                    col='red'
                elif trade_type=='Close Buy':
                    col='cyan'
                elif trade_type=='Close Sell':
                    col='purple'
                ax1.axvline(x=float(df.iloc[id]['index']), color=col, linewidth=1, linestyle='-.')
                ax1.plot(df.iloc[id]['index'], df.iloc[id]['AskClose'], color=col, marker='s')
                ax2.axvline(x=float(df.iloc[id]['index']), color=col, linewidth=1, linestyle='-.')
                ax3.axvline(x=float(df.iloc[id]['index']), color=col, linewidth=1, linestyle='-.')
                ax4.axvline(x=float(df.iloc[id]['index']), color=col, linewidth=1, linestyle='-.')
                ax1.text(df.iloc[id]['index'], df.iloc[id]['AskClose'],type_close, ha='right')
        #plt.show()
        if graph_back_test== True:
            name_file=tick.replace('/','') +'.png'
        else:
            name_file='filename.png'
        fig.savefig(name_file)
        if mail ==True :
            try:
                sendemail(attach=name_file, subject_mail=str(tick), body_mail=str(tick) + " " + str(type_signal))
            except Exception as e:
                print("issue with mails for " + tick)
        plt.close()

def session_status_changed(session: fxcorepy.O2GSession,
                           status: fxcorepy.AO2GSessionStatus.O2GSessionStatus):
    print("Trading session status: " + str(status))

def check_trades(tick, fx):
    ## improve check trades : Common.convert_row_to_dataframe()

    open_pos_status = 'No'
    orders_table = fx.table_manager.get_table(ForexConnect.TRADES)
    offers_table = fx.table_manager.get_table(ForexConnect.OFFERS)
    dj = pd.DataFrame(dtype="string")
    if len(orders_table) != 0:
        k = 0
        for row in orders_table:
            k = k + 1
            if row.instrument == tick:
                dj.loc[0, 'order_stop_id'] = row.stop_order_id
                dj.loc[0, 'order_limit_id'] = row.limit_order_id
                open_pos_status = 'Yes'
                dj.loc[0, 'tick_time'] = row.open_time
                dj.loc[0, 'tick_id'] = row.trade_id
                dj.loc[0, 'tick_type'] = row.buy_sell
                dj.loc[0, 'tick_limit'] = row.limit
                dj.loc[0, 'tick_stop'] = row.stop
                dj.loc[0, 'tick_price'] = row.close
                dj.loc[0, 'tick_open_price'] = row.open_rate
                dj.loc[0, 'tick_amount'] = row.amount
                dj.loc[0, 'profit_loss'] = row.pl
                dj.loc[0, 'pip_size'] = fx.table_manager.get_table(ForexConnect.OFFERS).get_row(k).point_size
                dj.loc[0, 'pip_cost'] = fx.table_manager.get_table(ForexConnect.OFFERS).get_row(k).pip_cost
    # get the pip size in all cases
    k = 0
    for row in offers_table:
        k = k + 1
        if row.instrument == tick:
            try:
                dj.loc[0, 'pip_cost'] = fx.table_manager.get_table(ForexConnect.OFFERS).get_row(k).pip_cost
                dj.loc[0, 'pip_size'] = fx.table_manager.get_table(ForexConnect.OFFERS).get_row(k).point_size
            except:
                dj.loc[0, 'pip_cost'] = 1
                dj.loc[0, 'pip_size'] = 1
    return open_pos_status, dj

def main():
    #print(str(datetime.now().strftime("%H:%M:%S")))
    with ForexConnect() as fx:
        # try:
        fx.login(Dict['FXCM']['str_user_i_d'], Dict['FXCM']['str_password'], Dict['FXCM']['str_url'],
                 Dict['FXCM']['str_connection'], Dict['FXCM']['str_session_id'], Dict['FXCM']['str_pin'],
                 session_status_callback=session_status_changed)
        login_rules = fx.login_rules
        trading_settings_provider = login_rules.trading_settings_provider
        for l0 in range(1, len(Dict['instrument'])):
            FX = Dict['instrument'][l0]['FX']

            for l1 in range(0, len(FX)):
                tick = FX[l1]
                print(FX[l1])
                # H1
                df = pd.DataFrame(fx.get_history(FX[l1], 'H1', Dict['indicators']['sd'], Dict['indicators']['ed']))
                df = indicators(df)
                if live == True:
                    if trading_settings_provider.get_market_status(FX[l1]) == fxcorepy.O2GMarketStatus.MARKET_STATUS_OPEN:
                        # Check the current open positions
                        open_pos_status, dj = check_trades(FX[l1], fx)
                        # if status not open then check if to open
                        if open_pos_status == 'No':
                            # if df.iloc[-2]['AskHigh'] + margin > df.iloc[-3]['AskLow']:
                            if l0 == 1 and datetime.now().weekday() == Dict['instrument'][l0]['day_open'] and int(
                                    datetime.now().strftime("%H")) < Dict['instrument'][l0]['hour_open']:
                                print('forex not hour')
                            elif l0 > 1 and int(datetime.now().strftime("%H")) < Dict['instrument'][l0]['hour_open']:
                                print('other not hour')
                            else:
                                df, type_signal, index, box_def, high_box, low_box, tp, sl, index_peak = \
                                    open_trade(df, fx, FX[l1], trading_settings_provider, dj,len(df)-2)
                                df_plot(df, tick,None, type_signal, index, box_def, high_box, low_box, tp, sl, index_peak)
                        # if status is open then check if to close
                        elif open_pos_status == 'Yes':
                            df, type_signal, index, box_def, high_box, low_box, tp, sl, index_peak = \
                                close_trade(df, fx, FX[l1], dj,len(df)-2)
                            df_plot(df, tick,None, type_signal, index, box_def, high_box, low_box, tp, sl, index_peak)
                else:
                    if l0==1:
                        # back-test
                        result, trades = backtest_strategy(df,fx,trading_settings_provider, tick)
                        backtest_result.append(result)
                        if graph_back_test == True:
                            df_plot(df, tick, trades)
    print(sum(backtest_result))

try:
    SOME_SECRET = os.environ["SOME_SECRET"]
except KeyError:
    SOME_SECRET = "Token not available"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if __name__ == "__main__":
    main()
    finish_time = str((datetime.now()).strftime("%H:%M:%S"))
    logger.info(f"Finish at: {finish_time}")
