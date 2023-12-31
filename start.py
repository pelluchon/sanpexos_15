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
import math
import close

#### All hours in GMT

Dict = {
    'FXCM': {
        'str_user_i_d': '71585261',
        'str_password': 'ugpj7fu',
        'str_connection': 'Demo',
        'str_account': '71585261',
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
        'sd': datetime.now() - relativedelta(weeks=16),
        'ed': datetime.now(),
    },
    'channel_length': 27 * 3,
    'amount': 1,
    'instrument':
        {

            1: {
                'hour_open': 0,  # opening time in UTC (for midnight put 24)
                'hour_close': 21,  # closing time in UTC
                'day_open': 6,  # 6 is Sunday
                'day_close': 4,  # 4 is Friday
                'FX': ['UK100', 'AUD/CAD', 'AUD/CHF', 'AUD/JPY', 'AUD/NZD', 'AUD/USD',
                       'CAD/CHF', 'CAD/JPY', 'CHF/JPY', 'EUR/AUD', 'EUR/CAD',
                       'EUR/CHF', 'EUR/GBP', 'EUR/JPY', 'EUR/NOK', 'EUR/NZD',
                       'EUR/SEK', 'EUR/TRY', 'EUR/USD', 'GBP/AUD', 'GBP/CAD',
                       'GBP/CHF', 'GBP/JPY', 'GBP/NZD', 'GBP/USD', 'NZD/CAD',
                       'NZD/CHF', 'NZD/JPY', 'NZD/USD', 'TRY/JPY', 'USD/CAD',
                       'USD/CHF', 'USD/CNH', 'USD/HKD', 'USD/JPY', 'USD/MXN',
                       'USD/NOK', 'USD/SEK', 'USD/ZAR', 'XAG/USD', 'XAU/USD',
                       'USD/ILS', 'BTC/USD', 'BCH/USD', 'ETH/USD', 'LTC/USD',
                       'JPN225', 'NAS100', 'NGAS', 'SPX500', 'US30', 'VOLX',
                       'US2000', 'AUS200', 'UKOil', 'USOil', 'USOilSpot', 'UKOilSpot',
                       'EMBasket', 'USDOLLAR', 'JPYBasket', 'CryptoMajor']},

            2: {
                'hour_open': 6,  # opening time in UTC (for midnight put 24)
                'hour_close': 20,  # closing time in UTC
                'FX': ['EUSTX50', 'FRA40']},

            3: {
                'hour_open': 6,  # opening time in UTC (for midnight put 24)
                'hour_close': 18,  # closing time in UTC
                'FX': ['ESP35', 'Bund']},

            4: {
                'hour_open': 1,  # opening time in UTC (for midnight put 24)
                'hour_close': 19,  # closing time in UTC
                'FX': ['GER30', 'HKG33', 'CHN50', 'UK100']},

            5: {
                'hour_open': 0,  # opening time in UTC (for midnight put 24)
                'hour_close': 18,  # closing time in UTC
                'FX': ['SOYF', 'WHEATF', 'CORNF']},

            6: {
                'hour_open': 14,  # opening time in UTC (for midnight put 24)
                'hour_close': 20,  # closing time in UTC
                'FX': ['ESPORTS', 'BIOTECH', 'FAANG',
                       'CHN.TECH', 'CHN.ECOMM',
                       'AIRLINES', 'CASINOS',
                       'TRAVEL', 'US.ECOMM',
                       'US.BANKS', 'US.AUTO',
                       'WFH', 'URANIUM']},

        },
}

class Position:
    def __init__(self, entry_price, position_size):
        self.entry_price = entry_price
        self.current_price = None
        self.profit_loss = 0
        self.exit_time = None
        self.position_size = position_size

    def update_price(self, price):
        self.current_price = price
        self.calculate_profit_loss()

    def calculate_profit_loss(self):
        if self.current_price is None:
            return

        price_change = self.current_price - self.entry_price
        self.profit_loss = price_change * self.position_size

    def exit(self, price):
        self.exit_time = price.timestamp
        self.update_price(price)

    def close_at_market(self, price):
        self.exit(price)
        return self.profit_loss

    def is_winner(self):
        return self.profit_loss > 0

class TradingStrategy:
    def __init__(self, entry_price_threshold, exit_price_threshold):
        self.entry_price_threshold = entry_price_threshold
        self.exit_price_threshold = exit_price_threshold

    def should_enter(self, data_point):
        """Checks for a buy signal based on the current price."""
        if data_point.price > self.entry_price_threshold:
            return True
        else:
            return False

    def should_exit(self, data_point):
        """Checks for a sell signal based on the current price."""
        if data_point.price < self.exit_price_threshold:
            return True
        else:
            return False

    def enter_long(self, data_point):
        """Simulates entering a long position."""
        position = Position(data_point.price, self.position_size)
        return position

    def exit(self, data_point):
        """Simulates exiting the current position."""
        if self.position:
            position_profit = self.position.close_at_market(data_point)
            self.position = None
            return position_profit
        else:
            return 0

def should_open_buy_trade(df):
    return (
        df.iloc[-3:-2]['ci'].mean() < 39 and
        df.iloc[-3:-2]['rsi'].mean() < 31 and
        df.iloc[-2]['tenkan_avg'] < df.iloc[-2]['kijun_avg'] and
        df.iloc[-2]['signal'] < df.iloc[-2]['macd'] and
        df.iloc[-2]['macd'] > df.iloc[-3]['macd']
    )

def should_open_sell_trade(df):
    return (
        df.iloc[-3:-2]['ci'].mean() < 39 and
        df.iloc[-3:-2]['rsi'].mean() > 69 and
        df.iloc[-2]['tenkan_avg'] > df.iloc[-2]['kijun_avg'] and
        df.iloc[-2]['signal'] > df.iloc[-2]['macd'] and
        df.iloc[-2]['macd'] < df.iloc[-3]['macd']
    )

def open_trade(df, fx, tick, trading_settings_provider, dj):
    def set_amount(lots, dj):
        account = Common.get_account(fx, Dict['FXCM']['str_account'])
        base_unit_size = trading_settings_provider.get_base_unit_size(tick, account)
        amount = base_unit_size * Dict['amount']
        return amount

    open_rev_index = 1
    box_def = False
    high_box = 0
    low_box = 0
    type_signal = 'No'
    tp = 0
    sl = 0
    index_peak = 0
    df = analysis(df, open_rev_index, tick)

    if should_open_buy_trade(df):
        min_entry = round((max(df.iloc[-27:-2]['kijun_avg']) - min(df.iloc[-27:-2]['AskLow'])) / (
            abs(df.iloc[-2]['BidClose'] - df.iloc[-2]['AskClose'])), 2)
        if min_entry >= 2:
            try:
                amount = (set_amount(Dict['amount'], dj))
                type_signal = ' BUY Amount:' + str(amount) + ' Bid/Ask:' + str(min_entry)
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
    elif should_open_sell_trade(df):

        min_entry = round((max(df.iloc[-27:-2]['AskHigh']) - min(df.iloc[-27:-2]['kijun_avg'])) / (
            abs(df.iloc[-2]['BidClose'] - df.iloc[-2]['AskClose'])), 2)
        if min_entry >= 2:
            try:
                amount = (set_amount(Dict['amount'], dj))
                type_signal = ' Sell Amount: ' + str(amount) + ' Bid/Ask: ' + str(min_entry)
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

    return df, type_signal, open_rev_index, box_def, high_box, low_box, tp, sl, index_peak

def close_trade(df, fx, tick, dj, l0):
    try:
        open_rev_index = \
            [len(df) - df.index[df['Date'].dt.strftime("%m%d%Y%H") == dj.loc[0, 'tick_time'].strftime("%m%d%Y%H")]][0][
                0]
    except:
        open_rev_index = 1
    type_signal = 'No'
    open_price = dj.loc[0, 'tick_open_price']
    price = dj.loc[0, 'tick_price']
    box_def = False
    high_box = 0
    low_box = 0
    index_peak = None
    df = analysis(df, open_rev_index, tick)
    tp = dj.loc[0, 'tick_limit']
    sl = dj.loc[0, 'tick_stop']
    offer = Common.get_offer(fx, tick)
    buy = fxcorepy.Constants.BUY
    sell = fxcorepy.Constants.SELL
    buy_sell = sell if dj.loc[0, 'tick_type'] == buy else buy
    candle_2 = (df.iloc[-2]['AskClose'] - df.iloc[-2]['AskOpen']) / (df.iloc[-2]['AskHigh'] - df.iloc[-2]['AskLow'])
    window_of_interest = 27
    margin = abs(0.1 * (np.nanmax(df.iloc[-window_of_interest:-2]['AskHigh']) - np.nanmin(
        df.iloc[-window_of_interest:-2]['AskLow'])))

    for i in range(-open_rev_index, -len(df) + 1, -1):
        if df.iloc[i]['rsi'] < 31:
            if (i == -2) and \
                    df.iloc[i]['rsi'] <= df.iloc[i - 1]['rsi'] and \
                    df.iloc[i]['rsi'] <= df.iloc[i - 2]['rsi'] and \
                    df.iloc[i]['rsi'] <= df.iloc[i + 1]['rsi']:
                index_peak = i
                break
            elif (i < -2) and \
                    df.iloc[i]['rsi'] <= df.iloc[i - 1]['rsi'] and \
                    df.iloc[i]['rsi'] <= df.iloc[i - 2]['rsi'] and \
                    df.iloc[i]['rsi'] <= df.iloc[i + 1]['rsi'] and \
                    df.iloc[i]['rsi'] <= df.iloc[i + 2]['rsi']:
                index_peak = i
                break
        elif df.iloc[i]['rsi'] > 69:
            if (i == -2) and \
                    df.iloc[i]['rsi'] >= df.iloc[i - 1]['rsi'] and \
                    df.iloc[i]['rsi'] >= df.iloc[i - 2]['rsi'] and \
                    df.iloc[i]['rsi'] >= df.iloc[i + 1]['rsi']:
                index_peak = i
                break
            elif (i < -2) and \
                    df.iloc[i]['rsi'] >= df.iloc[i - 1]['rsi'] and \
                    df.iloc[i]['rsi'] >= df.iloc[i - 2]['rsi'] and \
                    df.iloc[i]['rsi'] >= df.iloc[i + 1]['rsi'] and \
                    df.iloc[i]['rsi'] >= df.iloc[i + 2]['rsi']:
                index_peak = i
                break

    if open_rev_index < 1:
        print('open_rev_index too small')
    else:
        # BUY CONDIDITIONS
        if dj.loc[0, 'tick_type'] == 'B':
            current_ratio = (price - open_price) / (open_price - df.iloc[-open_rev_index:-2]['AskLow'].min())

            if df.iloc[-2]['AskClose'] < df.iloc[-2]['tenkan_avg'] and candle_2 < -0.25 \
                    and current_ratio > 0 and df.iloc[-open_rev_index:-2][df['rsi'] > 65].size > 0 and df.iloc[-2][
                'tenkan_avg'] < df.iloc[-3]['tenkan_avg'] and \
                    ((abs(df.iloc[-2]['macd']) < abs(df.iloc[-2]['signal'])) or (
                            df.iloc[-4:-2]['macd'].mean() < df.iloc[-6:-4]['macd'].mean())):
                try:
                    type_signal = ' Buy : Close for end of cycle' + str(current_ratio)
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
            if df.iloc[-2]['tenkan_avg'] < df.iloc[-2]['kijun_avg'] and current_ratio > 0 and \
                    df.iloc[-2]['AskLow'] < df.iloc[-2]['tenkan_avg'] and candle_2 < -0.25:
                try:
                    type_signal = ' Buy : Close for tenkan over kijun' + str(current_ratio)
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
            if (df.iloc[-2]['kijun_avg'] - margin) > open_price and (df.iloc[-2]['kijun_avg'] - margin) > df.iloc[-2][
                'AskClose'] and current_ratio > 0:
                try:
                    sl = df.iloc[-2]['kijun_avg'] - margin
                    type_signal = ' Buy : Adjust for being safe ' + str(current_ratio)
                    request = fx.create_order_request(
                        order_type=fxcorepy.Constants.Orders.LIMIT,
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
            if df.iloc[-2]['tenkan_avg'] < df.iloc[-2]['kijun_avg'] \
                    and df.iloc[-2]['AskClose'] < df.iloc[-2]['tenkan_avg'] \
                    and df.iloc[-2]['signal'] > df.iloc[-2]['macd'] \
                    and df.iloc[-3:-2]['macd'].mean() < df.iloc[-4:-3]['macd'].mean() \
                    and df.iloc[-3:-2]['rsi'].mean() < df.iloc[-4:-3]['rsi'].mean() \
                    and df.iloc[-3:-2]['tenkan_avg'].mean() < df.iloc[-4:-3]['tenkan_avg'].mean() \
                    and df.iloc[-3:-2]['kijun_avg'].mean() < df.iloc[-4:-3]['kijun_avg'].mean():
                try:
                    type_signal = ' Buy : Close for bad ' + str(current_ratio)
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

        # if was sell
        if dj.loc[0, 'tick_type'] == 'S':
            current_ratio = (open_price - price) / (df.iloc[-open_rev_index:-2]['AskHigh'].max() - open_price)

            if df.iloc[-2]['AskClose'] > df.iloc[-2]['tenkan_avg'] and candle_2 > 0.25 \
                    and current_ratio > 0 and df.iloc[-open_rev_index:-2][df['rsi'] < 35].size > 0 \
                    and df.iloc[-2]['tenkan_avg'] > df.iloc[-3]['tenkan_avg'] \
                    and ((abs(df.iloc[-2]['macd']) < abs(df.iloc[-2]['signal'])) or (
                    df.iloc[-4:-2]['macd'].mean() > df.iloc[-6:-4]['macd'].mean())):
                try:
                    type_signal = ' Sell : Close for end of cycle' + str(current_ratio)
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
            if df.iloc[-2]['tenkan_avg'] > df.iloc[-2]['kijun_avg'] and current_ratio > 0 and \
                    df.iloc[-2]['AskHigh'] > df.iloc[-2]['tenkan_avg'] and candle_2 > 0.25:
                try:
                    type_signal = ' Sell : Close for tenkan over kijun' + str(current_ratio)
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
            if (df.iloc[-2]['kijun_avg'] + margin) < open_price and (df.iloc[-2]['kijun_avg'] + margin) < df.iloc[-2][
                'AskClose'] and current_ratio > 0:
                try:
                    sl = (df.iloc[-2]['kijun_avg'] + margin)
                    type_signal = ' Sell : Adjust for being safe ' + str(current_ratio)
                    request = fx.create_order_request(
                        order_type=fxcorepy.Constants.Orders.LIMIT,
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
            if df.iloc[-2]['tenkan_avg'] > df.iloc[-2]['kijun_avg'] \
                    and df.iloc[-2]['AskClose'] > df.iloc[-2]['tenkan_avg'] \
                    and df.iloc[-2]['signal'] < df.iloc[-2]['macd'] \
                    and df.iloc[-3:-2]['macd'].mean() > df.iloc[-4:-3]['macd'].mean() \
                    and df.iloc[-3:-2]['rsi'].mean() > df.iloc[-4:-3]['rsi'].mean() \
                    and df.iloc[-3:-2]['tenkan_avg'].mean() > df.iloc[-4:-3]['tenkan_avg'].mean() \
                    and df.iloc[-3:-2]['kijun_avg'].mean() > df.iloc[-4:-3]['kijun_avg'].mean():
                try:
                    type_signal = ' Sell : Close for bad ' + str(current_ratio)
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

    return df, type_signal, open_rev_index, box_def, high_box, low_box, tp, sl, index_peak

def backtest_strategy(historical_data, trading_strategy):
    total_profit = 0
    wins = 0
    losses = 0
    for data_point in historical_data:
        # Check for entry and exit signals
        if trading_strategy.should_enter(data_point):
            # Simulate entering a long position
            position = trading_strategy.enter_long(data_point)
            if position.is_winner():
                wins += 1
                total_profit += position.profit
            else:
                losses += 1
        elif trading_strategy.should_exit(data_point):
            # Simulate exiting a position
            position = trading_strategy.exit(data_point)
        elif position and data_point.is_after(position.exit_time):
            # Simulate closing a position at expiry
            position = trading_strategy.close_at_expiry(data_point)

    average_profit = total_profit / (wins + losses)
    win_rate = wins / (wins + losses)
    return total_profit, average_profit, win_rate

def indicators(df):
    def ichimoku(df):
        # Tenkan Sen
        tenkan_max = df['AskHigh'].rolling(window=9).max()
        tenkan_min = df['AskLow'].rolling(window=9).min()
        df['tenkan_avg'] = (tenkan_max + tenkan_min) / 2

        # Kijun Sen
        kijun_max = df['AskHigh'].rolling(window=26).max()
        kijun_min = df['AskLow'].rolling(window=26).min()
        df['kijun_avg'] = (kijun_max + kijun_min) / 2

        # Senkou Span A
        # (Kijun + Tenkan) / 2 Shifted ahead by 26 periods
        df['senkou_a'] = ((df['kijun_avg'] + df['tenkan_avg']) / 2).shift(26)

        # Senkou Span B
        # 52 period High + Low / 2
        senkou_b_max = df['AskHigh'].rolling(window=52).max()
        senkou_b_min = df['AskLow'].rolling(window=52).min()
        df['senkou_b'] = ((senkou_b_max + senkou_b_min) / 2).shift(26)

        # Chikou Span
        # Current Close shifted -26
        df['chikou'] = (df['AskClose']).shift(-26)
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
        df['Delta'] = np.array(macd - exp3)
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

    df = ichimoku(df)
    df = macd(df)
    df['rsi'] = rsi(df, 14, True)
    df['ci'] = get_ci(df['AskHigh'], df['AskLow'], df['AskClose'], 28)

    return (df)

def analysis(df, ind, tick):
    def chikou_signal(df):

        # Check the Chikou
        df['chikou_signal'] = np.zeros(len(df['AskClose']))
        end_chikou_signal = 30
        if len(df['AskClose']) <= 27:
            end_chikou_signal = len(df['AskClose'])
        for p in range(27, end_chikou_signal):
            # Check if chikou more than anything
            if df.iloc[-p]['chikou'] > df.iloc[-p]['AskClose'].max() \
                    and df.iloc[-p]['chikou'] > df.iloc[-p]['tenkan_avg'].max() \
                    and df.iloc[-p]['chikou'] > df.iloc[-p]['kijun_avg'].max():
                df.loc[len(df) - p, 'chikou_signal'] = 1
            # Check if chikou is less than anything
            elif df.iloc[-p]['chikou'] < df.iloc[-p]['AskClose'].min() \
                    and df.iloc[-p]['chikou'] < df.iloc[-p]['tenkan_avg'].min() \
                    and df.iloc[-p]['chikou'] < df.iloc[-p]['kijun_avg'].min():
                df.loc[len(df) - p, 'chikou_signal'] = -1
            else:
                df.loc[len(df) - p, 'chikou_signal'] = 2

        return df

    def trend_channels(df, backcandles, wind, candleid, brange, ask_plot):
        df_low = df['AskLow']
        df_high = df['AskHigh']
        optbackcandles = backcandles
        sldiff = 1000
        sldist = 10000
        for r1 in range(backcandles - brange, backcandles + brange):
            maxim = np.array([])
            minim = np.array([])
            xxmin = np.array([])
            xxmax = np.array([])
            for i in range(candleid - backcandles, candleid + 1, wind):
                if df_low.iloc[i:i + wind].size != 0:
                    minim = np.append(minim, df_low.iloc[i:i + wind].min())
                    xxmin = np.append(xxmin, df_low.iloc[i:i + wind].idxmin())  # ;;;;;;;;;;;
            for i in range(candleid - backcandles, candleid + 1, wind):
                if df_high.iloc[i:i + wind].size != 0:
                    maxim = np.append(maxim, df_high.iloc[i:i + wind].max())
                    xxmax = np.append(xxmax, df_high.iloc[i:i + wind].idxmax())
            slmin, intercmin = np.polyfit(xxmin, minim, 1)
            slmax, intercmax = np.polyfit(xxmax, maxim, 1)

            dist = (slmax * candleid + intercmax) - (slmin * candleid + intercmin)
            if (dist < sldist) and (abs(slmin - slmax) < sldiff):
                sldiff = abs(slmin - slmax)
                sldist = dist
                optbackcandles = r1
                slminopt = slmin
                slmaxopt = slmax
                intercminopt = intercmin
                intercmaxopt = intercmax
                maximopt = maxim.copy()
                minimopt = minim.copy()
                xxminopt = xxmin.copy()
                xxmaxopt = xxmax.copy()
        adjintercmin = (df_low.iloc[xxminopt] - slminopt * xxminopt).min()
        adjintercmax = (df_high.iloc[xxmaxopt] - slmaxopt * xxmaxopt).max()

        # at the current index where is the channel
        xminopt = np.arange(int(xxminopt[0]), int(xxminopt[-1]), 1)
        xmaxopt = np.arange(int(xxmaxopt[0]), int(xxmaxopt[-1]), 1)
        ychannelmin = slminopt * xminopt + adjintercmin
        ychannelmax = slmaxopt * xmaxopt + adjintercmax

        df_chanmax = pd.DataFrame()
        df_chanmax['ychannelmax'] = ychannelmax

        df_chanmin = pd.DataFrame()
        df_chanmin['ychannelmin'] = ychannelmin

        df_opt = pd.DataFrame()
        df_opt['xxmaxopt'] = xxmaxopt
        df_opt['xxminopt'] = xxminopt

        df_single = pd.DataFrame(columns=['slminopt', 'adjintercmin', 'slmaxopt', 'adjintercmax'])
        df_single.loc[0] = [slminopt, adjintercmin, slmaxopt, adjintercmax]

        df = pd.concat([df, df_chanmax, df_chanmin, df_opt, df_single], axis=1)

        return df

    def find_last_peaks_old(df, ind):
        n = len(df)
        df['peaks'] = np.nan
        df['peaks_macd'] = np.nan
        df['slope'] = np.nan
        df['slope_macd'] = np.nan
        # peaks definition
        for i in range(-ind - 1, -n + 1, -1):
            # peaks macd
            if abs(df.iloc[i]['macd']) >= abs(df.iloc[i]['signal']):
                if (i == -2) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i - 1]['macd']) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i - 2]['macd']) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i + 1]['macd']):
                    df.loc[n + i, 'peaks_macd'] = df.iloc[i]['macd']
                    df.loc[n + i, 'peaks'] = df.iloc[i]['AskClose']
                elif (i < -2) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i - 1]['macd']) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i + 1]['macd']) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i - 2]['macd']) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i + 2]['macd']):
                    df.loc[n + i, 'peaks_macd'] = df.iloc[i]['macd']
                    df.loc[n + i, 'peaks'] = df.iloc[i]['AskClose']
        # slope definition & remove all the nans
        if df['peaks_macd'].dropna().size > 2 and df['peaks'].dropna().size > 2 and \
                df['peaks'].dropna().iloc[-1] != df['peaks'].dropna().iloc[-2] and \
                df['peaks_macd'].dropna().iloc[-1] != df['peaks_macd'].dropna().iloc[-2]:
            temp = df['peaks'].dropna().reset_index()
            tempm = df['peaks_macd'].dropna().reset_index()
            df.loc[2, 'slope_macd'] = tempm['peaks_macd'].iloc[-2]
            df.loc[4, 'slope_macd'] = tempm['index'].iloc[-2]
            df.loc[2, 'slope'] = temp['peaks'].iloc[-2]
            df.loc[4, 'slope'] = temp['index'].iloc[-2]
            df.loc[1, 'slope_macd'] = tempm['peaks_macd'].iloc[-1]
            df.loc[3, 'slope_macd'] = tempm['index'].iloc[-1]
            df.loc[1, 'slope'] = temp['peaks'].iloc[-1]
            df.loc[3, 'slope'] = temp['index'].iloc[-1]
            df.loc[0, 'slope'] = (df.loc[1, 'slope'] - df.loc[2, 'slope']) / (df.loc[3, 'slope'] - df.loc[4, 'slope'])
            df.loc[0, 'slope_macd'] = (df.loc[1, 'slope_macd'] - df.loc[2, 'slope_macd']) / (
                    df.loc[3, 'slope_macd'] - df.loc[4, 'slope_macd'])
        return df

    def find_last_peaks(df, ind):
        n = len(df)
        df['peaks'] = np.nan
        df['peaks_macd'] = np.nan
        df['slope'] = np.nan
        df['slope_macd'] = np.nan
        sign_first_peak = 0
        for i in range(-ind - 1, -n + 1, -1):
            if abs(df.iloc[i]['macd']) >= abs(df.iloc[i]['signal']):
                if (i == -2) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i - 1]['macd']) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i - 2]['macd']) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i + 1]['macd']):
                    df.loc[n + i, 'peaks_macd'] = df.iloc[i]['macd']
                    df.loc[n + i, 'peaks'] = df.iloc[i]['AskClose']
                    sign_first_peak = np.sign(df.iloc[i]['macd'])
                elif (i < -2) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i - 1]['macd']) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i + 1]['macd']) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i - 2]['macd']) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i + 2]['macd']):
                    if sign_first_peak == 0:
                        df.loc[n + i, 'peaks_macd'] = df.iloc[i]['macd']
                        df.loc[n + i, 'peaks'] = df.iloc[i]['AskClose']
                        sign_first_peak = np.sign(df.iloc[i]['macd'])
                    elif sign_first_peak == np.sign(df.iloc[i]['macd']):
                        df.loc[n + i, 'peaks_macd'] = df.iloc[i]['macd']
                        df.loc[n + i, 'peaks'] = df.iloc[i]['AskClose']
        # slope definition & remove all the nans
        if df['peaks_macd'].dropna().size > 2 and df['peaks'].dropna().size > 2 and \
                df['peaks'].dropna().iloc[-1] != df['peaks'].dropna().iloc[-2] and \
                df['peaks_macd'].dropna().iloc[-1] != df['peaks_macd'].dropna().iloc[-2]:
            temp = df['peaks'].dropna().reset_index()
            tempm = df['peaks_macd'].dropna().reset_index()
            df.loc[2, 'slope_macd'] = tempm['peaks_macd'].iloc[-2]
            df.loc[4, 'slope_macd'] = tempm['index'].iloc[-2]
            df.loc[2, 'slope'] = temp['peaks'].iloc[-2]
            df.loc[4, 'slope'] = temp['index'].iloc[-2]
            df.loc[1, 'slope_macd'] = tempm['peaks_macd'].iloc[-1]
            df.loc[3, 'slope_macd'] = tempm['index'].iloc[-1]
            df.loc[1, 'slope'] = temp['peaks'].iloc[-1]
            df.loc[3, 'slope'] = temp['index'].iloc[-1]
            df.loc[0, 'slope'] = (df.loc[1, 'slope'] - df.loc[2, 'slope']) / (df.loc[3, 'slope'] - df.loc[4, 'slope'])
            df.loc[0, 'slope_macd'] = (df.loc[1, 'slope_macd'] - df.loc[2, 'slope_macd']) / (
                    df.loc[3, 'slope_macd'] - df.loc[4, 'slope_macd'])
        return df

    def find_limit(df):
        df.iloc[-2]['AskClose']

    def mean_reversion(data, tick):
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

        # Calculate the rolling mean and standard deviation
        window = 20
        data['RollingMean'] = data['AskClose'].rolling(window=window).mean()
        data['RollingStd'] = data['AskClose'].rolling(window=window).std()
        # Calculate z-scores
        data['ZScore'] = (data['AskClose'] - data['RollingMean']) / data['RollingStd']
        # Define entry and exit thresholds
        entry_threshold = 1.0
        exit_threshold = 0.0
        # Initialize positions and signals
        data['Position'] = 0  # 1 for long, -1 for short, 0 for no position
        data['Signal'] = 0  # 1 for buy signal, -1 for sell signal, 0 for no signal

        # Generate trading signals
        for i in range(window, len(data)):
            if data.iloc[i]['ZScore'] > entry_threshold and data.iloc[i - 1]['ZScore'] <= entry_threshold:
                data.loc[i, 'Signal'] = -1  # Short position
            elif data.iloc[i]['ZScore'] < -entry_threshold and data.iloc[i - 1]['ZScore'] >= -entry_threshold:
                data.loc[i, 'Signal'] = 1  # Long position

        # Apply signals to positions
        for i in range(window, len(data)):
            if data.iloc[i]['Signal'] == 1:
                data.loc[i, 'Position'] = 1  # Long position
            elif data.iloc[i]['Signal'] == -1:
                data.loc[i, 'Position'] = -1  # Short position

        # Calculate strategy returns
        data['Returns'] = data['Position'].shift() * data['AskClose'].pct_change()

        # Calculate cumulative returns
        data['CumulativeReturns'] = (1 + data['Returns']).cumprod()

        # Plotting
        fig = plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['CumulativeReturns'])
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.title('Mean Reversion Strategy')
        fig.savefig('mean_reversion.png')
        plt.close()
        return data

    df = chikou_signal(df)
    # trend_channels defined by how many backcandles we are going RWD, so let's take a 3 months=90days,
    # then by the window of check, let's take 5 days, where I am starting today -4 (yesterday) and what optimization
    # backcandles i ma reday to follow 1 week so 5 days
    df = trend_channels(df, 27 * 3, 3, len(df) - 4 - ind, 5, False)
    df = find_last_peaks(df, ind)
    df = mean_reversion(df, tick)
    return df

def box(df, index):
    low_box = -1
    high_box = -1
    box_def = False

    # in the last 29 periods kijun has been flat take the last flat
    for m in range(len(df) - index, len(df) - 28 - index, -1):
        if df.iloc[m]['kijun_avg'] == df.iloc[m - 1]['kijun_avg'] and \
                df.iloc[m]['kijun_avg'] == df.iloc[m - 2]['kijun_avg'] and \
                df.iloc[m]['kijun_avg'] == df.iloc[m - 3]['kijun_avg']:
            box_def = True
            break

    # if box has been found
    if box_def == True:
        # Top limit
        if df['AskHigh'][-28 - index:-2 - index].max() >= df.iloc[-0 - index]['kijun_avg']:
            top_limit = df['AskHigh'][-28 - index:-2 - index].max() - df.iloc[-0 - index]['kijun_avg']
        else:
            top_limit = df.iloc[-0 - index]['kijun_avg'] - df['AskHigh'][-28 - index:-2 - index].max()
        # Lower limit
        if df['AskLow'][-28 - index:-2 - index].min() <= df.iloc[-0 - index]['kijun_avg']:
            low_limit = df.iloc[-0 - index]['kijun_avg'] - df['AskLow'][-28 - index:-2 - index].min()
        else:
            low_limit = df['AskLow'][-28 - index:-2 - index].min() - df.iloc[-0 - index]['kijun_avg']
        max_limit = max(top_limit, low_limit)
        low_box = df.iloc[-0 - index]['kijun_avg'] - max_limit
        high_box = df.iloc[-0 - index]['kijun_avg'] + max_limit
        # Check that kijun is in-between
        if ((low_box + high_box) * 0.45) < df.iloc[m]['kijun_avg'] < ((low_box + high_box) * 0.55):
            box_def == True
        else:
            box_def == False
    return low_box, high_box, box_def

def df_plot(df, tick, type_signal, index, box_def, high_box, low_box, tp, sl, index_peak):
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
        min_x = 27 * 10
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
        if tp != 0:
            ax1.axhline(y=float(tp), color='blue', linewidth=1, linestyle='-.')
        if sl != 0:
            ax1.axhline(y=float(sl), color='red', linewidth=1, linestyle='-.')
        ax1.axhline(y=float(df.iloc[-index]['AskClose']), color='black', linewidth=1, linestyle='-.')
        ax1.plot(df.iloc[-index]['index'], df.iloc[-index]['AskClose'], 'black', marker='s')
        ax1.axvline(x=df.iloc[-index]['index'], color='black', linewidth=1, linestyle='-.')
        ax1.axvline(x=df.iloc[index_peak]['index'], color='red', linewidth=1, linestyle='-.')
        ax1.plot([df.loc[3, 'slope'], df.loc[4, 'slope']], [df.loc[1, 'slope'], df.loc[2, 'slope']], linewidth=2,
                 color='yellow', marker='s')
        ax1.plot([df['index'][int(np.array(df['xxminopt'].dropna())[0])],
                  df['index'][int(np.array(df['xxminopt'].dropna())[-1])]],
                 [df['slminopt'].dropna() * int(np.array(df['xxminopt'].dropna())[0]) + df['adjintercmin'].dropna(),
                  df['slminopt'].dropna() * int(np.array(df['xxminopt'].dropna())[-1]) + df['adjintercmin'].dropna()],
                 linewidth=2, color='green')
        ax1.plot([df['index'][int(np.array(df['xxmaxopt'].dropna())[0])],
                  df['index'][int(np.array(df['xxmaxopt'].dropna())[-1])]],
                 [df['slmaxopt'].dropna() * int(np.array(df['xxmaxopt'].dropna())[0]) + df['adjintercmax'].dropna(),
                  df['slmaxopt'].dropna() * int(np.array(df['xxmaxopt'].dropna())[-1]) + df['adjintercmax'].dropna()],
                 linewidth=2, color='red')
        ax1.fill_between(df.index[-min_x:], df['senkou_a'][-min_x:], df['senkou_b'][-min_x:],
                         where=df['senkou_a'][-min_x:] >= df['senkou_b'][-min_x:],
                         color='lightgreen')
        ax1.fill_between(df.index[-min_x:], df['senkou_a'][-min_x:], df['senkou_b'][-min_x:],
                         where=df['senkou_a'][-min_x:] < df['senkou_b'][-min_x:],
                         color='lightcoral')
        quotes = [tuple(x) for x in df[['index', 'AskOpen', 'AskHigh', 'AskLow', 'AskClose']].values]
        candlestick_ohlc(ax1, quotes, width=0.2, colorup='g', colordown='r')

        # Range_box
        if box_def == True:
            xmin = df['AskLow'][-27 - index:-1 - index].idxmin()
            xmax = df['AskHigh'][-27 - index:-1 - index].idxmax()
            ax1.add_patch(patches.Rectangle((xmin, low_box), (xmax - xmin), (high_box - low_box), edgecolor='orange',
                                            facecolor='none', linewidth=1))
            ax1.hlines(y=float(0.4 * (high_box - low_box) + low_box), xmin=xmin, xmax=xmax, color='orange', linewidth=1,
                       linestyle='-.')
            ax1.hlines(y=float(0.6 * (high_box - low_box) + low_box), xmin=xmin, xmax=xmax, color='orange', linewidth=1,
                       linestyle='-.')
            ax1.hlines(y=float(0.9 * (high_box - low_box) + low_box), xmin=xmin, xmax=xmax, color='orange', linewidth=1,
                       linestyle='-.')
            ax1.hlines(y=float(0.1 * (high_box - low_box) + low_box), xmin=xmin, xmax=xmax, color='orange', linewidth=1,
                       linestyle='-.')
        ax1.grid()
        low_limit = np.nanmin(df['AskLow'][-min_x:])
        high_limit = np.nanmax(df['AskHigh'][-min_x:])
        ax1.set_ylim(low_limit - 0.1 * (high_limit - low_limit), high_limit + 0.1 * (high_limit - low_limit))
        ax1.set_xlim(np.nanmin(df['index'][-min_x:]), np.nanmax(df['index'][-min_x:]))
        ax1.set(xlabel=None)

        ###AX2
        ax2.bar(df.index[-min_x:], df['macd'][-min_x:], color='grey')
        ax2.plot(df.index[-min_x:], df['signal'][-min_x:], color='red')
        ax2.plot([df.loc[3, 'slope_macd'], df.loc[4, 'slope_macd']], [df.loc[1, 'slope_macd'], df.loc[2, 'slope_macd']],
                 linewidth=2,
                 color='yellow', marker='s')
        ax2.axvline(x=df.iloc[-index]['index'], color='black', linewidth=1, linestyle='-.')
        ax2.axvline(x=df.iloc[index_peak]['index'], color='red', linewidth=1, linestyle='-.')
        ax2.set_ylim(np.nanmin(df['macd'][-min_x:]), np.nanmax(df['macd'][-min_x:]))
        ax2.grid()
        ax2.set(xlabel=None)

        ###AX3
        ax3.bar(df.index[-min_x:], df['Delta'][-min_x:], color='black')
        ax3.axvline(x=df.iloc[-index]['index'], color='black', linewidth=1, linestyle='-.')
        ax3.axvline(x=df.iloc[index_peak]['index'], color='red', linewidth=1, linestyle='-.')
        ax3_0 = ax3.twinx()
        ax3_0.bar(df.index[-min_x:], df['Signal'][-min_x:], color='red')
        ax3.set_ylim(np.nanmin(df['Delta'][-min_x:]), np.nanmax(df['Delta'][-min_x:]))
        ax3.grid()
        ax3.set(xlabel=None)

        ###AX4
        ax4.plot(df.index[-min_x:], df['rsi'][-min_x:], color='black')
        ax4.axhline(y=30, color='grey', linestyle='-.')
        ax4.axhline(y=70, color='grey', linestyle='-.')
        ax4.plot(df.index[-min_x:], df['ci'][-min_x:], color='orange')
        ax4.axhline(y=38.2, color='yellow', linestyle='-.')
        ax4.axhline(y=61.8, color='yellow', linestyle='-.')
        ax4.axvline(x=df.iloc[-index]['index'], color='black', linewidth=1, linestyle='-.')
        ax4.set_ylim((0, 100))
        ax4.axhline(y=float(df.iloc[index_peak]['rsi']), color='red', linewidth=1, linestyle='-.')
        ax4.plot(df.iloc[index_peak]['index'], df.iloc[index_peak]['rsi'], 'red', marker='s')
        ax4.axvline(x=df.iloc[index_peak]['index'], color='red', linewidth=1, linestyle='-.')
        ax4.grid()

        # plt.show()
        fig.savefig('filename.png')
        try:
            sendemail(attach='filename.png', subject_mail=str(tick), body_mail=str(tick) + " " + str(type_signal))
        except Exception as e:
            print("issue with mails for " + tick)
            print("Exception: " + str(e))
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
    print(str(datetime.now().strftime("%H:%M:%S")))
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
                if trading_settings_provider.get_market_status(FX[l1]) == fxcorepy.O2GMarketStatus.MARKET_STATUS_OPEN:
                    tick = FX[l1]
                    print(FX[l1])
                    # H1
                    df = pd.DataFrame(fx.get_history(FX[l1], 'H1', Dict['indicators']['sd'], Dict['indicators']['ed']))
                    if len(df) < 7 * 5 * 3:
                        df = pd.DataFrame(
                            fx.get_history(FX[l1], 'm15', datetime.now() - relativedelta(weeks=6),
                                           Dict['indicators']['ed']))
                    # If there is history data
                    # Add all the indicators needed

                    window_of_interest = 27
                    margin = abs(0.1 * (np.nanmax(df.iloc[-window_of_interest:-2]['AskHigh']) - np.nanmin(
                        df.iloc[-window_of_interest:-2]['AskLow'])))

                    df = indicators(df)
                    # back-test
                    #total_profit, average_profit, win_rate = backtest_strategy(df, TradingStrategy)
                    #print("Total profit:", total_profit)
                    #print("Average profit per trade:", average_profit)
                    #print("Win rate:", win_rate)
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
                                open_trade(df, fx, FX[l1], trading_settings_provider, dj)
                            df_plot(df, tick, type_signal, index, box_def, high_box, low_box, tp, sl, index_peak)
                    # if status is open then check if to close
                    elif open_pos_status == 'Yes':
                        df, type_signal, index, box_def, high_box, low_box, tp, sl, index_peak = \
                            close_trade(df, fx, FX[l1], dj, l0)
                        df_plot(df, tick, type_signal, index, box_def, high_box, low_box, tp, sl, index_peak)


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
