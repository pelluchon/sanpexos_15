import os
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
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
import schedule
import math


FX = ['AUD/CAD', 'AUD/CHF', 'AUD/JPY', 'AUD/NZD', 'AUD/USD', 'AUS200',
      'Bund', 'CAD/CHF', 'CAD/JPY', 'CHF/JPY', 'CHN50', 'Copper',
      'ESP35', 'EUR/AUD', 'EUR/CAD', 'EUR/CHF', 'EUR/GBP', 'EUR/JPY',
      'EUR/NOK', 'EUR/NZD', 'EUR/SEK', 'EUR/TRY', 'EUR/USD', 'EUSTX50',
      'FRA40', 'GBP/AUD', 'GBP/CAD', 'GBP/CHF', 'GBP/JPY', 'GBP/NZD',
      'GBP/USD', 'GER30', 'HKG33', 'JPN225', 'NAS100', 'NGAS', 'NZD/CAD',
      'NZD/CHF', 'NZD/JPY', 'NZD/USD', 'SOYF', 'SPX500', 'TRY/JPY',
      'UK100', 'UKOil', 'US30', 'USD/CAD', 'USD/CHF', 'USD/CNH',
      'USD/HKD', 'USD/JPY', 'USD/MXN', 'USD/NOK', 'USD/SEK',
      'USD/ZAR', 'USDOLLAR', 'USOil', 'XAG/USD', 'XAU/USD', 'ZAR/JPY',
      'USD/ILS', 'VOLX', 'US2000', 'USOilSpot', 'UKOilSpot', 'WHEATF',
      'CORNF', 'EMBasket', 'JPYBasket', 'BTC/USD', 'BCH/USD', 'ETH/USD',
      'LTC/USD', 'CryptoMajor', 'ESPORTS', 'BIOTECH',
      'FAANG', 'CHN.TECH', 'CHN.ECOMM', 'USEquities', 'AIRLINES',
      'CASINOS', 'TRAVEL', 'US.ECOMM', 'US.BANKS', 'US.AUTO', 'WFH',
      'URANIUM', 'BA.us', 'BAC.us', 'BRKB.us', 'C.us', 'CRM.us',
      'DIS.us', 'F.us', 'JPM.us', 'KO.us', 'MA.us', 'MCD.us',
      'PFE.us', 'PG.us', 'SE.us', 'T.us', 'TGT.us', 'V.us', 'XOM.us',
      'AAPL.us', 'AMZN.us', 'BIDU.us', 'GOOG.us', 'INTC.us', 'MSFT.us',
      'SBUX.us', 'ACA.fr', 'AI.fr', 'ALO.fr', 'BN.fr', 'BNP.fr', 'CA.fr',
      'DG.fr', 'AIR.fr', 'ORA.fr', 'GLE.fr', 'MC.fr', 'ML.fr', 'OR.fr',
      'RNO.fr', 'SAN.fr', 'SGO.fr', 'SU.fr', 'VIE.fr', 'VIV.fr',
      'ADS.de', 'ALV.de', 'BAS.de', 'BAYN.de', 'BMW.de',
      'DB1.de', 'DBK.de', 'DPW.de', 'DTE.de', 'EOAN.de', 'IFX.de',
      'LHA.de', 'MRK.de', 'RWE.de', 'SAP.de', 'SIE.de', 'TUI1.de',
      'VOW.de', 'AV.uk', 'AZN.uk', 'BA.uk', 'BARC.uk', 'BATS.uk',
      'BP.uk', 'GSK.uk', 'HSBA.uk', 'IAG.uk', 'LGEN.uk', 'LLOY.uk',
      'RR.uk', 'STAN.uk', 'TSCO.uk', 'VOD.uk', 'BABA.us', 'DAL.us',
      'NFLX.us', 'TSLA.us', 'GLEN.uk', 'TTE.fr', 'ENGI.fr', 'VNA.de',
      'SQ.us', 'LYFT.us', 'UAL.us', 'DKNG.us', 'SHOP.us', 'BYND.us',
      'UBER.us', 'ZM.us', 'LCID.us', 'HOOD.us', 'CRWD.us', 'BEKE.us',
      'CPNG.us', 'NET.us', 'RBLX.us', 'ENR.de', 'BIDU.hk', 'COIN.us',
      'CSL.au', 'CBA.au', 'BHP.au', 'WBC.au', 'NAB.au', 'ANZ.au',
      'WOW.au', 'WES.au', 'FMG.au', 'MQG.au', 'TLS.au', 'RIO.au',
      'GMG.au', 'WPL.au', 'NCM.au', 'COL.au', 'ALL.au', 'A2M.au',
      'REA.au', 'XRO.au', 'QAN.au', 'Z1P.au', 'BT.A.uk', 'NWG.uk',
      'TW.uk', 'MRO.uk', 'MNG.uk', 'ROO.uk', 'CBK.de', 'DHER.de',
      'STM.fr', 'STLA.fr', 'FVRR.us', 'SPOT.us', 'MARA.us', 'BTBT.us',
      'BITF.us', 'WISH.us', 'RIVN.us', 'WE.us', 'JD.us', 'PDD.us',
      'TME.us', 'WB.us', 'BILI.us', 'NVDA.us', 'AMD.us', 'DADA.us',
      'PTON.us', 'TENC.hk', 'MEIT.hk', 'BYDC.hk', 'XIAO.hk', 'BABA.hk',
      'AIA.hk', 'HSBC.hk', 'WUXI.hk', 'HKEX.hk', 'GELY.hk', 'JD.hk',
      'NETE.hk', 'PING.hk', 'SMIC.hk', 'SBIO.hk', 'GALA.hk', 'KIDE.hk',
      'ALIH.hk', 'ICBC.hk', 'FLAT.hk', 'KSOF.hk', 'SMOO.hk', 'SUNN.hk',
      'BYDE.hk', 'MRNA.us', 'NIO.us', 'CCL.us', 'ABNB.us', 'DASH.us',
      'AMC.us', 'BNGO.us', 'FCEL.us', 'GME.us', 'PENN.us', 'PLTR.us',
      'PLUG.us', 'PYPL.us', 'SNAP.us', 'SNOW.us', 'SPCE.us', 'XPEV.us',
      'SONY.us']
Fx2 = ['CANNABIS']
#below all the values that have big peaks
#FX=FX+[', 'USD/TRY']
Dict = {
    'FXCM': {
            'str_user_i_d': '71533239',
            'str_password': 'qivV5',
            'str_url': "http://www.fxcorporate.com/Hosts.jsp",
            'str_connection': 'Demo',
            'str_session_id': None,
            'str_pin': None,
            'str_table': 'orders',
            'str_account': '71533239',
        },
    'indicators': {
            'sd': datetime.now() - relativedelta(weeks=4),
            'ed': datetime.now(),
        },
    'amount':1,
    'instrument':{
            1:{'open': 7,#opening time in UTC
               'close':16,#closing time in UTC
               'FX':['AUD/CAD', 'AUD/CHF', 'AUD/JPY', 'AUD/NZD', 'AUD/USD', 'AUS200',
      'Bund', 'CAD/CHF', 'CAD/JPY', 'CHF/JPY', 'CHN50', 'Copper',
      'ESP35', 'EUR/AUD', 'EUR/CAD', 'EUR/CHF', 'EUR/GBP', 'EUR/JPY',
      'EUR/NOK', 'EUR/NZD', 'EUR/SEK', 'EUR/TRY', 'EUR/USD', 'EUSTX50',
      'FRA40', 'GBP/AUD', 'GBP/CAD', 'GBP/CHF', 'GBP/JPY', 'GBP/NZD',
      'GBP/USD', 'GER30', 'HKG33', 'JPN225', 'NAS100', 'NGAS', 'NZD/CAD',
      'NZD/CHF', 'NZD/JPY', 'NZD/USD', 'SOYF', 'SPX500', 'TRY/JPY',
      'UK100', 'UKOil', 'US30', 'USD/CAD', 'USD/CHF', 'USD/CNH',
      'USD/HKD', 'USD/JPY', 'USD/MXN', 'USD/NOK', 'USD/SEK',
      'USD/ZAR', 'USDOLLAR', 'USOil', 'XAG/USD', 'XAU/USD', 'ZAR/JPY',
      'USD/ILS', 'VOLX', 'US2000', 'USOilSpot', 'UKOilSpot', 'WHEATF',
      'CORNF', 'EMBasket', 'JPYBasket', 'BTC/USD', 'BCH/USD', 'ETH/USD',
      'LTC/USD', 'CryptoMajor', 'ESPORTS', 'BIOTECH',
      'FAANG', 'CHN.TECH', 'CHN.ECOMM', 'USEquities', 'AIRLINES',
      'CASINOS', 'TRAVEL', 'US.ECOMM', 'US.BANKS', 'US.AUTO', 'WFH',
      'URANIUM', 'BA.us', 'BAC.us', 'BRKB.us', 'C.us', 'CRM.us',
      'DIS.us', 'F.us', 'JPM.us', 'KO.us', 'MA.us', 'MCD.us',
      'PFE.us', 'PG.us', 'SE.us', 'T.us', 'TGT.us', 'V.us', 'XOM.us',
      'AAPL.us', 'AMZN.us', 'BIDU.us', 'GOOG.us', 'INTC.us', 'MSFT.us',
      'SBUX.us', 'ACA.fr', 'AI.fr', 'ALO.fr', 'BN.fr', 'BNP.fr', 'CA.fr',
      'DG.fr', 'AIR.fr', 'ORA.fr', 'GLE.fr', 'MC.fr', 'ML.fr', 'OR.fr',
      'RNO.fr', 'SAN.fr', 'SGO.fr', 'SU.fr', 'VIE.fr', 'VIV.fr',
      'ADS.de', 'ALV.de', 'BAS.de', 'BAYN.de', 'BMW.de',
      'DB1.de', 'DBK.de', 'DPW.de', 'DTE.de', 'EOAN.de', 'IFX.de',
      'LHA.de', 'MRK.de', 'RWE.de', 'SAP.de', 'SIE.de', 'TUI1.de',
      'VOW.de', 'AV.uk', 'AZN.uk', 'BA.uk', 'BARC.uk', 'BATS.uk',
      'BP.uk', 'GSK.uk', 'HSBA.uk', 'IAG.uk', 'LGEN.uk', 'LLOY.uk',
      'RR.uk', 'STAN.uk', 'TSCO.uk', 'VOD.uk', 'BABA.us', 'DAL.us',
      'NFLX.us', 'TSLA.us', 'GLEN.uk', 'TTE.fr', 'ENGI.fr', 'VNA.de',
      'SQ.us', 'LYFT.us', 'UAL.us', 'DKNG.us', 'SHOP.us', 'BYND.us',
      'UBER.us', 'ZM.us', 'LCID.us', 'HOOD.us', 'CRWD.us', 'BEKE.us',
      'CPNG.us', 'NET.us', 'RBLX.us', 'ENR.de', 'BIDU.hk', 'COIN.us',
      'CSL.au', 'CBA.au', 'BHP.au', 'WBC.au', 'NAB.au', 'ANZ.au',
      'WOW.au', 'WES.au', 'FMG.au', 'MQG.au', 'TLS.au', 'RIO.au',
      'GMG.au', 'WPL.au', 'NCM.au', 'COL.au', 'ALL.au', 'A2M.au',
      'REA.au', 'XRO.au', 'QAN.au', 'Z1P.au', 'BT.A.uk', 'NWG.uk',
      'TW.uk', 'MRO.uk', 'MNG.uk', 'ROO.uk', 'CBK.de', 'DHER.de',
      'STM.fr', 'STLA.fr', 'FVRR.us', 'SPOT.us', 'MARA.us', 'BTBT.us',
      'BITF.us', 'WISH.us', 'RIVN.us', 'WE.us', 'JD.us', 'PDD.us',
      'TME.us', 'WB.us', 'BILI.us', 'NVDA.us', 'AMD.us', 'DADA.us',
      'PTON.us', 'TENC.hk', 'MEIT.hk', 'BYDC.hk', 'XIAO.hk', 'BABA.hk',
      'AIA.hk', 'HSBC.hk', 'WUXI.hk', 'HKEX.hk', 'GELY.hk', 'JD.hk',
      'NETE.hk', 'PING.hk', 'SMIC.hk', 'SBIO.hk', 'GALA.hk', 'KIDE.hk',
      'ALIH.hk', 'ICBC.hk', 'FLAT.hk', 'KSOF.hk', 'SMOO.hk', 'SUNN.hk',
      'BYDE.hk', 'MRNA.us', 'NIO.us', 'CCL.us', 'ABNB.us', 'DASH.us',
      'AMC.us', 'BNGO.us', 'FCEL.us', 'GME.us', 'PENN.us', 'PLTR.us',
      'PLUG.us', 'PYPL.us', 'SNAP.us', 'SNOW.us', 'SPCE.us', 'XPEV.us',
      'SONY.us'],
            },
        },
}

def indicators(df):
    def ichimoku(df):
        # Tenkan Sen
        tenkan_max = df['BidHigh'].rolling(window=0, min_periods=0).max()
        tenkan_min = df['BidLow'].rolling(window=0, min_periods=0).min()
        df['tenkan_avg'] = (tenkan_max + tenkan_min) / 2

        # Kijun Sen
        kijun_max = df['BidHigh'].rolling(window=26, min_periods=0).max()
        kijun_min = df['BidLow'].rolling(window=26, min_periods=0).min()
        df['kijun_avg'] = (kijun_max + kijun_min) / 2

        # Senkou Span A
        # (Kijun + Tenkan) / 2 Shifted ahead by 26 periods
        df['senkou_a'] = ((df['kijun_avg'] + df['tenkan_avg']) / 2).shift(26)

        # Senkou Span B
        # 52 period High + Low / 2
        senkou_b_max = df['BidHigh'].rolling(window=52, min_periods=0).max()
        senkou_b_min = df['BidLow'].rolling(window=52, min_periods=0).min()
        df['senkou_b'] = ((senkou_b_max + senkou_b_min) / 2).shift(26)

        # Chikou Span
        # Current Close shifted -26
        df['chikou'] = (df['BidClose']).shift(-26)
        return df

    def macd(df):
        dm = df[['BidClose', 'BidOpen', 'BidLow', 'BidHigh']]
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
        close_delta = df['BidClose'].diff()

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
    df['ci'] = get_ci(df['BidHigh'], df['BidLow'], df['BidClose'], 28)

    return (df)

def analysis(df, ind):
    def chikou_signal(df):

        # Check the Chikou
        df['chikou_signal'] = np.zeros(len(df['BidClose']))
        end_chikou_signal = 30
        if len(df['BidClose']) <= 27:
            end_chikou_signal = len(df['BidClose'])
        for p in range(27, end_chikou_signal):
            # Check if chikou more than anything
            if df.iloc[-p]['chikou'] > df.iloc[-p]['BidClose'].max() \
                    and df.iloc[-p]['chikou'] > df.iloc[-p]['tenkan_avg'].max() \
                    and df.iloc[-p]['chikou'] > df.iloc[-p]['kijun_avg'].max():
                df.loc[len(df) - p, 'chikou_signal'] = 1
            # Check if chikou is less than anything
            elif df.iloc[-p]['chikou'] < df.iloc[-p]['BidClose'].min() \
                    and df.iloc[-p]['chikou'] < df.iloc[-p]['tenkan_avg'].min() \
                    and df.iloc[-p]['chikou'] < df.iloc[-p]['kijun_avg'].min():
                df.loc[len(df) - p, 'chikou_signal'] = -1
            else:
                df.loc[len(df) - p, 'chikou_signal'] = 2

        return df

    def trend_channels(df, backcandles, wind, candleid, brange, ask_plot):
        df_low = df['BidLow']
        df_high = df['BidHigh']
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

    def find_last_peaks(df, ind):
        n = len(df)
        df['peaks'] = np.nan
        df['peaks_macd'] = np.nan
        df['slope'] = np.nan
        df['slope_macd'] = np.nan
        # peaks definition
        for i in range(-ind-1,-n+1, -1):
            # peaks macd
            if abs(df.iloc[i]['macd'])>=abs(df.iloc[i]['signal']):
                # if wants to have slopes on the same side
                # if ((df['slminopt'].dropna()[0] > 0 or df['slmaxopt'].dropna()[0] > 0) and df.iloc[i]['macd'] > 0) or \
                #         ((df['slminopt'].dropna()[0] < 0 or df['slmaxopt'].dropna()[0] < 0) and df.iloc[i]['macd'] < 0):
                if (i ==-2) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i - 1]['macd']) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i - 2]['macd']) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i +1]['macd']):
                    df.loc[n+i, 'peaks_macd'] = df.iloc[i]['macd']
                    df.loc[n+i, 'peaks'] = df.iloc[i]['BidClose']
                elif (i < -2) and\
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i - 1]['macd']) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i + 1]['macd']) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i - 2]['macd']) and \
                        abs(df.iloc[i]['macd']) >= abs(df.iloc[i + 2]['macd']):
                    df.loc[n+i, 'peaks_macd'] = df.iloc[i]['macd']
                    df.loc[n+i, 'peaks'] = df.iloc[i]['BidClose']
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
            df.loc[0, 'slope_macd'] = (df.loc[1, 'slope_macd'] - df.loc[2, 'slope_macd']) / (df.loc[3, 'slope_macd'] - df.loc[4, 'slope_macd'])

        return df

    df = chikou_signal(df)
    # trend_channels defined by how many backcandles we are going RWD, so let's take a 3 months=90days,
    # then by the window of check, let's take 5 days, where I am starting today -4 (yesterday) and what optimization
    # backcandles i ma reday to follow 1 week so 5 days
    df = trend_channels(df, 27*3, 3, len(df) - 4 - ind, 5, False)
    df = find_last_peaks(df, ind)

    # Generate signals
    df['Tenkan-Kijun Cross'] = np.where(df['tenkan_avg'] > df['kijun_avg'], 1, -1)
    df['Price-Above-Cloud'] = np.where(df['BidClose'] > df['senkou_a'], 1, -1)
    df['Price-Above-Kumo'] = np.where(df['BidClose'] > df['senkou_b'], 1, -1)
    df['Cloud-Breakout'] = np.where((df['BidClose'].shift(26) > df['senkou_a'].shift(26)) & (df['BidClose'] > df['senkou_a']), 1, -1)

    # Define entry conditions
    entry_conditions_buy = [
        (df['Tenkan-Kijun Cross'] > 0) &  # Bullish Tenkan-Kijun Cross
        (df['Price-Above-Cloud'] > 0) &  # Price is above the Cloud
        (df['Price-Above-Kumo'] > 0) &  # Price is above Kumo
        (df['Cloud-Breakout'] > 0) &  # Cloud Breakout
        (df['Chikou-Span-Above-Price'] > 0) &  # Chikou Span is above price
        (df['BidClose'] > df['BidClose'].rolling(200).mean()) # Reinforce buy signal if the close price is above the 200-day moving average
    ]
    df['Buy']== np.where(any(entry_conditions_buy), 1, 0)

    # Define entry conditions for sell
    entry_conditions_sell = [
        (df['Tenkan-Kijun Cross'] < 0) &  # Bearish Tenkan-Kijun Cross
        (df['Price-Above-Cloud'] < 0) &  # Price is below the Cloud
        (df['Price-Above-Kumo'] < 0) &  # Price is below Kumo
        (df['Cloud-Breakout'] < 0) &  # Cloud Breakdown
        (df['Chikou-Span-Above-Price'] < 0) & # Chikou Span is below price
        (df['BidClose'] < df['BidClose'].rolling(200).mean())# Reinforce sell signal if the close price is below the 200-day moving average

    ]
    df['Sell']== np.where(any(entry_conditions_sell), 1, 0)

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
        if df['BidHigh'][-28 - index:-2 - index].max() >= df.iloc[-0 - index]['kijun_avg']:
            top_limit = df['BidHigh'][-28 - index:-2 - index].max() - df.iloc[-0 - index]['kijun_avg']
        else:
            top_limit = df.iloc[-0 - index]['kijun_avg'] - df['BidHigh'][-28 - index:-2 - index].max()
        # Lower limit
        if df['BidLow'][-28 - index:-2 - index].min() <= df.iloc[-0 - index]['kijun_avg']:
            low_limit = df.iloc[-0 - index]['kijun_avg'] - df['BidLow'][-28 - index:-2 - index].min()
        else:
            low_limit = df['BidLow'][-28 - index:-2 - index].min() - df.iloc[-0 - index]['kijun_avg']
        max_limit = max(top_limit, low_limit)
        low_box = df.iloc[-0 - index]['kijun_avg'] - max_limit
        high_box = df.iloc[-0 - index]['kijun_avg'] + max_limit
        # Check that kijun is in-between
        if ((low_box + high_box) * 0.45) < df.iloc[m]['kijun_avg'] < ((low_box + high_box) * 0.55):
            box_def == True
        else:
            box_def == False
    return low_box, high_box, box_def

def df_plot(df, tick, type_signal, index, box_def, high_box, low_box, tp, sl):
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
        fig = plt.figure(figsize=(2190 / my_dpi, 1200 / my_dpi), dpi=my_dpi)
        fig.suptitle(tick + type_signal, fontsize=12)
        ax1 = plt.subplot2grid((9, 1), (0, 0), rowspan=4)
        ax2 = plt.subplot2grid((9, 1), (4, 0), rowspan=2, sharex=ax1)
        ax3 = plt.subplot2grid((9, 1), (6, 0), rowspan=1, sharex=ax1)
        ax4 = plt.subplot2grid((9, 1), (7, 0), rowspan=2, sharex=ax1)

        ###AX1
        # Where enter
        ax1.plot(df.index, df['tenkan_avg'], linewidth=2, color='red')
        ax1.plot(df.index, df['kijun_avg'], linewidth=2, color='blue')
        ax1.plot(df.index, df['senkou_a'], linewidth=0.5, color='black')
        ax1.plot(df.index, df['senkou_b'], linewidth=0.5, color='black')
        ax1.plot(df.index, df['chikou'], linewidth=2, color='brown')
        ax1.axhline(y=float(tp), color='blue', linewidth=1, linestyle='-.')
        ax1.axhline(y=float(sl), color='red', linewidth=1, linestyle='-.')
        ax1.plot(df.iloc[-index]['index'], df.iloc[-index]['BidClose'], 'black', marker='s')
        ax1.plot(df.index, df['Buy'], 'green', marker='s')
        ax1.plot(df.index, df['Sell'], 'red', marker='s')
        ax1.plot([df.loc[3, 'slope'],df.loc[4, 'slope']],[df.loc[1, 'slope'],df.loc[2, 'slope']],linewidth=2, color= 'yellow', marker='s')
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
        ax1.fill_between(df.index, df['senkou_a'], df['senkou_b'], where=df['senkou_a'] >= df['senkou_b'],
                         color='lightgreen')
        ax1.fill_between(df.index, df['senkou_a'], df['senkou_b'], where=df['senkou_a'] < df['senkou_b'],
                         color='lightcoral')
        quotes = [tuple(x) for x in df[['index', 'BidOpen', 'BidHigh', 'BidLow', 'BidClose']].values]
        candlestick_ohlc(ax1, quotes, width=0.2, colorup='g', colordown='r')
        # Range_box
        if box_def == True:
            xmin = df['BidLow'][-27 - index:-1 - index].idxmin()
            xmax = df['BidHigh'][-27 - index:-1 - index].idxmax()
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
        low_limit=np.nanmin(df['BidLow'])
        high_limit = np.nanmax(df['BidHigh'])
        ax1.set_ylim(low_limit-0.1*(high_limit-low_limit),high_limit+0.1*(high_limit-low_limit))
        ax1.set_xlim(np.nanmin(df['index']), np.nanmax(df['index']))
        ax1.set(xlabel=None)

        ###AX2
        ax2.bar(df.index, df['macd'], color='grey')
        ax2.plot(df.index, df['signal'], color='red')
        ax2.plot([df.loc[3, 'slope_macd'], df.loc[4, 'slope_macd']], [df.loc[1, 'slope_macd'], df.loc[2, 'slope_macd']], linewidth=2,
                 color='yellow', marker='s')
        ax2.set_ylim(np.nanmin(df['macd']), np.nanmax(df['macd']))
        ax2.grid()
        ax2.set(xlabel=None)

        ax3.bar(df.index, df['Delta'], color='black')
        ax3.set_ylim(np.nanmin(df['Delta']), np.nanmax(df['Delta']))
        ax3.grid()
        ax3.set(xlabel=None)

        ###AX3
        ax4.plot(df.index, df['rsi'], color='black')
        ax4.axhline(y=30, color='grey', linestyle='-.')
        ax4.axhline(y=70, color='grey', linestyle='-.')
        ax4.plot(df.index, df['ci'], color='orange')
        ax4.axhline(y=38.2, color='yellow', linestyle='-.')
        ax4.axhline(y=61.8, color='yellow', linestyle='-.')
        ax4.set_ylim((0, 100))
        ax4.grid()

        #plt.show()
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
        k=0
        for row in orders_table:
            k=k+1
            if row.instrument == tick:
                dj.loc[0,'order_stop_id'] = row.stop_order_id
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
    #get the pip size in all cases
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

def open_trade_old(df, fx, tick, trading_settings_provider,dj,df15):
    def set_amount(lots,dj):
        account = Common.get_account(fx, Dict['FXCM']['str_account'])
        base_unit_size = trading_settings_provider.get_base_unit_size(tick, account)
        amount = int(math.ceil(lots/(dj.loc[0, 'pip_cost']/dj.loc[0, 'pip_size']))*base_unit_size)#int(base_unit_size * lots)
        if amount == 0 : amount=1
        return amount

    open_rev_index = 1
    box_def = False
    high_box = 0
    low_box = 0
    type_signal = 'No'
    tp = 0
    sl = 0
    df = analysis(df, open_rev_index)
    wd = 13
    candle_2 = (df.iloc[-2]['BidClose'] - df.iloc[-2]['BidOpen']) / (df.iloc[-2]['BidHigh'] - df.iloc[-2]['BidLow'])
    margin = 0.1 * (np.nanmax(df.iloc[-27:-2]['BidHigh']) - np.nanmin(df.iloc[-27:-2]['BidLow']))

    #if no gap
    if not (df.iloc[-2]['BidLow'] > margin + df.iloc[-3]['BidHigh']) or \
            (df.iloc[-2]['BidHigh'] < margin + df.iloc[-3]['BidLow']):
        #SELL
        if df.iloc[-2]['signal'] > df.iloc[-2]['macd']  \
            and df.iloc[-4:-2]['macd'].mean() < df.iloc[-5:-3]['macd'].mean()\
            and df.iloc[-4:-2]['Delta'].mean() < df.iloc[-5:-3]['Delta'].mean()\
            and df15.iloc[-4:-2]['macd'].mean() > df15.iloc[-5:-3]['macd'].mean()\
            and df15.iloc[-4:-2]['Delta'].mean() > df15.iloc[-5:-3]['Delta'].mean()\
            and df.iloc[-2]['macd'] < df.iloc[-3]['macd'] \
            and candle_2 != 0.5\
            and df15.iloc[-2]['signal'] > df15.iloc[-2]['macd']\
            and df15.iloc[-3]['signal'] > df15.iloc[-3]['macd']\
            and df.iloc[-2]['tenkan_avg'] < df.iloc[-2]['kijun_avg'] \
            and df15.iloc[-2]['tenkan_avg'] < df15.iloc[-2]['kijun_avg']\
            and df.iloc[-7:-2]['rsi'].min() > 30:
            sl = np.nanmax(df.iloc[-27:-2]['BidHigh']) + margin
            open_price = df.iloc[-2]['BidClose']
            tp = open_price-2*(sl - open_price)
            #SELL in TENDANCE: KUMO is red, slope are negatives
            if df.loc[0, 'slope_macd'] < 0 and df.loc[0, 'slope'] < 0 \
                    and df.iloc[-2]['senkou_a'] < df.iloc[-2]['senkou_b']   \
                    and open_price > (np.array(df['ychannelmin'].dropna())[-1]+np.array(df['ychannelmax'].dropna())[-1])/2 \
                    and df['chikou_signal'].iloc[-28] == -1 \
                    and df['slminopt'].dropna()[0] < 0 \
                    and df['slmaxopt'].dropna()[0] < 0:
                try:
                    amount=set_amount(int(Dict['amount']), dj)
                    type_signal = ' Sell TENDANCE '
                    request = fx.create_order_request(
                        order_type=fxcorepy.Constants.Orders.TRUE_MARKET_OPEN,
                        ACCOUNT_ID=Dict['FXCM']['str_account'],
                        BUY_SELL=fxcorepy.Constants.SELL,
                        AMOUNT=amount,
                        SYMBOL=tick,
                        RATE_STOP=sl,
                        RATE_LIMIT=tp,
                    )
                    fx.send_request(request)
                except Exception as e:
                    type_signal = type_signal + ' not working for ' + str(e)
                    pass
            #SELL in opposite: KUMO is not define and open_price below tenkan & kijun and previoulsy signal was above macd
            elif open_price < df.iloc[-2]['kijun_avg'] and open_price < df.iloc[-2]['tenkan_avg'] \
                    and (df.loc[0, 'slope_macd'] > 0 and df.loc[0, 'slope'] > 0)\
                    and open_price > (np.array(df['ychannelmin'].dropna())[-1]+np.array(df['ychannelmax'].dropna())[-1])/2:
                try:
                    amount=set_amount(int(Dict['amount']), dj)
                    type_signal = ' Sell OPPOSITE '
                    request = fx.create_order_request(
                        order_type=fxcorepy.Constants.Orders.TRUE_MARKET_OPEN,
                        ACCOUNT_ID=Dict['FXCM']['str_account'],
                        BUY_SELL=fxcorepy.Constants.SELL,
                        AMOUNT=amount,
                        SYMBOL=tick,
                        RATE_STOP=sl,
                        RATE_LIMIT=tp,
                    )
                    fx.send_request(request)
                except Exception as e:
                    type_signal = type_signal + ' not working for ' + str(e)
                    pass
            # SELL in opposite: KUMO is not define and open_price below tenkan & kijun and previoulsy signal was above macd
            elif open_price < df.iloc[-2]['kijun_avg'] and open_price < df.iloc[-2]['tenkan_avg'] \
                    and (df.loc[0, 'slope_macd'] < 0 and df.loc[0, 'slope'] > 0) \
                    and df['chikou_signal'].iloc[-28] == -1:
                try:
                    amount = set_amount(int(Dict['amount']), dj)
                    type_signal = ' Sell DIVERGENCE '
                    request = fx.create_order_request(
                        order_type=fxcorepy.Constants.Orders.TRUE_MARKET_OPEN,
                        ACCOUNT_ID=Dict['FXCM']['str_account'],
                        BUY_SELL=fxcorepy.Constants.SELL,
                        AMOUNT=amount,
                        SYMBOL=tick,
                        RATE_STOP=sl,
                        RATE_LIMIT=tp,
                    )
                    fx.send_request(request)
                except Exception as e:
                    type_signal = type_signal + ' not working for ' + str(e)
                    pass


        #BUY
        elif df.iloc[-2]['signal'] < df.iloc[-2]['macd']\
            and df.iloc[-4:-2]['macd'].mean() > df.iloc[-5:-3]['macd'].mean() \
            and df.iloc[-4:-2]['Delta'].mean() > df.iloc[-5:-3]['Delta'].mean() \
            and df15.iloc[-4:-2]['macd'].mean() > df15.iloc[-5:-3]['macd'].mean()\
            and df15.iloc[-4:-2]['Delta'].mean() > df15.iloc[-5:-3]['Delta'].mean()\
            and df.iloc[-2]['macd'] > df.iloc[-3]['macd'] \
            and candle_2 !=-0.5\
            and df15.iloc[-2]['signal'] < df15.iloc[-2]['macd']\
            and df15.iloc[-3]['signal'] < df15.iloc[-3]['macd']\
            and df.iloc[-2]['tenkan_avg'] > df.iloc[-2]['kijun_avg']\
            and df15.iloc[-2]['tenkan_avg'] > df15.iloc[-2]['kijun_avg']\
            and df.iloc[-7:-2]['rsi'].max() < 70:
            sl = np.nanmin(df.iloc[-27:-2]['BidLow']) - margin
            open_price = df.iloc[-2]['BidClose']
            tp=(2*(open_price - sl)+open_price)
            #BUY in TENDANCE: KUMO is green, slope are positives
            if df.loc[0, 'slope_macd'] > 0 and df.loc[0, 'slope'] > 0  \
                    and df.iloc[-2]['senkou_a'] > df.iloc[-2]['senkou_b'] \
                    and open_price < (np.array(df['ychannelmin'].dropna())[-1]+np.array(df['ychannelmax'].dropna())[-1])/2\
                    and df['chikou_signal'].iloc[-28] == 1 \
                    and df['slminopt'].dropna()[0] > 0 \
                    and df['slmaxopt'].dropna()[0] > 0 :
                try:
                    amount=set_amount(int(Dict['amount']), dj)
                    type_signal = ' BUY TENDANCE '
                    request = fx.create_order_request(
                        order_type=fxcorepy.Constants.Orders.TRUE_MARKET_OPEN,
                        ACCOUNT_ID=Dict['FXCM']['str_account'],
                        BUY_SELL=fxcorepy.Constants.BUY,
                        AMOUNT=amount,
                        SYMBOL=tick,
                        RATE_STOP=sl,
                        RATE_LIMIT=tp,
                    )
                    fx.send_request(request)
                except Exception as e:
                    type_signal = type_signal + ' not working for ' + str(e)
                    pass
            #BUY in opposite: KUMO is not define and open_price above tenkan & kijun and previoulsy signal was below macd
            elif open_price > df.iloc[-2]['kijun_avg'] and open_price > df.iloc[-2]['tenkan_avg']\
                    and (df.loc[0, 'slope_macd'] < 0 and df.loc[0, 'slope'] < 0) \
                    and open_price < (np.array(df['ychannelmin'].dropna())[-1]+np.array(df['ychannelmax'].dropna())[-1])/2:
                try:
                    amount=set_amount(int(Dict['amount']), dj)
                    type_signal = ' BUY OPPOSITE '
                    request = fx.create_order_request(
                        order_type=fxcorepy.Constants.Orders.TRUE_MARKET_OPEN,
                        ACCOUNT_ID=Dict['FXCM']['str_account'],
                        BUY_SELL=fxcorepy.Constants.BUY,
                        AMOUNT=amount,
                        SYMBOL=tick,
                        RATE_STOP=sl,
                        RATE_LIMIT=tp,
                    )
                    fx.send_request(request)
                except Exception as e:
                    type_signal = type_signal + ' not working for ' + str(e)
                    pass
            #BUY in Divergence: KUMO is not define and open_price above tenkan & kijun and previoulsy signal was below macd
            elif open_price > df.iloc[-2]['kijun_avg'] and open_price > df.iloc[-2]['tenkan_avg']\
                    and (df.loc[0, 'slope_macd'] > 0 and df.loc[0, 'slope'] < 0) \
                    and df['chikou_signal'].iloc[-28] == 1:
                try:
                    amount=set_amount(int(Dict['amount']), dj)
                    type_signal = ' BUY Divergence '
                    request = fx.create_order_request(
                        order_type=fxcorepy.Constants.Orders.TRUE_MARKET_OPEN,
                        ACCOUNT_ID=Dict['FXCM']['str_account'],
                        BUY_SELL=fxcorepy.Constants.BUY,
                        AMOUNT=amount,
                        SYMBOL=tick,
                        RATE_STOP=sl,
                        RATE_LIMIT=tp,
                    )
                    fx.send_request(request)
                except Exception as e:
                    type_signal = type_signal + ' not working for ' + str(e)
                    pass

    return df, tick, type_signal, open_rev_index, box_def, high_box, low_box, tp, sl

def open_trade_old2(df, fx, tick, trading_settings_provider,dj,dfd1):
    def set_amount(lots,dj):
        account = Common.get_account(fx, Dict['FXCM']['str_account'])
        base_unit_size = trading_settings_provider.get_base_unit_size(tick, account)
        amount = int(math.ceil(lots/(dj.loc[0, 'pip_cost']/dj.loc[0, 'pip_size']))*base_unit_size)#int(base_unit_size * lots)
        if amount == 0 : amount=1
        return amount

    def take_profit(type,open_price,df, sl):
        tp=None
        #Found the next range
        for i in range(5, len(df)-5):
            if df.iloc[-i]['kijun_avg']==df.iloc[-i-1]['kijun_avg'] \
                and df.iloc[-i]['kijun_avg'] == df.iloc[-i - 2]['kijun_avg'] \
                and df.iloc[-i]['kijun_avg'] == df.iloc[-i - 3]['kijun_avg'] \
                and df.iloc[-i]['kijun_avg'] == df.iloc[-i - 4]['kijun_avg']:
                    if type=="sell" and df.iloc[-i]['kijun_avg'] < open_price:
                        tp = df.iloc[-i]['kijun_avg']
                        return tp
                    elif type == "buy" and df.iloc[-i]['kijun_avg'] > open_price:
                        tp = df.iloc[-i]['kijun_avg']
                        return tp
        # if kijun not found then look for the max
        if tp is None:
            if type == "sell" and min(df.iloc[-27*3:-2]['BidLow'])< open_price:
                tp = min(df.iloc[-27*3:-2]['BidLow'])
                return tp
            elif type == "buy" and max(df.iloc[-27*3:-2]['BidHigh'])> open_price:
                tp = max(df.iloc[-27*3:-2]['BidHigh'])
                return tp
        # if kijun not found no max peak then take the double
        if tp is None:
            if type == "sell":
                tp = open_price - 2 * (sl - open_price)
                return tp
            elif type == "buy":
                tp = (2 * (open_price - sl) + open_price)
                return tp

    open_rev_index = 1
    box_def = False
    high_box = 0
    low_box = 0
    type_signal = 'No'
    tp = 0
    sl = 0
    df = analysis(df, open_rev_index)
    wd = 13
    candle_2 = (df.iloc[-2]['BidClose'] - df.iloc[-2]['BidOpen']) / (df.iloc[-2]['BidHigh'] - df.iloc[-2]['BidLow'])
    margin = 0.1 * (np.nanmax(df.iloc[-27:-2]['BidHigh']) - np.nanmin(df.iloc[-27:-2]['BidLow']))

    #if no gap and tenkan not flat and a strong canfdles
    if (df.iloc[-2]['BidLow'] <= df.iloc[-3]['BidHigh'] or df.iloc[-2]['BidHigh'] <= df.iloc[-3]['BidLow']) or \
        (df.iloc[-3]['BidLow'] <= df.iloc[-4]['BidHigh'] or df.iloc[-3]['BidHigh'] <= df.iloc[-4]['BidLow']) or \
        (df.iloc[-4]['BidLow'] <= df.iloc[-5]['BidHigh'] or df.iloc[-4]['BidHigh'] <= df.iloc[-5]['BidLow']) \
        and df.iloc[-2]['tenkan_avg'] != df.iloc[-3]['tenkan_avg']\
        and df.iloc[-1]['tenkan_avg'] != df.iloc[-2]['tenkan_avg']\
        and (df.iloc[-2]['BidHigh']-df.iloc[-1]['BidLow'])<3*np.mean(df.iloc[-7:-2]['BidHigh']-df.iloc[-7:-2]['BidLow']):

        #SELL TENDANCE
        if df.iloc[-2]['BidClose'] < max(df.iloc[-2]['senkou_a'],df.iloc[-2]['senkou_b'])\
            and df.iloc[-2]['kijun_avg'] < max(df.iloc[-2]['senkou_a'],df.iloc[-2]['senkou_b'])\
            and df.iloc[-2]['tenkan_avg'] < max(df.iloc[-2]['senkou_a'],df.iloc[-2]['senkou_b']) \
            and df.iloc[-2]['BidClose'] < df.iloc[-2]['kijun_avg'] \
            and df.iloc[-2]['BidClose'] < df.iloc[-2]['tenkan_avg'] \
            and df.iloc[-2]['tenkan_avg'] < df.iloc[-2]['kijun_avg'] \
            and df.iloc[-2]['signal'] > df.iloc[-2]['macd']\
            and df.iloc[-4:-2]['macd'].mean() < df.iloc[-5:-3]['macd'].mean() \
            and df.iloc[-4:-2]['Delta'].mean() < df.iloc[-5:-3]['Delta'].mean() \
            and df.iloc[-2]['macd'] < df.iloc[-3]['macd'] \
            and candle_2 < 0\
            and df.iloc[-7:-2]['rsi'].min() > 35\
            and df['chikou_signal'].iloc[-28] == -1:
            sl = df.iloc[-2]['kijun_avg']
            open_price = df.iloc[-2]['BidClose']
            tp = take_profit("sell",open_price,df, sl)
            print("ratio:" + str((open_price- tp)/(sl - open_price)))
            if (open_price- tp)/(sl - open_price)>1:
                try:
                    amount = set_amount(int(Dict['amount']), dj)
                    type_signal = ' Sell TENDANCE '
                    request = fx.create_order_request(
                        order_type=fxcorepy.Constants.Orders.TRUE_MARKET_OPEN,
                        ACCOUNT_ID=Dict['FXCM']['str_account'],
                        BUY_SELL=fxcorepy.Constants.SELL,
                        AMOUNT=amount,
                        SYMBOL=tick,
                        RATE_STOP=sl+margin,
                        RATE_LIMIT=tp+margin,
                    )
                    fx.send_request(request)
                except Exception as e:
                    type_signal = type_signal + ' not working for ' + str(e)
                    pass
        #BUY TENDANCE
        elif df.iloc[-2]['BidClose'] > max(df.iloc[-2]['senkou_a'],df.iloc[-2]['senkou_b'])\
            and df.iloc[-2]['kijun_avg'] > max(df.iloc[-2]['senkou_a'],df.iloc[-2]['senkou_b'])\
            and df.iloc[-2]['tenkan_avg'] > max(df.iloc[-2]['senkou_a'],df.iloc[-2]['senkou_b']) \
            and df.iloc[-2]['BidClose'] > df.iloc[-2]['kijun_avg'] \
            and df.iloc[-2]['BidClose'] > df.iloc[-2]['tenkan_avg'] \
            and df.iloc[-2]['tenkan_avg'] > df.iloc[-2]['kijun_avg'] \
            and df.iloc[-2]['signal'] < df.iloc[-2]['macd']\
            and df.iloc[-4:-2]['macd'].mean() > df.iloc[-5:-3]['macd'].mean() \
            and df.iloc[-4:-2]['Delta'].mean() > df.iloc[-5:-3]['Delta'].mean() \
            and df.iloc[-2]['macd'] > df.iloc[-3]['macd'] \
            and candle_2 > 0\
            and df.iloc[-7:-2]['rsi'].max() < 65\
            and df['chikou_signal'].iloc[-28] == 1:
            sl = df.iloc[-2]['kijun_avg']
            open_price = df.iloc[-2]['BidClose']
            tp = take_profit("buy",open_price,df, sl)
            print("ratio:" + str((tp-open_price) / (open_price - sl)))
            if (tp-open_price) / (open_price - sl)>1:
                try:
                    amount=set_amount(int(Dict['amount']), dj)
                    type_signal = ' BUY TENDANCE '
                    request = fx.create_order_request(
                        order_type=fxcorepy.Constants.Orders.TRUE_MARKET_OPEN,
                        ACCOUNT_ID=Dict['FXCM']['str_account'],
                        BUY_SELL=fxcorepy.Constants.BUY,
                        AMOUNT=amount,
                        SYMBOL=tick,
                        RATE_STOP=sl-margin,
                        RATE_LIMIT=tp-margin,
                    )
                    fx.send_request(request)
                except Exception as e:
                    type_signal = type_signal + ' not working for ' + str(e)
                    pass

    return df, tick, type_signal, open_rev_index, box_def, high_box, low_box, tp, sl

def open_trade_old(df, fx, tick, trading_settings_provider,dj,dfd1):
    def set_amount(lots,dj):
        account = Common.get_account(fx, Dict['FXCM']['str_account'])
        base_unit_size = trading_settings_provider.get_base_unit_size(tick, account)
        amount = int(math.ceil(lots/(dj.loc[0, 'pip_cost']/dj.loc[0, 'pip_size']))*base_unit_size)#int(base_unit_size * lots)
        if amount == 0 : amount=1
        return amount

    def take_profit(type,open_price,df):
        tp=None
        #Found the next range
        for i in range(5, len(df)-5):
            if df.iloc[-i]['kijun_avg']==df.iloc[-i-1]['kijun_avg'] \
                and df.iloc[-i]['kijun_avg'] == df.iloc[-i - 2]['kijun_avg'] \
                and df.iloc[-i]['kijun_avg'] == df.iloc[-i - 3]['kijun_avg'] \
                and df.iloc[-i]['kijun_avg'] == df.iloc[-i - 4]['kijun_avg']:
                    if type=="sell" and df.iloc[-i]['kijun_avg'] < open_price:
                        tp = df.iloc[-i]['kijun_avg']
                        return tp
                    elif type == "buy" and df.iloc[-i]['kijun_avg'] > open_price:
                        tp = df.iloc[-i]['kijun_avg']
                        return tp
        # if kijun not found then look for the max
        if tp is None:
            if type == "sell" and min(df.iloc[-27*3:-2]['BidLow'])< open_price:
                tp = min(df.iloc[-27*3:-2]['BidLow'])
                return tp
            elif type == "buy" and max(df.iloc[-27*3:-2]['BidHigh'])> open_price:
                tp = max(df.iloc[-27*3:-2]['BidHigh'])
                return tp
        # if kijun not found no max peak then take the double
        if tp is None:
            if type == "sell":
                tp = np.array(df['ychannelmin'].dropna())[-1]
                return tp
            elif type == "buy":
                tp = np.array(df['ychannelmax'].dropna())[-1]
                return tp
        if tp is None:
            #fibonacci when possible
            tp=open_price
            return tp

    def stop_loss(type,open_price,df):
        sl=None
        #Found the next range
        for i in range(5, len(df)-5):
            if df.iloc[-i]['kijun_avg']==df.iloc[-i-1]['kijun_avg'] \
                and df.iloc[-i]['kijun_avg'] == df.iloc[-i - 2]['kijun_avg'] \
                and df.iloc[-i]['kijun_avg'] == df.iloc[-i - 3]['kijun_avg'] \
                and df.iloc[-i]['kijun_avg'] == df.iloc[-i - 4]['kijun_avg']:
                    if type=="sell" and df.iloc[-i]['kijun_avg'] > open_price:
                        sl = df.iloc[-i]['kijun_avg']
                        return sl
                    elif type == "buy" and df.iloc[-i]['kijun_avg'] < open_price:
                        sl = df.iloc[-i]['kijun_avg']
                        return sl
        # if kijun not found then look for the max
        if sl is None:
            if type == "sell" and max(df.iloc[-27*3:-2]['BidHigh'])> open_price:
                sl = max(df.iloc[-27*3:-2]['BidHigh'])
                return sl
            elif type == "buy" and min(df.iloc[-27*3:-2]['BidLow'])< open_price:
                sl = min(df.iloc[-27*3:-2]['BidLow'])
                return sl
        # if kijun not found no max peak then take the double
        if sl is None:
            if type == "sell":
                sl = np.array(df['ychannelmax'].dropna())[-1]
                return sl
            elif type == "buy":
                sl = np.array(df['ychannelmin'].dropna())[-1]
                return sl
        if sl is None:
            #fibonacci when possible
            sl=open_price
            return sl

    open_rev_index = 1
    box_def = False
    high_box = 0
    low_box = 0
    type_signal = 'No'
    tp = 0
    sl = 0
    df = analysis(df, open_rev_index)
    wd = 13
    candle_2 = (df.iloc[-2]['BidClose'] - df.iloc[-2]['BidOpen']) / (df.iloc[-2]['BidHigh'] - df.iloc[-2]['BidLow'])
    margin = abs(0.1 * (np.nanmax(df.iloc[-27:-2]['BidHigh']) - np.nanmin(df.iloc[-27:-2]['BidLow'])))

    #if no gap and tenkan not flat and a strong canfdles
    if (df.iloc[-2]['BidLow'] <= df.iloc[-3]['BidHigh'] or df.iloc[-2]['BidHigh'] <= df.iloc[-3]['BidLow']) or \
        (df.iloc[-3]['BidLow'] <= df.iloc[-4]['BidHigh'] or df.iloc[-3]['BidHigh'] <= df.iloc[-4]['BidLow']) or \
        (df.iloc[-4]['BidLow'] <= df.iloc[-5]['BidHigh'] or df.iloc[-4]['BidHigh'] <= df.iloc[-5]['BidLow']) \
        and (df.iloc[-2]['BidHigh']-df.iloc[-1]['BidLow'])<3*np.mean(df.iloc[-7:-2]['BidHigh']-df.iloc[-7:-2]['BidLow']):

        #SELL TENDANCE
        if df.iloc[-28]['chikou_signal'] == -1\
            and df.loc[0, 'slope_macd'] < 0\
            and df.loc[0, 'slope'] < 0\
            and df.iloc[-2]['macd'] < df.iloc[-3]['macd']\
            and df.iloc[-2]['Delta'] < df.iloc[-3]['Delta']\
            and df.iloc[-2]['tenkan_avg'] < df.iloc[-3]['tenkan_avg']\
            and df.iloc[-2]['kijun_avg'] > df.iloc[-3]['kijun_avg']\
            and df.iloc[-2]['signal'] > df.iloc[-2]['macd'] \
            and df.iloc[-2]['tenkan_avg'] < df.iloc[-2]['kijun_avg']        \
            and candle_2<0\
            and df.iloc[-7:-2]['rsi'].min() > 30:
            open_price = df.iloc[-2]['BidClose']
            sl = stop_loss("sell", open_price, df)
            tp = take_profit("sell",open_price,df)
            #print("ratio:" + str((open_price- tp)/(sl - open_price)))
            if (open_price- tp)/(sl - open_price)>2:
                try:
                    amount = set_amount(int(Dict['amount']), dj)
                    type_signal = ' Sell TENDANCE ratio: '+ str((open_price- tp)/(sl - open_price))
                    request = fx.create_order_request(
                        order_type=fxcorepy.Constants.Orders.TRUE_MARKET_OPEN,
                        ACCOUNT_ID=Dict['FXCM']['str_account'],
                        BUY_SELL=fxcorepy.Constants.SELL,
                        AMOUNT=amount,
                        SYMBOL=tick,
                        RATE_STOP=sl+margin,
                        RATE_LIMIT=tp+margin,
                    )
                    fx.send_request(request)
                except Exception as e:
                    type_signal = type_signal + ' not working for ' + str(e)
                    pass
        #BUY TENDANCE
        elif df.iloc[-28]['chikou_signal'] == 1\
            and df.loc[0, 'slope_macd'] > 0\
            and df.loc[0, 'slope'] > 0\
            and df.iloc[-2]['macd'] > df.iloc[-3]['macd']\
            and df.iloc[-2]['Delta'] > df.iloc[-3]['Delta']\
            and df.iloc[-2]['tenkan_avg'] > df.iloc[-3]['tenkan_avg']\
            and df.iloc[-2]['kijun_avg'] > df.iloc[-3]['kijun_avg']\
            and df.iloc[-2]['signal'] < df.iloc[-2]['macd']\
            and df.iloc[-2]['tenkan_avg'] > df.iloc[-2]['kijun_avg'] \
            and candle_2 > 0 \
            and df.iloc[-7:-2]['rsi'].max() < 70:
            open_price = df.iloc[-2]['BidClose']
            sl = stop_loss("buy", open_price, df)
            tp = take_profit("buy",open_price,df)
            #print("ratio:" + str((tp-open_price) / (open_price - sl)))
            if (tp-open_price) / (open_price - sl)>2:
                try:
                    amount=set_amount(int(Dict['amount']), dj)
                    type_signal = ' BUY TENDANCE ratio: '+ str((tp-open_price) / (open_price - sl))
                    request = fx.create_order_request(
                        order_type=fxcorepy.Constants.Orders.TRUE_MARKET_OPEN,
                        ACCOUNT_ID=Dict['FXCM']['str_account'],
                        BUY_SELL=fxcorepy.Constants.BUY,
                        AMOUNT=amount,
                        SYMBOL=tick,
                        RATE_STOP=sl-margin,
                        RATE_LIMIT=tp-margin,
                    )
                    fx.send_request(request)
                except Exception as e:
                    type_signal = type_signal + ' not working for ' + str(e)
                    pass

    return df, tick, type_signal, open_rev_index, box_def, high_box, low_box, tp, sl

def open_trade_new(df, fx, tick, trading_settings_provider,dj,dfd1):
    def set_amount(lots,dj):
        account = Common.get_account(fx, Dict['FXCM']['str_account'])
        base_unit_size = trading_settings_provider.get_base_unit_size(tick, account)
        amount = int(math.ceil(lots/(dj.loc[0, 'pip_cost']/dj.loc[0, 'pip_size']))*base_unit_size)#int(base_unit_size * lots)
        if amount == 0 : amount=1
        return amount

    def take_profit(type,open_price,df):
        tp=None
        #Found the next range
        for i in range(5, len(df)-5):
            if df.iloc[-i]['kijun_avg']==df.iloc[-i-1]['kijun_avg'] \
                and df.iloc[-i]['kijun_avg'] == df.iloc[-i - 2]['kijun_avg'] \
                and df.iloc[-i]['kijun_avg'] == df.iloc[-i - 3]['kijun_avg'] \
                and df.iloc[-i]['kijun_avg'] == df.iloc[-i - 4]['kijun_avg']:
                    if type=="sell" and df.iloc[-i]['kijun_avg'] < open_price:
                        tp = df.iloc[-i]['kijun_avg']
                        return tp
                    elif type == "buy" and df.iloc[-i]['kijun_avg'] > open_price:
                        tp = df.iloc[-i]['kijun_avg']
                        return tp
        # if kijun not found then look for the max
        if tp is None:
            if type == "sell" and min(df.iloc[-27*3:-2]['BidLow'])< open_price:
                tp = min(df.iloc[-27*3:-2]['BidLow'])
                return tp
            elif type == "buy" and max(df.iloc[-27*3:-2]['BidHigh'])> open_price:
                tp = max(df.iloc[-27*3:-2]['BidHigh'])
                return tp
        # if kijun not found no max peak then take the double
        if tp is None:
            if type == "sell":
                tp = np.array(df['ychannelmin'].dropna())[-1]
                return tp
            elif type == "buy":
                tp = np.array(df['ychannelmax'].dropna())[-1]
                return tp
        if tp is None:
            #fibonacci when possible
            tp=open_price
            return tp

    def stop_loss(type,open_price,df):
        sl=None
        #Found the next range
        for i in range(5, len(df)-5):
            if df.iloc[-i]['kijun_avg']==df.iloc[-i-1]['kijun_avg'] \
                and df.iloc[-i]['kijun_avg'] == df.iloc[-i - 2]['kijun_avg'] \
                and df.iloc[-i]['kijun_avg'] == df.iloc[-i - 3]['kijun_avg'] \
                and df.iloc[-i]['kijun_avg'] == df.iloc[-i - 4]['kijun_avg']:
                    if type=="sell" and df.iloc[-i]['kijun_avg'] > open_price:
                        sl = df.iloc[-i]['kijun_avg']
                        return sl
                    elif type == "buy" and df.iloc[-i]['kijun_avg'] < open_price:
                        sl = df.iloc[-i]['kijun_avg']
                        return sl
        # if kijun not found then look for the max
        if sl is None:
            if type == "sell" and max(df.iloc[-27*3:-2]['BidHigh'])> open_price:
                sl = max(df.iloc[-27*3:-2]['BidHigh'])
                return sl
            elif type == "buy" and min(df.iloc[-27*3:-2]['BidLow'])< open_price:
                sl = min(df.iloc[-27*3:-2]['BidLow'])
                return sl
        # if kijun not found no max peak then take the double
        if sl is None:
            if type == "sell":
                sl = np.array(df['ychannelmax'].dropna())[-1]
                return sl
            elif type == "buy":
                sl = np.array(df['ychannelmin'].dropna())[-1]
                return sl
        if sl is None:
            #fibonacci when possible
            sl=open_price
            return sl

    open_rev_index = 1
    box_def = False
    high_box = 0
    low_box = 0
    type_signal = 'No'
    tp = 0
    sl = 0
    df = analysis(df, open_rev_index)
    wd = 13
    candle_2 = (df.iloc[-2]['BidClose'] - df.iloc[-2]['BidOpen']) / (df.iloc[-2]['BidHigh'] - df.iloc[-2]['BidLow'])
    margin = abs(0.1 * (np.nanmax(df.iloc[-27:-2]['BidHigh']) - np.nanmin(df.iloc[-27:-2]['BidLow'])))

    #if no gap and tenkan not flat and a strong canfdles
    if (df.iloc[-2]['BidLow'] <= df.iloc[-3]['BidHigh'] or df.iloc[-2]['BidHigh'] <= df.iloc[-3]['BidLow']) or \
        (df.iloc[-3]['BidLow'] <= df.iloc[-4]['BidHigh'] or df.iloc[-3]['BidHigh'] <= df.iloc[-4]['BidLow']) or \
        (df.iloc[-4]['BidLow'] <= df.iloc[-5]['BidHigh'] or df.iloc[-4]['BidHigh'] <= df.iloc[-5]['BidLow']) \
        and (df.iloc[-2]['BidHigh']-df.iloc[-1]['BidLow'])<3*np.mean(df.iloc[-7:-2]['BidHigh']-df.iloc[-7:-2]['BidLow']):

        #SELL TENDANCE
        if df.iloc[-1]['Sell'] == 1:
            open_price = df.iloc[-2]['BidClose']
            sl = stop_loss("sell", open_price, df)
            tp = take_profit("sell",open_price,df)
            #print("ratio:" + str((open_price- tp)/(sl - open_price)))
            if (open_price- tp)/(sl - open_price)>2:
                try:
                    amount = set_amount(int(Dict['amount']), dj)
                    type_signal = ' Sell TENDANCE ratio: '+ str((open_price- tp)/(sl - open_price))
                    request = fx.create_order_request(
                        order_type=fxcorepy.Constants.Orders.TRUE_MARKET_OPEN,
                        ACCOUNT_ID=Dict['FXCM']['str_account'],
                        BUY_SELL=fxcorepy.Constants.SELL,
                        AMOUNT=amount,
                        SYMBOL=tick,
                        RATE_STOP=sl+margin,
                        RATE_LIMIT=tp+margin,
                    )
                    fx.send_request(request)
                except Exception as e:
                    type_signal = type_signal + ' not working for ' + str(e)
                    pass
        #BUY TENDANCE
        elif df.iloc[-1]['Buy'] == 1:
            open_price = df.iloc[-2]['BidClose']
            sl = stop_loss("buy", open_price, df)
            tp = take_profit("buy",open_price,df)
            #print("ratio:" + str((tp-open_price) / (open_price - sl)))
            if (tp-open_price) / (open_price - sl)>2:
                try:
                    amount=set_amount(int(Dict['amount']), dj)
                    type_signal = ' BUY TENDANCE ratio: '+ str((tp-open_price) / (open_price - sl))
                    request = fx.create_order_request(
                        order_type=fxcorepy.Constants.Orders.TRUE_MARKET_OPEN,
                        ACCOUNT_ID=Dict['FXCM']['str_account'],
                        BUY_SELL=fxcorepy.Constants.BUY,
                        AMOUNT=amount,
                        SYMBOL=tick,
                        RATE_STOP=sl-margin,
                        RATE_LIMIT=tp-margin,
                    )
                    fx.send_request(request)
                except Exception as e:
                    type_signal = type_signal + ' not working for ' + str(e)
                    pass

    return df, tick, type_signal, open_rev_index, box_def, high_box, low_box, tp, sl

def close_trade(df, fx, tick,dj,df15):
    try:
        open_rev_index = [len(df) - df.index[df['Date'].dt.strftime("%m%d%Y%H") == dj.loc[0,'tick_time'].strftime("%m%d%Y%H")]][0][
            0]
    except:
        open_rev_index = 1
    type_signal = 'No'
    open_price = dj.loc[0,'tick_open_price']
    price = dj.loc[0,'tick_price']
    box_def = False
    high_box = 0
    low_box = 0
    df = analysis(df, open_rev_index)
    tp = dj.loc[0,'tick_limit']
    sl = dj.loc[0,'tick_stop']
    offer = Common.get_offer(fx, tick)
    buy = fxcorepy.Constants.BUY
    sell = fxcorepy.Constants.SELL
    buy_sell = sell if dj.loc[0,'tick_type'] == buy else buy
    order_id = None
    candle_2 = (df.iloc[-2]['BidClose'] - df.iloc[-2]['BidOpen'])/(df.iloc[-2]['BidHigh'] - df.iloc[-2]['BidLow'])
    candle_3 = (df.iloc[-3]['BidClose'] - df.iloc[-3]['BidOpen']) / (df.iloc[-3]['BidHigh'] - df.iloc[-3]['BidLow'])
    candle_4 = (df.iloc[-4]['BidClose'] - df.iloc[-4]['BidOpen']) / (df.iloc[-4]['BidHigh'] - df.iloc[-4]['BidLow'])

    if df['ychannelmin'].dropna().size != 0:
        # if market was in range
        if open_rev_index<1:
                print('open_rev_index too small')
        else:
            # if was buy
            if dj.loc[0,'tick_type'] == 'B':
                open_sl = df.iloc[-27:-2]['BidLow'].min() - 0.1 * (
                            df.iloc[-27:-2]['BidHigh'].max() - df.iloc[-27:-2]['BidLow'].min())
                current_ratio = (price - open_price) / (open_price - open_sl)
                if (df.iloc[-2]['macd'] < df.iloc[-3]['macd'] and df.iloc[-3]['macd'] < df.iloc[-4]['macd'] and current_ratio>0)\
                    or \
                        (df.iloc[-2]['signal'] > df.iloc[-2]['macd'] and candle_2 <-0.5 and current_ratio>0) \
                    or \
                        (df.iloc[-2]['BidLow'] < df.iloc[-2]['tenkan_avg'] and current_ratio>0 and df15.iloc[-2]['tenkan_avg']<df15.iloc[-2]['kijun_avg']) \
                    or \
                        (df.iloc[-2]['signal'] > df.iloc[-2]['macd'] and df.iloc[-3]['signal'] > df.iloc[-3]['macd']
                         and (candle_2<-0.5 or candle_3<-0.5) and current_ratio>0) \
                    or \
                        (df.iloc[-2]['macd'] < df.iloc[-3]['macd'] and (candle_2 < -0.5 or candle_3 < -0.5)
                         and df.iloc[-2]['BidLow'] < df.iloc[-2]['tenkan_avg'] and current_ratio>0)\
                    or \
                        (df.iloc[-3:-2]['macd'].mean() < df.iloc[-4:-3]['macd'].mean() \
                        and abs(df.iloc[-3:-2]['Delta'].mean()) < abs(df.iloc[-4:-3]['Delta'].mean()) \
                        and df.iloc[-2]['signal'] > df.iloc[-2]['macd'] and (candle_2 < -0.5 or candle_3 < -0.5)\
                         and df.iloc[-2]['BidLow'] < df.iloc[-2]['tenkan_avg'])\
                    or \
                        ((df.iloc[-2]['BidHigh'] - df.iloc[-2]['BidLow']) > (df.iloc[-3]['BidHigh'] - df.iloc[-3]['BidLow'])
                         and current_ratio>0 and candle_2 < 0 ) \
                    or \
                        (max(df.iloc[-7:-2]['rsi']) > 70 and current_ratio > 0 and candle_2 < 0)\
                    or \
                        (df.iloc[-2]['macd'] < df.iloc[-3]['macd'] and abs(df.iloc[-2]['Delta']) < abs(df.iloc[-3]['Delta']) and current_ratio > 0 and candle_2 < 0):
                    try:
                        type_signal = ' Buy : Close for Signal higher than MACD ' + str(current_ratio)
                        request = fx.create_order_request(
                            order_type=fxcorepy.Constants.Orders.TRUE_MARKET_CLOSE,
                            OFFER_ID=offer.offer_id,
                            ACCOUNT_ID=Dict['FXCM']['str_account'],
                            BUY_SELL=buy_sell,
                            AMOUNT=int(dj.loc[0,'tick_amount']),
                            TRADE_ID=dj.loc[0,'tick_id']
                        )
                        resp = fx.send_request(request)
                    except Exception as e:
                        type_signal = type_signal + ' not working for ' + str(e)
                        pass
                elif df.iloc[-2]['tenkan_avg'] == df.iloc[-3]['tenkan_avg'] and df.iloc[-3]['tenkan_avg'] == df.iloc[-4]['tenkan_avg']\
                    and abs(df.iloc[-3:-1]['Delta'].mean()) < abs(df.iloc[-4:-2]['Delta'].mean()):
                        if current_ratio > 0:
                            try:
                                type_signal = ' Buy : Close Tenkan Flat ' + str(current_ratio)
                                request = fx.create_order_request(
                                    order_type=fxcorepy.Constants.Orders.TRUE_MARKET_CLOSE,
                                    OFFER_ID=offer.offer_id,
                                    ACCOUNT_ID=Dict['FXCM']['str_account'],
                                    BUY_SELL=buy_sell,
                                    AMOUNT=int(dj.loc[0,'tick_amount']),
                                    TRADE_ID=dj.loc[0,'tick_id']
                                )
                                resp = fx.send_request(request)
                            except Exception as e:
                                type_signal = type_signal + ' not working for ' + str(e)
                                pass
                        else:
                            try:
                                type_signal = ' Buy : Adjust for Tenkan Flat ' + str(current_ratio)
                                sl = df.iloc[-3:-1]['BidLow'].min() - 0.1 * (
                                            df.iloc[-27:-1]['BidHigh'].max() - df.iloc[-27:-1]['BidLow'].min())
                                if sl >= price: sl = df.iloc[-1]['BidLow']
                                request = fx.create_order_request(
                                    order_type=fxcorepy.Constants.Orders.LIMIT,
                                    command=fxcorepy.Constants.Commands.EDIT_ORDER,
                                    OFFER_ID=offer.offer_id,
                                    ACCOUNT_ID=Dict['FXCM']['str_account'],
                                    BUY_SELL=buy_sell,
                                    AMOUNT=int(dj.loc[0, 'tick_amount']),
                                    TRADE_ID=dj.loc[0, 'tick_id'],
                                    RATE=sl,
                                    ORDER_ID=dj.loc[0, 'order_stop_id']
                                )
                                resp = fx.send_request(request)
                            except Exception as e:
                                type_signal = type_signal + ' not working for ' + str(e)
                                pass

                else:
                    if (df.iloc[-5:-2]['macd'].mean() < df.iloc[-6:-3]['macd'].mean()
                        and abs(df.iloc[-5:-2]['Delta'].mean()) < abs(df.iloc[-6:-3]['Delta'].mean())
                        and (candle_2 < -0.5 or candle_3 < -0.5 or candle_4 < -0.5))\
                        or \
                            (df.iloc[-2]['BidLow']<df.iloc[-2]['kijun_avg']):
                        try:
                            type_signal = ' Buy : Adjust for MACD, delta, candles or kijun cross  ' + str(current_ratio)
                            sl = df.iloc[-3:-1]['BidLow'].min()-0.1*(df.iloc[-27:-1]['BidHigh'].max()-df.iloc[-27:-1]['BidLow'].min())
                            if sl >= price: sl=df.iloc[-1]['BidLow']
                            request = fx.create_order_request(
                                order_type=fxcorepy.Constants.Orders.LIMIT,
                                command=fxcorepy.Constants.Commands.EDIT_ORDER,
                                OFFER_ID=offer.offer_id,
                                ACCOUNT_ID=Dict['FXCM']['str_account'],
                                BUY_SELL=buy_sell,
                                AMOUNT=int(dj.loc[0, 'tick_amount']),
                                TRADE_ID=dj.loc[0, 'tick_id'],
                                RATE=sl,
                                ORDER_ID=dj.loc[0, 'order_stop_id']
                            )
                            resp = fx.send_request(request)
                        except Exception as e:
                            type_signal = type_signal + ' not working for ' + str(e)
                            pass
                    elif current_ratio > 3 and round(sl, 3) < round((open_price - open_sl) * 2 + open_price, 3):
                        try:
                            type_signal = ' Buy : Adjust for ratio ' + str(current_ratio)
                            sl = (open_price - open_sl) * 2 + open_price
                            if sl >= price: sl=df.iloc[-1]['BidLow']
                            request = fx.create_order_request(
                                order_type=fxcorepy.Constants.Orders.LIMIT,
                                command=fxcorepy.Constants.Commands.EDIT_ORDER,
                                OFFER_ID=offer.offer_id,
                                ACCOUNT_ID=Dict['FXCM']['str_account'],
                                BUY_SELL=buy_sell,
                                AMOUNT=int(dj.loc[0, 'tick_amount']),
                                TRADE_ID=dj.loc[0,'tick_id'],
                                RATE=sl,
                                ORDER_ID=dj.loc[0,'order_stop_id']
                            )
                            resp = fx.send_request(request)
                        except Exception as e:
                            type_signal = type_signal + ' not working for ' + str(e)
                            pass
                    elif current_ratio > 2.5 and round(sl, 3) < round((open_price - open_sl) * 1.5 + open_price, 3):
                        try:
                            type_signal = ' Buy : Adjust for ratio ' + str(current_ratio)
                            sl = (open_price - open_sl) * 1.5 + open_price
                            if sl >= price: sl=df.iloc[-1]['BidLow']
                            request = fx.create_order_request(
                                order_type=fxcorepy.Constants.Orders.LIMIT,
                                command=fxcorepy.Constants.Commands.EDIT_ORDER,
                                OFFER_ID=offer.offer_id,
                                ACCOUNT_ID=Dict['FXCM']['str_account'],
                                BUY_SELL=buy_sell,
                                AMOUNT=int(dj.loc[0, 'tick_amount']),
                                TRADE_ID=dj.loc[0,'tick_id'],
                                RATE=sl,
                                ORDER_ID=dj.loc[0,'order_stop_id']
                            )
                            resp = fx.send_request(request)
                        except Exception as e:
                            type_signal = type_signal + ' not working for ' + str(e)
                            pass
                    elif current_ratio > 2 and round(sl, 3) < round((open_price - open_sl) * 1 + open_price, 3):
                        try:
                            type_signal = ' Buy : Adjust for ratio ' + str(current_ratio)
                            sl = (open_price - open_sl) * 1 + open_price
                            if sl >= price: sl=df.iloc[-1]['BidLow']
                            request = fx.create_order_request(
                                order_type=fxcorepy.Constants.Orders.LIMIT,
                                command=fxcorepy.Constants.Commands.EDIT_ORDER,
                                OFFER_ID=offer.offer_id,
                                ACCOUNT_ID=Dict['FXCM']['str_account'],
                                BUY_SELL=buy_sell,
                                AMOUNT=int(dj.loc[0, 'tick_amount']),
                                TRADE_ID=dj.loc[0,'tick_id'],
                                RATE=sl,
                                ORDER_ID=dj.loc[0,'order_stop_id']
                            )
                            resp = fx.send_request(request)
                        except Exception as e:
                            type_signal = type_signal + ' not working for ' + str(e)
                            pass
                    elif current_ratio > 1.5 and round(sl, 3) < round((open_price - open_sl) * 0.5 + open_price, 3):
                        try:
                            type_signal = ' Buy : Adjust for ratio ' + str(current_ratio)
                            sl = (open_price - open_sl) * 0.5 + open_price
                            if sl >= price: sl=df.iloc[-1]['BidLow']
                            request = fx.create_order_request(
                                order_type=fxcorepy.Constants.Orders.LIMIT,
                                command=fxcorepy.Constants.Commands.EDIT_ORDER,
                                OFFER_ID=offer.offer_id,
                                ACCOUNT_ID=Dict['FXCM']['str_account'],
                                BUY_SELL=buy_sell,
                                AMOUNT=int(dj.loc[0, 'tick_amount']),
                                TRADE_ID=dj.loc[0,'tick_id'],
                                RATE=sl,
                                ORDER_ID=dj.loc[0,'order_stop_id']
                            )
                            resp = fx.send_request(request)
                        except Exception as e:
                            type_signal = type_signal + ' not working for ' + str(e)
                            pass
                    elif current_ratio > 1 and round(sl, 3) < round(open_price, 3):
                        try:
                            type_signal = ' Buy : Adjust for ratio ' + str(current_ratio)
                            sl = open_price
                            request = fx.create_order_request(
                                order_type=fxcorepy.Constants.Orders.LIMIT,
                                command=fxcorepy.Constants.Commands.EDIT_ORDER,
                                OFFER_ID=offer.offer_id,
                                ACCOUNT_ID=Dict['FXCM']['str_account'],
                                BUY_SELL=buy_sell,
                                AMOUNT=int(dj.loc[0, 'tick_amount']),
                                TRADE_ID=dj.loc[0,'tick_id'],
                                RATE=sl,
                                ORDER_ID=dj.loc[0,'order_stop_id']
                            )
                            resp = fx.send_request(request)
                        except Exception as e:
                            type_signal = type_signal + ' not working for ' + str(e)
                            pass
                    elif current_ratio > 0.5 and current_ratio < 1:
                        try:
                            type_signal = ' Buy : Adjust for ratio ' + str(current_ratio)
                            sl = open_price
                            if sl >= price: sl=df.iloc[-1]['BidLow']
                            request = fx.create_order_request(
                                order_type=fxcorepy.Constants.Orders.LIMIT,
                                command=fxcorepy.Constants.Commands.EDIT_ORDER,
                                OFFER_ID=offer.offer_id,
                                ACCOUNT_ID=Dict['FXCM']['str_account'],
                                BUY_SELL=buy_sell,
                                AMOUNT=int(dj.loc[0, 'tick_amount']),
                                TRADE_ID=dj.loc[0,'tick_id'],
                                RATE=sl,
                                ORDER_ID=dj.loc[0,'order_stop_id']
                            )
                            resp = fx.send_request(request)
                        except Exception as e:
                            type_signal = type_signal + ' not working for ' + str(e)
                            pass
            # if was sell
            if dj.loc[0,'tick_type'] == 'S':
                open_sl = df.iloc[-27:-2]['BidHigh'].max() + 0.1 * (
                            df.iloc[-27:-2]['BidHigh'].max() - df.iloc[-27:-2]['BidLow'].min())
                current_ratio = (open_price - price) / (open_sl - open_price)
                # signal cross macd sell
                if (df.iloc[-2]['macd'] > df.iloc[-3]['macd'] and df.iloc[-3]['macd'] > df.iloc[-4]['macd'] and current_ratio>0)\
                    or \
                        (df.iloc[-2]['signal'] < df.iloc[-2]['macd'] and candle_2>0.5 and current_ratio>0) \
                    or \
                        (df.iloc[-2]['BidHigh'] > df.iloc[-2]['tenkan_avg'] and current_ratio > 0 and df15.iloc[-2][
                            'tenkan_avg'] > df15.iloc[-2]['kijun_avg']) \
                    or \
                        (df.iloc[-2]['signal'] < df.iloc[-2]['macd'] and df.iloc[-3]['signal'] < df.iloc[-3]['macd']
                         and (candle_2>0.5 or candle_3>0.5) and current_ratio>0) \
                    or \
                        (df.iloc[-2]['macd'] > df.iloc[-3]['macd'] and (candle_2 > 0.5 or candle_3 > 0.5)
                         and df.iloc[-2]['BidHigh'] > df.iloc[-2]['tenkan_avg'] and current_ratio>0)\
                    or \
                        (df.iloc[-3:-2]['macd'].mean() > df.iloc[-4:-3]['macd'].mean() \
                        and abs(df.iloc[-3:-2]['Delta'].mean()) < abs(df.iloc[-4:-3]['Delta'].mean()) \
                        and df.iloc[-2]['signal'] < df.iloc[-2]['macd'] and (candle_2 > 0.5 or candle_3 > 0.5) \
                        and df.iloc[-2]['BidHigh'] > df.iloc[-2]['tenkan_avg'])\
                    or \
                        ((df.iloc[-2]['BidHigh'] - df.iloc[-2]['BidLow']) > (df.iloc[-3]['BidHigh'] - df.iloc[-3]['BidLow'])
                         and current_ratio>0 and candle_2 > 0 ) \
                    or \
                        (min(df.iloc[-7:-2]['rsi'])<30 and current_ratio > 0 and candle_2 > 0) \
                    or \
                        (df.iloc[-2]['macd'] > df.iloc[-3]['macd'] and abs(df.iloc[-2]['Delta']) < abs(
                            df.iloc[-3]['Delta']) and current_ratio > 0 and candle_2 > 0):
                    try:
                        type_signal = ' Sell : Close for Signal MACD ' + str(current_ratio)
                        request = fx.create_order_request(
                            order_type=fxcorepy.Constants.Orders.TRUE_MARKET_CLOSE,
                            OFFER_ID=offer.offer_id,
                            ACCOUNT_ID=Dict['FXCM']['str_account'],
                            BUY_SELL=buy_sell,
                            AMOUNT=int(dj.loc[0, 'tick_amount']),
                            TRADE_ID=dj.loc[0,'tick_id']
                        )
                        resp = fx.send_request(request)
                    except Exception as e:
                        type_signal = type_signal + ' not working for ' + str(e)
                        pass
                elif df.iloc[-2]['tenkan_avg'] == df.iloc[-3]['tenkan_avg'] and df.iloc[-3]['tenkan_avg'] == df.iloc[-4]['tenkan_avg'] \
                        and abs(df.iloc[-2]['Delta']) < abs(df.iloc[-3]['Delta']):
                    if current_ratio > 0:
                        try:
                            type_signal = ' Sell : Close Tenkan Flat ' + str(current_ratio)
                            request = fx.create_order_request(
                                order_type=fxcorepy.Constants.Orders.TRUE_MARKET_CLOSE,
                                OFFER_ID=offer.offer_id,
                                ACCOUNT_ID=Dict['FXCM']['str_account'],
                                BUY_SELL=buy_sell,
                                AMOUNT=int(dj.loc[0, 'tick_amount']),
                                TRADE_ID=dj.loc[0,'tick_id']
                            )
                            resp = fx.send_request(request)
                        except Exception as e:
                            type_signal = type_signal + ' not working for ' + str(e)
                            pass
                    else:
                        try:
                            type_signal = ' Sell : Adjust Tenkan Flat ' + str(current_ratio)
                            sl = df.iloc[-3:-1]['BidHigh'].max() + 0.1 * (
                                    df.iloc[-27:-1]['BidHigh'].max() - df.iloc[-27:-1]['BidLow'].min())
                            if sl <= price: sl = df.iloc[-1]['BidHigh']
                            request = fx.create_order_request(
                                order_type=fxcorepy.Constants.Orders.LIMIT,
                                command=fxcorepy.Constants.Commands.EDIT_ORDER,
                                OFFER_ID=offer.offer_id,
                                ACCOUNT_ID=Dict['FXCM']['str_account'],
                                BUY_SELL=buy_sell,
                                AMOUNT=int(dj.loc[0, 'tick_amount']),
                                TRADE_ID=dj.loc[0, 'tick_id'],
                                RATE=sl,
                                ORDER_ID=dj.loc[0, 'order_stop_id']
                            )
                            resp = fx.send_request(request)
                        except Exception as e:
                            type_signal = type_signal + ' not working for ' + str(e)
                            pass
                else:
                    if df.iloc[-5:-2]['macd'].mean() > df.iloc[-6:-3]['macd'].mean() and \
                        abs(df.iloc[-5:-2]['Delta'].mean()) > abs(df.iloc[-6:-3]['Delta'].mean()) and \
                            (candle_2 > 0.5 or candle_3 > 0.5 or candle_4 > 0.5)\
                        or \
                            (df.iloc[-2]['BidHigh']>df.iloc[-2]['kijun_avg']):
                        try:
                            type_signal = ' Sell : Adjust for MACD, delta, candles ' + str(current_ratio)
                            sl = df.iloc[-3:-1]['BidHigh'].max() + 0.1 * (
                                        df.iloc[-27:-1]['BidHigh'].max() - df.iloc[-27:-1]['BidLow'].min())
                            if sl <= price: sl=df.iloc[-1]['BidHigh']
                            request = fx.create_order_request(
                                order_type=fxcorepy.Constants.Orders.LIMIT,
                                command=fxcorepy.Constants.Commands.EDIT_ORDER,
                                OFFER_ID=offer.offer_id,
                                ACCOUNT_ID=Dict['FXCM']['str_account'],
                                BUY_SELL=buy_sell,
                                AMOUNT=int(dj.loc[0, 'tick_amount']),
                                TRADE_ID=dj.loc[0,'tick_id'],
                                RATE=sl,
                                ORDER_ID=dj.loc[0,'order_stop_id']
                            )
                            resp = fx.send_request(request)
                        except Exception as e:
                            type_signal = type_signal + ' not working for ' + str(e)
                            pass
                    elif current_ratio > 3 and round(sl, 3) >= round(open_price - 2 * (open_sl - open_price), 3):
                        try:
                            type_signal = ' Sell : Adjust for ratio ' + str(current_ratio)
                            sl = open_price - 2 * (open_sl - open_price)
                            if sl <= price: sl=df.iloc[-1]['BidHigh']
                            request = fx.create_order_request(
                                order_type=fxcorepy.Constants.Orders.LIMIT,
                                command=fxcorepy.Constants.Commands.EDIT_ORDER,
                                OFFER_ID=offer.offer_id,
                                ACCOUNT_ID=Dict['FXCM']['str_account'],
                                BUY_SELL=buy_sell,
                                AMOUNT=int(dj.loc[0, 'tick_amount']),
                                TRADE_ID=dj.loc[0,'tick_id'],
                                RATE=sl,
                                ORDER_ID=dj.loc[0,'order_stop_id']
                            )
                            resp = fx.send_request(request)
                        except Exception as e:
                            type_signal = type_signal + ' not working for ' + str(e)
                            pass
                    elif current_ratio > 2.5 and round(sl, 3) > round(open_price - 1.5 * (open_sl - open_price), 3):
                        try:
                            type_signal = ' Sell : Adjust for ratio ' + str(current_ratio)
                            sl = open_price - 1.5 * (open_sl - open_price)
                            if sl <= price: sl=df.iloc[-1]['BidHigh']
                            request = fx.create_order_request(
                                order_type=fxcorepy.Constants.Orders.LIMIT,
                                command=fxcorepy.Constants.Commands.EDIT_ORDER,
                                OFFER_ID=offer.offer_id,
                                ACCOUNT_ID=Dict['FXCM']['str_account'],
                                BUY_SELL=buy_sell,
                                AMOUNT=int(dj.loc[0, 'tick_amount']),
                                TRADE_ID=dj.loc[0,'tick_id'],
                                RATE=sl,
                                ORDER_ID=dj.loc[0,'order_stop_id']
                            )
                            resp = fx.send_request(request)
                        except Exception as e:
                            type_signal = type_signal + ' not working for ' + str(e)
                            pass
                    elif current_ratio > 2 and round(sl, 3) > round(open_price - 1 * (open_sl - open_price), 3):
                        try:
                            type_signal = ' Sell : Adjust for ratio ' + str(current_ratio)
                            sl = open_price - 1 * (open_sl - open_price)
                            if sl <= price: sl=df.iloc[-1]['BidHigh']
                            request = fx.create_order_request(
                                order_type=fxcorepy.Constants.Orders.LIMIT,
                                command=fxcorepy.Constants.Commands.EDIT_ORDER,
                                OFFER_ID=offer.offer_id,
                                ACCOUNT_ID=Dict['FXCM']['str_account'],
                                BUY_SELL=buy_sell,
                                AMOUNT=int(dj.loc[0, 'tick_amount']),
                                TRADE_ID=dj.loc[0,'tick_id'],
                                RATE=sl,
                                ORDER_ID=dj.loc[0,'order_stop_id']
                            )
                            resp = fx.send_request(request)
                        except Exception as e:
                            type_signal = type_signal + ' not working for ' + str(e)
                            pass
                    elif current_ratio > 1.5 and round(sl, 3) > round(open_price - 0.5 * (open_sl - open_price), 3):
                        try:
                            type_signal = ' Sell : Adjust for ratio ' + str(current_ratio)
                            sl = open_price - 0.5 * (open_sl - open_price)
                            if sl <= price: sl=df.iloc[-1]['BidHigh']
                            request = fx.create_order_request(
                                order_type=fxcorepy.Constants.Orders.LIMIT,
                                command=fxcorepy.Constants.Commands.EDIT_ORDER,
                                OFFER_ID=offer.offer_id,
                                ACCOUNT_ID=Dict['FXCM']['str_account'],
                                BUY_SELL=buy_sell,
                                AMOUNT=int(dj.loc[0, 'tick_amount']),
                                TRADE_ID=dj.loc[0,'tick_id'],
                                RATE=sl,
                                ORDER_ID=dj.loc[0,'order_stop_id']
                            )
                            resp = fx.send_request(request)
                        except Exception as e:
                            type_signal = type_signal + ' not working for ' + str(e)
                            pass
                    elif current_ratio > 1 and round(sl, 3) > round(open_price, 3):
                        try:
                            type_signal = ' Sell : Adjust for ratio ' + str(current_ratio)
                            sl = open_price
                            request = fx.create_order_request(
                                order_type=fxcorepy.Constants.Orders.LIMIT,
                                command=fxcorepy.Constants.Commands.EDIT_ORDER,
                                OFFER_ID=offer.offer_id,
                                ACCOUNT_ID=Dict['FXCM']['str_account'],
                                BUY_SELL=buy_sell,
                                AMOUNT=int(dj.loc[0, 'tick_amount']),
                                TRADE_ID=dj.loc[0,'tick_id'],
                                RATE=sl,
                                ORDER_ID=dj.loc[0,'order_stop_id']
                            )
                            resp = fx.send_request(request)
                        except Exception as e:
                            type_signal = type_signal + ' not working for ' + str(e)
                            pass
                    elif current_ratio > 0.5 and current_ratio < 1:
                        try:
                            type_signal = ' Sell : Adjust for ratio ' + str(current_ratio)
                            sl = open_price
                            if sl <= price: sl=df.iloc[-1]['BidHigh']
                            request = fx.create_order_request(
                                order_type=fxcorepy.Constants.Orders.LIMIT,
                                command=fxcorepy.Constants.Commands.EDIT_ORDER,
                                OFFER_ID=offer.offer_id,
                                ACCOUNT_ID=Dict['FXCM']['str_account'],
                                BUY_SELL=buy_sell,
                                AMOUNT=int(dj.loc[0, 'tick_amount']),
                                TRADE_ID=dj.loc[0,'tick_id'],
                                RATE=sl,
                                ORDER_ID=dj.loc[0,'order_stop_id']
                            )
                            resp = fx.send_request(request)
                        except Exception as e:
                            type_signal = type_signal + ' not working for ' + str(e)
                            pass

            # if now in range --> Sell
    return df, tick, type_signal, open_rev_index, box_def, high_box, low_box, tp, sl

def main():
    currentDateAndTime = datetime.now()
    currentday = currentDateAndTime.weekday()
    currenthour = currentDateAndTime.hour

    #if True==True:
    #if not ((currentday == 5 and currenthour > 2) or (currentday == 6 and currenthour < 20)):
    print(str(currentDateAndTime.strftime("%H:%M:%S")))
    with ForexConnect() as fx:
        #try:
        fx.login(Dict['FXCM']['str_user_i_d'], Dict['FXCM']['str_password'], Dict['FXCM']['str_url'],
                 Dict['FXCM']['str_connection'], Dict['FXCM']['str_session_id'], Dict['FXCM']['str_pin'],
                 session_status_callback=session_status_changed)
        login_rules = fx.login_rules
        trading_settings_provider = login_rules.trading_settings_provider
        for l1 in range(0, len(FX)):
            #hours = fx.getTradingHours(FX[l1])
            if trading_settings_provider.get_market_status(FX[l1])==fxcorepy.O2GMarketStatus.MARKET_STATUS_OPEN:
                print(FX[l1])
                #H1
                df = pd.DataFrame(fx.get_history(FX[l1], 'm15', Dict['indicators']['sd'], Dict['indicators']['ed']))
                if len(df) < 7*5*3:
                    df = pd.DataFrame(
                        fx.get_history(FX[l1], 'm15', datetime.now() - relativedelta(weeks=6), Dict['indicators']['ed']))
                # If there is history data
                # Add all the indicators needed
                df = indicators(df)
                #M15
                df15 = pd.DataFrame(fx.get_history(FX[l1], 'm15', Dict['indicators']['sd'], Dict['indicators']['ed']))
                df15 = indicators(df15)
                #d1
                dfd1 = pd.DataFrame(fx.get_history(FX[l1], 'D1', Dict['indicators']['sd'], Dict['indicators']['ed']))
                dfd1 = indicators(dfd1)
                # Check the current open positions
                open_pos_status, dj = check_trades(FX[l1], fx)
                # if status not open then check if to open
                if open_pos_status == 'No':
                    df, tick, type_signal, index, box_def, high_box, low_box, tp, sl = \
                        open_trade(df, fx, FX[l1],trading_settings_provider,dj,dfd1)
                # if status is open then check if to close
                elif open_pos_status == 'Yes':
                    df, tick, type_signal, index, box_def, high_box, low_box, tp, sl = \
                        close_trade(df, fx, FX[l1], dj,df15)
                df_plot(df, tick, type_signal, index, box_def, high_box, low_box, tp, sl)

        # except Exception as e:
        #       print("Exception: " + str(e))
        try:
            fx.logout()
        except Exception as e:
            print("Exception: " + str(e))

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

    # schedule.every(1).hours.at(":03").do(main)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)

#For next improvement
    #How    to    use    this    fxcorepy.Constants.SystemProperties.END_TRADING_DAY ?
