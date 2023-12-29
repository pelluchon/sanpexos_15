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

            #   2:{
            #       'hour_open': 0,#opening time in UTC (for midnight put 24)
            #       'hour_close':6,#closing time in UTC
            #       'FX':[
            # 'CSL.au', 'CBA.au', 'BHP.au', 'WBC.au', 'NAB.au', 'ANZ.au',
            # 'WOW.au', 'WES.au', 'FMG.au', 'MQG.au', 'TLS.au', 'RIO.au',
            # 'GMG.au', 'WPL.au', 'NCM.au', 'COL.au', 'ALL.au', 'A2M.au',
            # 'REA.au', 'XRO.au', 'QAN.au', 'Z1P.au']},

            #   3:{
            #       'hour_open': 7,#opening time in UTC (for midnight put 24)
            #       'hour_close':15,#closing time in UTC
            #       'FX':['ADS.de', 'ALV.de', 'BAS.de', 'BAYN.de', 'BMW.de',
            # 'DB1.de', 'DBK.de', 'DPW.de', 'DTE.de', 'EOAN.de', 'IFX.de',
            # 'LHA.de', 'MRK.de', 'RWE.de', 'SAP.de', 'SIE.de', 'TUI1.de',
            # 'VOW.de','VNA.de','ENR.de','CBK.de', 'DHER.de']},

            #   4:{
            #       'hour_open': 14,#opening time in UTC (for midnight put 24)
            #       'hour_close':20,#closing time in UTC
            #       'FX':['FVRR.us', 'SPOT.us', 'MARA.us', 'BTBT.us',
            # 'BITF.us', 'WISH.us', 'RIVN.us', 'WE.us', 'JD.us', 'PDD.us',
            # 'TME.us', 'WB.us', 'BILI.us', 'NVDA.us', 'AMD.us', 'DADA.us',
            # 'PTON.us', 'MRNA.us', 'NIO.us', 'CCL.us', 'ABNB.us', 'DASH.us',
            # 'AMC.us', 'BNGO.us', 'FCEL.us', 'GME.us', 'PENN.us', 'PLTR.us',
            # 'PLUG.us', 'PYPL.us', 'SNAP.us', 'SNOW.us', 'SPCE.us', 'XPEV.us',
            # 'SONY.us','BA.us', 'BAC.us', 'BRKB.us', 'C.us', 'CRM.us',
            # 'DIS.us', 'F.us', 'JPM.us', 'KO.us', 'MA.us', 'MCD.us',
            # 'PFE.us', 'PG.us', 'SE.us', 'T.us', 'TGT.us', 'V.us', 'XOM.us',
            # 'AAPL.us', 'AMZN.us', 'AskU.us', 'GOOG.us', 'INTC.us', 'MSFT.us',
            # 'SBUX.us','BABA.us', 'DAL.us',
            # 'NFLX.us', 'TSLA.us','SQ.us', 'LYFT.us', 'UAL.us', 'DKNG.us', 'SHOP.us', 'BYND.us',
            # 'UBER.us', 'ZM.us', 'LCID.us', 'HOOD.us', 'CRWD.us', 'BEKE.us',
            # 'CPNG.us', 'NET.us', 'RBLX.us', 'COIN.us']},

            #   5:{
            #       'hour_open': 2,#opening time in UTC (for midnight put 24)
            #       'hour_close':8,#closing time in UTC
            #       'FX':['TENC.hk', 'MEIT.hk', 'BYDC.hk', 'XIAO.hk', 'BABA.hk',
            # 'AIA.hk', 'HSBC.hk', 'WUXI.hk', 'HKEX.hk', 'GELY.hk', 'JD.hk',
            # 'NETE.hk', 'PING.hk', 'SMIC.hk', 'SBIO.hk', 'GALA.hk', 'KIDE.hk',
            # 'ALIH.hk', 'ICBC.hk', 'FLAT.hk', 'KSOF.hk', 'SMOO.hk', 'SUNN.hk',
            # 'BYDE.hk','AskU.hk']},

            #   6:{
            #       'hour_open': 7,#opening time in UTC (for midnight put 24)
            #       'hour_close':15,#closing time in UTC
            #       'FX':['ACA.fr', 'AI.fr', 'ALO.fr', 'BN.fr', 'BNP.fr', 'CA.fr',
            # 'DG.fr', 'AIR.fr', 'ORA.fr', 'GLE.fr', 'MC.fr', 'ML.fr', 'OR.fr',
            # 'RNO.fr', 'SAN.fr', 'SGO.fr', 'SU.fr', 'VIE.fr', 'VIV.fr','TTE.fr', 'ENGI.fr','STM.fr', 'STLA.fr']},

            #   7:{
            #       'hour_open': 7,#opening time in UTC (for midnight put 24)
            #       'hour_close':15,#closing time in UTC
            #       'FX':['AV.uk', 'AZN.uk', 'BA.uk', 'BARC.uk', 'BATS.uk',
            # 'BP.uk', 'GSK.uk', 'HSBA.uk', 'IAG.uk', 'LGEN.uk', 'LLOY.uk',
            # 'RR.uk', 'STAN.uk', 'TSCO.uk', 'VOD.uk','GLEN.uk',
            #  'BT.A.uk', 'NWG.uk','TW.uk', 'MRO.uk', 'MNG.uk', 'ROO.uk']},

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
        # try:
        #    sendemail(attach='mean_reversion.png', subject_mail=str(tick), body_mail=str(tick) + 'mean_reversion')
        # except Exception as e:
        #    print("issue with mails for " + tick)
        #    print("Exception: " + str(e))
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
    candle_2 = (df.iloc[-2]['AskClose'] - df.iloc[-2]['AskOpen']) / (df.iloc[-2]['AskHigh'] - df.iloc[-2]['AskLow'])
    margin = abs(0.2 * (np.nanmax(df.iloc[-27:-2]['AskHigh']) - np.nanmin(df.iloc[-27:-2]['AskLow'])))

    # peak of RSI
    for i in range(-2, -len(df) + 1, -1):
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

    # BUY
    # if index of under 31 is the highest, means the latest down (under 31) is after the last high
    if index_peak < 0 \
            and df.iloc[-7:-2]['ci'].mean() < 45\
            and df.iloc[-7:-2]['rsi'].mean() < 65 \
            and df.iloc[index_peak:-2]['rsi'].mean() < 65\
            and df.iloc[-2]['AskClose'] > df.iloc[index_peak:-2]['AskClose'].mean() \
            and df.iloc[-2]['AskClose'] > df.iloc[-2]['tenkan_avg'] \
            and df.iloc[-2]['AskClose'] > df.iloc[-2]['kijun_avg'] \
            and df.iloc[-2]['AskClose'] > max(df.iloc[index_peak]['senkou_a'], df.iloc[index_peak]['senkou_b']) \
            and df.iloc[-2]['tenkan_avg'] > df.iloc[-2]['kijun_avg'] \
            and df.iloc[index_peak]['kijun_avg'] < min(df.iloc[index_peak]['senkou_a'], df.iloc[index_peak]['senkou_b']) \
            and df.iloc[index_peak]['tenkan_avg'] < df.iloc[index_peak]['kijun_avg'] \
            and df.iloc[index_peak - 27]['chikou'] < df.iloc[index_peak - 27]['AskHigh'] \
            and df.iloc[index_peak - 27]['chikou'] < df.iloc[index_peak - 27]['tenkan_avg'] \
            and df.iloc[index_peak - 27]['chikou'] < df.iloc[index_peak - 27]['kijun_avg']:
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
    # SELL
    elif index_peak < 0 \
            and df.iloc[-7:-2]['ci'].mean() < 45\
            and df.iloc[-7:-2]['rsi'].mean() > 35 \
            and df.iloc[index_peak:-2]['rsi'].mean() > 35 \
            and df.iloc[-2]['AskClose'] < df.iloc[index_peak:-2]['AskClose'].mean() \
            and df.iloc[-2]['AskClose'] < df.iloc[-2]['tenkan_avg'] \
            and df.iloc[-2]['AskClose'] < df.iloc[-2]['kijun_avg'] \
            and df.iloc[-2]['AskClose'] < min(df.iloc[index_peak]['senkou_a'], df.iloc[index_peak]['senkou_b']) \
            and df.iloc[-2]['tenkan_avg'] < df.iloc[-2]['kijun_avg'] \
            and df.iloc[index_peak]['kijun_avg'] > max(df.iloc[index_peak]['senkou_a'], df.iloc[index_peak]['senkou_b']) \
            and df.iloc[index_peak]['tenkan_avg'] > df.iloc[index_peak]['kijun_avg'] \
            and df.iloc[index_peak - 27]['chikou'] > df.iloc[index_peak - 27]['AskHigh'] \
            and df.iloc[index_peak - 27]['chikou'] > df.iloc[index_peak - 27]['tenkan_avg'] \
            and df.iloc[index_peak - 27]['chikou'] > df.iloc[index_peak - 27]['kijun_avg']:
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
        [len(df) - df.index[df['Date'].dt.strftime("%m%d%Y%H") == dj.loc[0, 'tick_time'].strftime("%m%d%Y%H")]][0][0]
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
            # END OF DAY CONDITIONS
            # if l0 == 1 and datetime.now().weekday() == Dict['instrument'][l0]['day_close'] and \
            #         int(datetime.now().strftime("%H")) == Dict['instrument'][l0]['hour_close'] - 1:
            #     try:
            #         type_signal = ' Buy: Adjust for End of Day ' + str(current_ratio)
            #         sl = min(df.iloc[-open_rev_index:-1]['AskLow']) - margin
            #         request = fx.create_order_request(
            #             order_type=fxcorepy.Constants.Orders.LIMIT,
            #             command=fxcorepy.Constants.Commands.EDIT_ORDER,
            #             OFFER_ID=offer.offer_id,
            #             ACCOUNT_ID=Dict['FXCM']['str_account'],
            #             BUY_SELL=buy_sell,
            #             AMOUNT=int(dj.loc[0, 'tick_amount']),
            #             TRADE_ID=dj.loc[0, 'tick_id'],
            #             RATE=df.iloc[-27:-2]['AskLow'].min()-margin,
            #             RATE_LIMIT=df.iloc[-27*2:-2]['AskHigh'].max()+margin,
            #             ORDER_ID=dj.loc[0, 'order_stop_id']
            #         )
            #         resp = fx.send_request(request)
            #     except Exception as e:
            #         type_signal = type_signal + ' not working for ' + str(e)
            #         pass
            # elif l0 > 1 and int(datetime.now().strftime("%H")) == Dict['instrument'][l0]['hour_close'] - 1:
            #     try:
            #         type_signal = ' Buy: Adjust for End of Day ' + str(current_ratio)
            #         sl = min(df.iloc[-open_rev_index:-1]['AskLow']) - margin
            #         request = fx.create_order_request(
            #             order_type=fxcorepy.Constants.Orders.LIMIT,
            #             command=fxcorepy.Constants.Commands.EDIT_ORDER,
            #             OFFER_ID=offer.offer_id,
            #             ACCOUNT_ID=Dict['FXCM']['str_account'],
            #             BUY_SELL=buy_sell,
            #             AMOUNT=int(dj.loc[0, 'tick_amount']),
            #             TRADE_ID=dj.loc[0, 'tick_id'],
            #             RATE=df.iloc[-27:-2]['AskLow'].min()-margin,
            #             RATE_LIMIT=df.iloc[-27*2:-2]['AskHigh'].max()+margin,
            #             ORDER_ID=dj.loc[0, 'order_stop_id']
            #         )
            #         resp = fx.send_request(request)
            #     except Exception as e:
            #         type_signal = type_signal + ' not working for ' + str(e)
            #         pass
            if df.iloc[-2]['AskClose'] < df.iloc[-2]['tenkan_avg'] and candle_2 < -0.25 \
                and current_ratio > 0 and df.iloc[-open_rev_index:-2][df['rsi'] > 65].size > 0 and df.iloc[-2]['tenkan_avg'] < df.iloc[-3]['tenkan_avg'] and \
                ((abs(df.iloc[-2]['macd']) < abs(df.iloc[-2]['signal'])) or (df.iloc[-4:-2]['macd'].mean() < df.iloc[-6:-4]['macd'].mean())):
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
            if (df.iloc[-2]['kijun_avg'] - margin) > open_price and (df.iloc[-2]['kijun_avg'] - margin) > df.iloc[-2]['AskClose'] and current_ratio > 0:
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
                # if df.iloc[-2]['rsi'] >= 40 and df.iloc[-4:-2]['rsi'].mean() < df.iloc[-5:-3]['rsi'].mean() and \
                #         df.iloc[-2]['AskClose'] < df.iloc[-2]['tenkan_avg'] \
                #         and df.iloc[-open_rev_index:-2]['AskClose'].max()>df.iloc[-open_rev_index:-2]['tenkan_avg'].max() \
                #         and df.iloc[-2]['tenkan_avg'] < df.iloc[-2]['kijun_avg'] \
                #         and current_ratio>0 and open_rev_index>7:
                #     try:
                #         type_signal = ' Buy : Close for Signal over macd ' + str(current_ratio)
                #         request = fx.create_order_request(
                #             order_type=fxcorepy.Constants.Orders.TRUE_MARKET_CLOSE,
                #             OFFER_ID=offer.offer_id,
                #             ACCOUNT_ID=Dict['FXCM']['str_account'],
                #             BUY_SELL=buy_sell,
                #             AMOUNT=int(dj.loc[0, 'tick_amount']),
                #             TRADE_ID=dj.loc[0, 'tick_id']
                #         )
                #         resp = fx.send_request(request)
                #     except Exception as e:
                #         type_signal = type_signal + ' not working for ' + str(e)
                #         pass
        # if was sell
        if dj.loc[0, 'tick_type'] == 'S':
            current_ratio = (open_price - price) / (df.iloc[-open_rev_index:-2]['AskHigh'].max() - open_price)
            # END OF DAY CONDITIONS
            # if l0 == 1 and datetime.now().weekday() == Dict['instrument'][l0]['day_close'] and \
            #         int(datetime.now().strftime("%H")) == Dict['instrument'][l0]['hour_close'] - 1:
            #     try:
            #         type_signal = ' Buy: Adjust for End of Day ' + str(current_ratio)
            #         request = fx.create_order_request(
            #             order_type=fxcorepy.Constants.Orders.LIMIT,
            #             command=fxcorepy.Constants.Commands.EDIT_ORDER,
            #             OFFER_ID=offer.offer_id,
            #             ACCOUNT_ID=Dict['FXCM']['str_account'],
            #             BUY_SELL=buy_sell,
            #             AMOUNT=int(dj.loc[0, 'tick_amount']),
            #             TRADE_ID=dj.loc[0, 'tick_id'],
            #             RATE=df.iloc[-27:-2]['AskHigh'].max()+margin,
            #             RATE_LIMIT=df.iloc[-27*2:-2]['AskLow'].min()-margin,
            #             ORDER_ID=dj.loc[0, 'order_stop_id']
            #         )
            #         resp = fx.send_request(request)
            #     except Exception as e:
            #         type_signal = type_signal + ' not working for ' + str(e)
            #         pass
            # elif l0 > 1 and int(datetime.now().strftime("%H")) == Dict['instrument'][l0]['hour_close'] - 1:
            #     try:
            #         type_signal = ' Buy: Adjust for End of Day ' + str(current_ratio)
            #         sl = max(df.iloc[-open_rev_index:-1]['AskHigh']) + margin
            #         request = fx.create_order_request(
            #             order_type=fxcorepy.Constants.Orders.LIMIT,
            #             command=fxcorepy.Constants.Commands.EDIT_ORDER,
            #             OFFER_ID=offer.offer_id,
            #             ACCOUNT_ID=Dict['FXCM']['str_account'],
            #             BUY_SELL=buy_sell,
            #             AMOUNT=int(dj.loc[0, 'tick_amount']),
            #             TRADE_ID=dj.loc[0, 'tick_id'],
            #             RATE=df.iloc[-27:-2]['AskHigh'].max()+margin,
            #             RATE_LIMIT=df.iloc[-27*2:-2]['AskLow'].min()-margin,
            #             ORDER_ID=dj.loc[0, 'order_stop_id']
            #         )
            #         resp = fx.send_request(request)
            #     except Exception as e:
            #         type_signal = type_signal + ' not working for ' + str(e)
            #         pass

            if df.iloc[-2]['AskClose'] > df.iloc[-2]['tenkan_avg'] and candle_2 > 0.25 \
                and current_ratio > 0 and df.iloc[-open_rev_index:-2][df['rsi'] < 35].size > 0 \
                and df.iloc[-2]['tenkan_avg'] > df.iloc[-3]['tenkan_avg'] \
                and ((abs(df.iloc[-2]['macd']) < abs(df.iloc[-2]['signal'])) or (df.iloc[-4:-2]['macd'].mean() > df.iloc[-6:-4]['macd'].mean())):
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
            if (df.iloc[-2]['kijun_avg'] + margin) < open_price and (df.iloc[-2]['kijun_avg'] + margin) < df.iloc[-2]['AskClose'] and current_ratio > 0:
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
                and df.iloc[-2]['AskClose'] > df.iloc[-2]['tenkan_avg']\
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
            # if df.iloc[-7:-2]['rsi'].mean() > df.iloc[-8:-3]['rsi'].mean() \
            #     and df.iloc[-5:-2]['tenkan_avg'].mean() > df.iloc[-6:-3]['tenkan_avg'].mean() \
            #     and df.iloc[-2]['tenkan_avg'] >= df.iloc[-2]['kijun_avg'] \
            #     and candle_2 > 0.25\
            #     and df.iloc[-5:-2]['kijun_avg'].mean() > df.iloc[-6:-3]['kijun_avg'].mean()\
            #     and df.iloc[-2]['AskHigh'] > df.iloc[-open_rev_index:-2]['AskHigh'].max():
            #     try:
            #         type_signal = ' Sell : Adjust for wrong direction ' + str(current_ratio)
            #         sl = df.iloc[-2]['kijun_avg'] + margin
            #         tp = df.iloc[-open_rev_index:-2]['AskLow'].min() - margin
            #         request = fx.create_order_request(
            #             order_type=fxcorepy.Constants.Orders.LIMIT,
            #             command=fxcorepy.Constants.Commands.CREATE_ORDER,
            #             OFFER_ID=offer.offer_id,
            #             ACCOUNT_ID=Dict['FXCM']['str_account'],
            #             BUY_SELL=buy_sell,
            #             AMOUNT=int(dj.loc[0, 'tick_amount']),
            #             TRADE_ID=dj.loc[0, 'tick_id'],
            #             RATE=sl,
            #             RATE_LIMIT=tp,
            #         )
            #         resp = fx.send_request(request)
            #     except Exception as e:
            #         type_signal = type_signal + ' not working for ' + str(e)
            #         pass
            # if df.iloc[-2]['rsi'] <= 60 and df.iloc[-4:-2]['rsi'].mean() > df.iloc[-5:-3]['rsi'].mean() and \
            #         df.iloc[-2]['AskClose'] > df.iloc[-2]['tenkan_avg'] \
            #         and df.iloc[-open_rev_index:-2]['AskClose'].min()<df.iloc[-open_rev_index:-2]['tenkan_avg'].min() \
            #         and df.iloc[-2]['tenkan_avg'] > df.iloc[-2]['kijun_avg'] \
            #         and current_ratio>0 and open_rev_index>7:
            #     try:
            #         type_signal = ' Sell : Close for AskClose crossing tenkan ' + str(current_ratio)
            #         request = fx.create_order_request(
            #             order_type=fxcorepy.Constants.Orders.TRUE_MARKET_CLOSE,
            #             OFFER_ID=offer.offer_id,
            #             ACCOUNT_ID=Dict['FXCM']['str_account'],
            #             BUY_SELL=buy_sell,
            #             AMOUNT=int(dj.loc[0, 'tick_amount']),
            #             TRADE_ID=dj.loc[0, 'tick_id']
            #         )
            #         resp = fx.send_request(request)
            #     except Exception as e:
            #         type_signal = type_signal + ' not working for ' + str(e)
            #         pass

    return df, type_signal, open_rev_index, box_def, high_box, low_box, tp, sl, index_peak


def rsi_algorithm(data, tick):
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

    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Prepare the data
    features = data[['AskOpen', 'AskHigh', 'AskLow', 'AskClose', 'Volume']]
    labels = tick

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the machine learning model with hyperparameter tuning
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    cross_val_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)  # Cross-validation
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    print(f"Cross-Validation Scores: {cross_val_scores}")

    # try:
    #     sendemail(attach='rsi_algorithm.png', subject_mail=str(tick), body_mail=str(tick) + 'rsi_algorithm.png')
    # except Exception as e:
    #    print("issue with mails for " + tick)
    #    print("Exception: " + str(e))


def kmeans(data, tick):
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    # This code first loads candle stick data, calculates MACD, RSI, and Ichimoku indicators, and combines and standardizes the data. Then, it performs K-means clustering on the standardized data and assigns cluster labels to the data. Finally, it makes trading decisions based on the cluster labels.
    #
    # Please note that this is just a basic example of how to use the K-means algorithm for trading. In practice, you would need to develop a more sophisticated trading system that takes into account other factors, such as market conditions and risk management.

    def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
        """Calcualtes the MACD indicator."""
        # Calculate the MACD
        macd = pd.Series(close.ewm(span=fastperiod, min_periods=fastperiod - 1).mean(), name='MACD_Fast')
        ema_slow = pd.Series(close.ewm(span=slowperiod, min_periods=slowperiod - 1).mean(), name='MACD_Slow')
        macd_diff = macd - ema_slow
        ema_signal = pd.Series(macd_diff.ewm(span=signalperiod, min_periods=signalperiod - 1).mean(),
                               name='MACD_Signal')

        # Calculate the MACD histogram
        macd_hist = macd_diff - ema_signal

        return macd, macd_diff, macd_signal, macd_hist

    def RelativeStrengthIndex(close, timeperiod=14):
        """Calculates the Relative Strength Index (RSI) indicator."""
        # Calculate the difference between the close price and the maximum price
        up = close.diff(1)
        up = up[up > 0]
        down = -close.diff(1)
        down = down[down > 0]

        # Calculate the average up and down price movements
        up_avg = up.ewm(span=timeperiod, min_periods=timeperiod - 1).mean()
        down_avg = down.ewm(span=timeperiod, min_periods=timeperiod - 1).mean()

        # Calculate the relative strength index
        relative_strength = up_avg / down_avg
        rsi = 100 - (100 / (1 + relative_strength))

        return rsi

    def IchimokuIndicator(open, high, low, close, fastperiod=9, slowperiod=26, signalperiod=5,
                          conversionlineperiod=9, ):
        """Calculates the Ichimoku Kinko Hyo (Ichimoku) indicator."""

        # Calculate the Tenkan-sen (Conversion Line)
        tenkan_sen = (high + low) / 2
        tenkan_sen = tenkan_sen.ewm(span=conversionlineperiod, min_periods=conversionlineperiod - 1).mean()

        # Calculate the Kijun-sen (Base Line)
        kijun_sen = (high + low) / 2
        kijun_sen = kijun_sen.ewm(span=slowperiod, min_periods=slowperiod - 1).mean()

        # Calculate the Senkou Span A (Leading Span A)
        senkou_span_a_a = (tenkan_sen + kijun_sen) / 2
        senkou_span_a = senkou_span_a_a.ewm(span=signalperiod, min_periods=signalperiod - 1).mean()

        # Calculate the Senkou Span B (Leading Span B)
        senkou_span_b_a = high.ewm(span=52, min_periods=52 - 1).mean()
        senkou_span_b = senkou_span_b_a.shift(26)

        # Calculate the Chikou Span (Lagging Span)
        chikou_span = close.shift(9)

        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

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

    # Calculate MACD
    macd, macd_diff, macd_signal, macd_hist = MACD(data['AskClose'], fastperiod=12, slowperiod=26, signalperiod=9)
    #macd_diff = macd.MACDDiff()
    #macd_signal = macd.MACDSignal()

    # Calculate RSI
    rsi = RelativeStrengthIndex(data['AskClose'], timeperiod=14)
    rsi_values = rsi.RSI()

    # Calculate Ichimoku
    ichimoku = IchimokuIndicator(
        data['AskOpen'], data['AskHigh'], data['AskLow'], data['AskClose']
    )
    i_tenkan_sen = ichimoku.TenkanSen()
    i_kijun_sen = ichimoku.KijunSen()
    i_senkou_span_a = ichimoku.SenkouSpanA()
    i_senkou_span_b = ichimoku.SenkouSpanB()
    i_chikou_span = ichimoku.ChikouSpan()

    # Combine and standardize the data
    data = np.column_stack(
        [data['AskClose'], macd_diff, rsi_values, i_tenkan_sen, i_kijun_sen, i_senkou_span_a, i_senkou_span_b,
         i_chikou_span])
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)

    # Assign cluster labels to the data
    cluster_labels = kmeans.labels_

    # Make trading decisions based on cluster labels
    for i in range(len(cluster_labels)):
        if cluster_labels[i] == 0:
            # Buy
            print('Buy signal')
        elif cluster_labels[i] == 1:
            # Hold
            print('Hold signal')
        else:
            # Sell
            print('Sell signal')

    # Create a figure and axes
    fig, axes = plt.subplots(5, 1)

    # Plot candles
    axes[0].plot(data['Date'], data['AskClose'], color='blue', label='Close')
    axes[0].legend()

    # Plot MACD
    axes[1].plot(data['Date'], macd_diff, color='orange', label='MACD Diff')
    axes[1].plot(data['Date'], macd_signal, color='green', label='MACD Signal')
    axes[1].legend()

    # Plot RSI
    axes[2].plot(data['Date'], rsi_values, color='red', label='RSI')
    axes[2].legend()

    # Plot Ichimoku
    axes[3].plot(data['Date'], i_tenkan_sen, color='purple', label='Tenkan-sen')
    axes[3].plot(data['Date'], i_kijun_sen, color='cyan', label='Kijun-sen')
    axes[3].plot(data['Date'], i_senkou_span_a, color='olive', label='Senkou Span A')
    axes[3].plot(data['Date'], i_senkou_span_b, color='pink', label='Senkou Span B')
    axes[3].plot(data['Date'], i_chikou_span, color='gray', label='Chikou Span')
    axes[3].legend()

    # Plot K-means predictions
    axes[4].plot(data['Date'], cluster_labels, color='black', label='K-means Predictions')
    axes[4].legend()

    # Set the figure title and labels
    fig.suptitle('Candles, MACD, RSI, Ichimoku, and K-means Predictions')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Close Price')
    axes[1].set_ylabel('MACD Values')

    # plt.show()
    fig.savefig('k-means.png')
    try:
        sendemail(attach='filename.png', subject_mail=str(tick), body_mail=str(tick) + "k-means")
    except Exception as e:
        print("issue with mails for " + tick)
        print("Exception: " + str(e))
    plt.close()

def main():
    print(str(datetime.now().strftime("%H:%M:%S")))
    # print('launch close')
    # close.main()
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
                    # Check the current open positions
                    open_pos_status, dj = check_trades(FX[l1], fx)
                    # if status not open then check if to open
                    if open_pos_status == 'No':
                        #if df.iloc[-2]['AskHigh'] + margin > df.iloc[-3]['AskLow']:
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
                        kmeans(df,FX[l1])
                        #rsi_algorithm(df,FX[l1])
            # except Exception as e:
            #     print("Exception: " + str(e))
            # try:
            #     fx.logout()
            # except Exception as e:
            #     print("Exception: " + str(e))


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

# For next improvement
# How    to    use    this    fxcorepy.Constants.SystemProperties.END_TRADING_DAY ?
