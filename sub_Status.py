from datetime import datetime
from forexconnect import fxcorepy, ForexConnect, Common
import time

Dict = {
    'FXCM': {
            'str_user_i_d': '71597804',
            'str_password': 'pz1qfjp',
            'str_connection': 'Demo',
            'str_account': '71597804', 
            #'str_user_i_d': '71592345',
            # 'str_password': 'ioj4bse',
            # 'str_connection': 'Demo',
            # 'str_account': '71592345',
            #'str_user_i_d': '87053959',
            #'str_password': 'sanpexos1703!',
            #'str_account': '87053959',
            #'str_connection': 'Real',
            'str_url': "http://www.fxcorporate.com/Hosts.jsp",
            'str_session_id': None,
            'str_pin': None,
            'str_table': 'orders',
        },
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
                       'NZD/CHF', 'NZD/JPY', 'NZD/USD', 'USD/CAD',
                       'USD/CHF', 'USD/HKD', 'USD/JPY', 'USD/MXN',
                       'USD/NOK', 'USD/SEK', 'USD/ZAR', 'XAG/USD', 'XAU/USD',
                       'USD/ILS', 'BTC/USD', 'BCH/USD', 'ETH/USD',
                       'JPN225', 'NAS100', 'NGAS', 'SPX500', 'US30', 'VOLX',
                       'US2000', 'AUS200', 'UKOil', 'USOil', 'USOilSpot', 'UKOilSpot',
                       'EMBasket', 'USDOLLAR', 'JPYBasket', 'CryptoMajor']},

            2: {
                'hour_open': 0,  # opening time in UTC (for midnight put 24)
                'hour_close': 6,  # closing time in UTC
                'FX': [
                    'CSL.au', 'CBA.au', 'BHP.au', 'WBC.au', 'NAB.au', 'ANZ.au',
                    'WOW.au', 'WES.au', 'FMG.au', 'MQG.au', 'TLS.au', 'RIO.au',
                    'GMG.au', 'COL.au', 'ALL.au', 'A2M.au',
                    'REA.au', 'XRO.au', 'QAN.au', 'Z1P.au']},

            3: {
                'hour_open': 7,  # opening time in UTC (for midnight put 24)
                'hour_close': 15,  # closing time in UTC
                'FX': ['ADS.de', 'ALV.de', 'BAS.de', 'BAYN.de', 'BMW.de',
                       'DB1.de', 'DBK.de', 'DPW.de', 'DTE.de', 'IFX.de',
                       'LHA.de', 'MRK.de', 'RWE.de', 'SAP.de', 'SIE.de', 'TUI1.de',
                       'VOW.de', 'VNA.de', 'ENR.de', 'CBK.de', 'DHER.de']},

            4: {
                'hour_open': 14,  # opening time in UTC (for midnight put 24)
                'hour_close': 20,  # closing time in UTC
                'FX': ['FVRR.us', 'SPOT.us', 'MARA.us', 'BTBT.us',
                       'BITF.us', 'WISH.us', 'RIVN.us', 'JD.us', 'PDD.us',
                       'TME.us', 'WB.us', 'BILI.us', 'NVDA.us', 'AMD.us', 'DADA.us',
                       'PTON.us', 'MRNA.us', 'NIO.us', 'CCL.us', 'ABNB.us', 'DASH.us',
                       'AMC.us', 'FCEL.us', 'GME.us', 'PENN.us', 'PLTR.us',
                       'PLUG.us', 'PYPL.us', 'SNAP.us', 'SNOW.us', 'SPCE.us', 'XPEV.us',
                       'SONY.us', 'BA.us', 'BAC.us', 'BRKB.us', 'C.us', 'CRM.us',
                       'DIS.us', 'F.us', 'JPM.us', 'KO.us', 'MA.us', 'MCD.us',
                       'PFE.us', 'PG.us', 'SE.us', 'T.us', 'TGT.us', 'V.us', 'XOM.us',
                       'AAPL.us', 'AMZN.us', 'GOOG.us', 'INTC.us', 'MSFT.us',
                       'SBUX.us', 'BABA.us', 'DAL.us',
                       'NFLX.us', 'TSLA.us', 'SQ.us', 'LYFT.us', 'UAL.us', 'DKNG.us', 'SHOP.us', 'BYND.us',
                       'UBER.us', 'ZM.us', 'LCID.us', 'HOOD.us', 'CRWD.us', 'BEKE.us',
                       'CPNG.us', 'NET.us', 'RBLX.us', 'COIN.us']},

            5: {
                'hour_open': 2,  # opening time in UTC (for midnight put 24)
                'hour_close': 8,  # closing time in UTC
                'FX': ['TENC.hk', 'MEIT.hk', 'BYDC.hk', 'XIAO.hk', 'BABA.hk',
                       'AIA.hk', 'HSBC.hk', 'WUXI.hk', 'HKEX.hk', 'GELY.hk', 'JD.hk',
                       'NETE.hk', 'PING.hk', 'SMIC.hk', 'SBIO.hk', 'GALA.hk', 'KIDE.hk',
                       'ALIH.hk', 'ICBC.hk', 'FLAT.hk', 'KSOF.hk', 'SMOO.hk', 'SUNN.hk',
                       'BYDE.hk']},

            6: {
                'hour_open': 7,  # opening time in UTC (for midnight put 24)
                'hour_close': 15,  # closing time in UTC
                'FX': ['ACA.fr', 'AI.fr', 'ALO.fr', 'BNP.fr', 'CA.fr',
                       'AIR.fr', 'ORA.fr', 'GLE.fr', 'MC.fr', 'ML.fr', 'OR.fr',
                       'RNO.fr', 'SAN.fr', 'SU.fr', 'VIV.fr', 'TTE.fr', 'ENGI.fr', 'STM.fr',
                       'STLA.fr']},

            7: {
                'hour_open': 7,  # opening time in UTC (for midnight put 24)
                'hour_close': 15,  # closing time in UTC
                'FX': ['AV.uk', 'AZN.uk', 'BA.uk', 'BARC.uk', 'BATS.uk',
                       'BP.uk', 'GSK.uk', 'HSBA.uk', 'IAG.uk', 'LLOY.uk',
                       'RR.uk', 'STAN.uk', 'TSCO.uk', 'VOD.uk', 'GLEN.uk',
                       'BT.A.uk', 'NWG.uk', 'TW.uk', 'MRO.uk', 'ROO.uk']},

            8: {
                'hour_open': 6,  # opening time in UTC (for midnight put 24)
                'hour_close': 20,  # closing time in UTC
                'FX': ['EUSTX50', 'FRA40']},

            9: {
                'hour_open': 6,  # opening time in UTC (for midnight put 24)
                'hour_close': 18,  # closing time in UTC
                'FX': ['ESP35', 'Bund']},

            10: {
                'hour_open': 1,  # opening time in UTC (for midnight put 24)
                'hour_close': 19,  # closing time in UTC
                'FX': ['GER30', 'HKG33', 'CHN50', 'UK100']},

            11: {
                'hour_open': 0,  # opening time in UTC (for midnight put 24)
                'hour_close': 18,  # closing time in UTC
                'FX': ['SOYF', 'WHEATF', 'CORNF']},

            12: {
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

#Removed
#, 'AskU.us','AskU.hk'

def session_status_changed(session: fxcorepy.O2GSession,
                           status: fxcorepy.AO2GSessionStatus.O2GSessionStatus):
    print("Trading session status: " + str(status))


print(str(datetime.now().strftime("%H:%M:%S")))
with ForexConnect() as fx:
    fx.login(Dict['FXCM']['str_user_i_d'], Dict['FXCM']['str_password'], Dict['FXCM']['str_url'],
             Dict['FXCM']['str_connection'], Dict['FXCM']['str_session_id'], Dict['FXCM']['str_pin'],
             session_status_callback=session_status_changed)
    login_rules = fx.login_rules
    trading_settings_provider = login_rules.trading_settings_provider
    for l0 in range(1, len(Dict['instrument'])):
        FX = Dict['instrument'][l0]['FX']
        for l1 in range(0, len(FX)):
            tick = FX[l1]

            account = Common.get_account(fx, Dict['FXCM']['str_account'])
            base_unit_size = trading_settings_provider.get_base_unit_size(tick, account)
            amount = base_unit_size*1

            def get_offer(fx, s_instrument):
                table_manager = fx.table_manager
                offers_table = table_manager.get_table(ForexConnect.OFFERS)
                for offer_row in offers_table:
                    if offer_row.instrument == s_instrument:
                        return offer_row

            offer = get_offer(fx, tick)
            if not offer:
                raise Exception(
                    "The instrument '{0}' is not valid".format(tick))

            status=None
            old_status = None
            string = 'instrument=' + offer.instrument + '; subscription_status=' + offer.subscription_status
            old_status = offer.subscription_status



            if status == old_status:
                raise Exception('New status = current status')

            offers_table = fx.get_table(ForexConnect.OFFERS)
            if offer.subscription_status != fxcorepy.Constants.SubscriptionStatuses.TRADABLE:
                # Subscribe to the offer
                request = fx.create_request({
                    fxcorepy.O2GRequestParamsEnum.COMMAND: fxcorepy.Constants.Commands.SET_SUBSCRIPTION_STATUS,
                    fxcorepy.O2GRequestParamsEnum.OFFER_ID: offer.offer_id,
                    fxcorepy.O2GRequestParamsEnum.SUBSCRIPTION_STATUS: fxcorepy.Constants.SubscriptionStatuses.TRADABLE
                })
                fx.send_request(request)
                print(string)
                #Change subscription_status
