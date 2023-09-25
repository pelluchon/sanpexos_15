from datetime import datetime
from forexconnect import fxcorepy, ForexConnect, Common

Dict = {
    'FXCM': {
            'str_user_i_d': '87053959',
            'str_password': 'S4Tpj3P!zz.Mm2p',
            'str_url': "http://www.fxcorporate.com/Hosts.jsp",
            'str_connection': 'Real',
            'str_session_id': None,
            'str_pin': None,
            'str_table': 'orders',
            'str_account': '87053959',
        },
}

def session_status_changed(session: fxcorepy.O2GSession,
                           status: fxcorepy.AO2GSessionStatus.O2GSessionStatus):
    print("Trading session status: " + str(status))

tick='BTC/USD'
print(str(datetime.now().strftime("%H:%M:%S")))
with ForexConnect() as fx:
    fx.login(Dict['FXCM']['str_user_i_d'], Dict['FXCM']['str_password'], Dict['FXCM']['str_url'],
             Dict['FXCM']['str_connection'], Dict['FXCM']['str_session_id'], Dict['FXCM']['str_pin'],
             session_status_callback=session_status_changed)
    login_rules = fx.login_rules
    trading_settings_provider = login_rules.trading_settings_provider


    account = Common.get_account(fx, Dict['FXCM']['str_account'])
    base_unit_size = trading_settings_provider.get_base_unit_size(tick, account)
    amount = base_unit_size * 1

    offer = Common.get_offer(fx, tick)
    if not offer:
        raise Exception(
            "The instrument '{0}' is not valid".format(tick))
    request = fx.create_order_request(
        order_type=fxcorepy.Constants.Orders.TRUE_MARKET_OPEN,
        ACCOUNT_ID=Dict['FXCM']['str_account'],
        BUY_SELL=fxcorepy.Constants.BUY,
        AMOUNT=amount,
        SYMBOL=offer.instrument,
    )
    fx.send_request(request)