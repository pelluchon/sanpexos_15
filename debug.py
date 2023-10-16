from datetime import datetime
from forexconnect import fxcorepy, ForexConnect, Common
import time

Dict = {
    'FXCM': {
            'str_user_i_d': '71581198',
            'str_password': '8ylyjhu',
            'str_connection': 'Demo',
            'str_account': '71581198',
            # 'str_user_i_d': '87053959',
            # 'str_password': 'S4Tpj3P!zz.Mm2p',
            # 'str_account': '87053959',
            # 'str_connection': 'Real',
            'str_url': "http://www.fxcorporate.com/Hosts.jsp",
            'str_session_id': None,
            'str_pin': None,
            'str_table': 'orders',
        },

}

def session_status_changed(session: fxcorepy.O2GSession,
                           status: fxcorepy.AO2GSessionStatus.O2GSessionStatus):
    print("Trading session status: " + str(status))

tick='AUD/CAD'
print(str(datetime.now().strftime("%H:%M:%S")))
with ForexConnect() as fx:
    fx.login(Dict['FXCM']['str_user_i_d'], Dict['FXCM']['str_password'], Dict['FXCM']['str_url'],
             Dict['FXCM']['str_connection'], Dict['FXCM']['str_session_id'], Dict['FXCM']['str_pin'],
             session_status_callback=session_status_changed)
    login_rules = fx.login_rules
    trading_settings_provider = login_rules.trading_settings_provider


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
    print(string)


    if status == old_status:
        raise Exception('New status = current status')

    offers_table = fx.get_table(ForexConnect.OFFERS)
    # Subscribe to the offer
    request = fx.create_request({
        fxcorepy.O2GRequestParamsEnum.COMMAND: fxcorepy.Constants.Commands.SET_SUBSCRIPTION_STATUS,
        fxcorepy.O2GRequestParamsEnum.OFFER_ID: offer.offer_id,
        fxcorepy.O2GRequestParamsEnum.SUBSCRIPTION_STATUS: fxcorepy.Constants.SubscriptionStatuses.TRADABLE
    })
    fx.send_request(request)

    # Wait for the subscription to be updated
    while offer.subscription_status != fxcorepy.Constants.SubscriptionStatuses.TRADABLE:
        time.sleep(60)
        print('wait')

    print('now yes')
    # Now that the offer is subscribed to, you can place your order
    request = fx.create_order_request(
        order_type=fxcorepy.Constants.Orders.TRUE_MARKET_OPEN,
        ACCOUNT_ID=Dict['FXCM']['str_account'],
        BUY_SELL=fxcorepy.Constants.BUY,
        #RATE = 1,
        AMOUNT=amount,
        SYMBOL=tick,#offer.instrument,
    )
    fx.send_request(request)