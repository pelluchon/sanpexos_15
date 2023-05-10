import fxcmpy

# inputs
TOKEN = 'e44970156e2a002a60d8b040ea85de02f03f14c0'
server_name = 'demo2'

con = fxcmpy.fxcmpy(access_token=TOKEN, log_level='error', server=server_name, log_file='log.txt')
con.close_all_for_symbol('EUR/SEK')
#con.close_all_for_symbol('JPYBasket')
#con.close_all_for_symbol('EMBasket')

# request = fx.create_order_request(
#     order_type=fxcorepy.Constants.Orders.TRUE_MARKET_CLOSE,
#     OFFER_ID=offer.offer_id,
#     ACCOUNT_ID=Dict['FXCM']['str_account'],
#     BUY_SELL=buy_sell,
#     AMOUNT=float(dj.loc[0, 'tick_amount']),
#     TRADE_ID=dj.loc[0, 'tick_id']
# )
# resp = fx.send_request(request)