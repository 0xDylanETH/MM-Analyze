import pandas as pd
import requests
from binance.client import Client
from datetime import datetime
import numpy as np
import time
from config import *

class DataHandler:
    def __init__(self, api_key=None, api_secret=None):
        self.client = Client(api_key, api_secret, requests_params={
                'proxies': {
                    'http': binance_proxy,
                    'https': binance_proxy,
                    }
                })

    @staticmethod
    def transform_df(df):
        df = df.iloc[:, :6]
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'], unit='ms', utc=True)
        df.set_index('Date', inplace=True)
        return df.astype(float)

    def get_historical_klines(self, symbol='BTCUSDT', interval='1h', start_str='30 days ago UTC', end_str=None):
        raw_data = self.client.get_historical_klines(symbol=symbol, interval=interval, start_str=start_str, end_str=end_str)
        df = pd.DataFrame(raw_data)
        return self.transform_df(df)

    def get_binance_symbols(self, asset='USDT'):
        exchange_info = self.client.get_exchange_info()

        symbols = [
            s['symbol'] for s in exchange_info['symbols']
            if s['status'] == 'TRADING'
               and s['isSpotTradingAllowed']
               and s['quoteAsset'] == asset
        ]

        return symbols

def load_multi_symbol_data(handler, symbols, interval='1h', start_str='30 days ago UTC', end_str=None):
    all_data = []
    for symbol in symbols:
        df = handler.get_historical_klines(symbol=symbol, interval=interval, start_str=start_str, end_str=end_str)
        df['symbol'] = symbol
        all_data.append(df)

    df_all = pd.concat(all_data)
    df_all.set_index(['symbol'], append=True, inplace=True)
    df_all = df_all.reorder_levels(['symbol', 'Date']).sort_index()
    return df_all

def get_coingecko_metadata(vs_currency="usd", per_page=250, page_limit=4):
    """Fetch coin metadata from CoinGecko: id, symbol, name, market cap, total_volume."""
    all_data = []
    for page in range(1, page_limit + 1):
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": page,
            "sparkline": "false"
        }
        response = requests.get(url, params=params)  # 修复：传入 params 参数
        if response.status_code != 200:
            break
        data_page = response.json()
        if not data_page:
            break
        all_data.extend(data_page)
    df = pd.DataFrame(all_data)
    df['symbol'] = df['symbol'].str.upper()

    return df[['id', 'symbol', 'name', 'market_cap', 'total_volume']]

def merge_metadata(df_all, metadata):
    """
    将多币种 OHLCV 数据 df_all 与 CoinGecko 的 metadata 按 symbol 合并，
    并恢复 MultiIndex 结构。
    """
    # 将所有索引层级都 reset 成普通列
    df_temp = df_all.reset_index()  # 此时 'symbol' 和 'Date' 都变成了普通列

    # 新增 base_symbol 列，用于匹配币种基础代码
    df_temp['base_symbol'] = df_temp['symbol'].apply(lambda x: x[:-4] if x.endswith('USDT') else x)

    # 对 metadata 做处理，添加 base_symbol 列
    meta = metadata.copy()
    meta['base_symbol'] = meta['symbol']

    # 按 base_symbol 进行左连接合并
    df_merged = pd.merge(df_temp, meta, on='base_symbol', how='left', suffixes=('', '_meta'))

    # 删除辅助列
    df_merged.drop('base_symbol', axis=1, inplace=True)

    # 将 'symbol' 和 'Date' 列重新设为 MultiIndex，并排序
    df_merged = df_merged.set_index(['symbol', 'Date']).sort_index()

    return df_merged

# if __name__ == '__main__':

    # data_handler = DataHandler()
    # symbols = data_handler.get_binance_symbols()[:20]
    # print(symbols)
    #
    # df_all = load_multi_symbol_data(data_handler, symbols, interval='1h', start_str='2024-07-01 00:00:00')
    # print(df_all.info())
    # print(df_all.head())
    # df_all.to_pickle('data/price.pkl')

    # metadata = get_coingecko_metadata()
    # print(metadata.info())
    # print(metadata.head())
    # metadata.to_pickle('data/metadata.pkl')
    #
    # df_all = pd.read_pickle('data/price.pkl')
    # metadata = pd.read_pickle('data/metadata.pkl')
    # df_merged = merge_metadata(df_all, metadata)
    # print(df_merged.info())
    # print(df_merged.head())
    # df_merged.to_pickle('data/merged_df.pkl')

