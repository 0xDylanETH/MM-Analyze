from dotenv import load_dotenv
import os

# load proxies
load_dotenv('proxy.env')
binance_proxy = os.getenv('binance_proxy')
