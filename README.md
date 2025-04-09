# Crypto 101 Alphas

Crypto 101 Alphas is a quantitative research project designed for the crypto market. Inspired by the Machine Learning for Trading (ML4T) repository, this project gathers data from Binance and CoinGecko to construct and study crypto alphas from WorldQuant 101 Alphas (quantitative signals).

## Table of Contents

- [Crypto 101 Alphas](#crypto-101-alphas)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Requirements](#requirements)
  - [Usage](#usage)

## Features

- **Data Acquisition & Processing**  
  - **Binance Data Loader:** Uses the Binance API to retrieve historical OHLCV data for multiple cryptocurrencies.
  - **CoinGecko Metadata:** Fetches market metadata (market cap, total volume, etc.) from CoinGecko.
  - **Data Merging:** Merges Binance price data with CoinGecko metadata while reconciling symbol differences (e.g., stripping “USDT” from Binance symbols).

- **Factor Calculation & Analysis**  
  - **alpha_utils.py:** Provides various utility functions for time series operations (e.g., rolling standard deviation, rolling rank, weighted moving average, etc.) and factor computations.
  - **Example Factors:** Includes implementations of alpha formulas (e.g., Alpha#1) and analysis routines (e.g., linear regression, model diagnostics).
  - **Notebook Examples:** Detailed Jupyter Notebooks to examine factor performance and test signal predictability.

## Requirements

- Python 3.6+
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [requests](https://docs.python-requests.org/)
- [python-binance](https://github.com/sammchardy/python-binance)
- [python-dotenv](https://github.com/theskumar/python-dotenv)
- [ta](https://github.com/bukosabino/ta)
- [statsmodels](https://www.statsmodels.org/)
- [seaborn](https://seaborn.pydata.org/) & [matplotlib](https://matplotlib.org/)
- (Optional) [requests_html](https://requests-html.kennethreitz.org/) – for advanced scraping if needed


## Usage

Run `crypto_data_loader.py` to fetch historical price data and metadata from Binance and CoinGecko, then merge them into a single DataFrame. 

Run `notebooks\alpha101.ipynb` to examine alphas' effect on cryptos


