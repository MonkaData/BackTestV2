# BackTestV2

This repository contains a simple backtesting script for a breakout strategy on BTC/USDT using historical data from [data.binance.vision](https://data.binance.vision/).

## Requirements
```
pip install -r requirements.txt
```

## Running the backtest
```
python backtest.py
```
The script automatically downloads monthly candle data and performs a grid search over several strategy parameters. It outputs the three best equity curves and saves plots as `equity_curve_1.png`, `equity_curve_2.png`, and `equity_curve_3.png`.

