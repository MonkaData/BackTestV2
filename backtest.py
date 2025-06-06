import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Trade:
    entry_price: float
    qty: float
    stop_loss: float
    take_profit: float
    trailing_stop: float

@dataclass
class BacktestResult:
    params: Dict[str, Any]
    equity_curve: List[float]
    timestamps: List[pd.Timestamp]
    fees_total: float


def fetch_ohlcv(symbol: str, timeframe: str, since: int, limit: int = 1000) -> pd.DataFrame:
    exchange = ccxt.binance()
    all_candles = []
    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not candles:
            break
        all_candles += candles
        since = candles[-1][0] + 1
        if len(candles) < limit:
            break
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df


def SMA(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()


def ATR(df: pd.DataFrame, length: int) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=length).mean()


def run_backtest(df: pd.DataFrame, params: Dict[str, Any]) -> BacktestResult:
    capital = params['capital_init']
    equity = capital
    last_equity_high = capital
    portfolio: Dict[str, Trade] = {}
    last_loss_time: Dict[str, pd.Timestamp] = {}
    fees_paid_total = 0.0
    equity_curve = []
    timestamps = []

    for i in range(len(df)):
        price = df.loc[df.index[i], 'close']
        now_ts = df.index[i]
        ma = df['ma'].iloc[i]
        atr = df['atr'].iloc[i]
        symbol = 'BTC/USDT'

        if symbol in last_loss_time and now_ts < last_loss_time[symbol] + pd.Timedelta(hours=params['cooldown_h']):
            equity_curve.append(equity)
            timestamps.append(now_ts)
            continue

        # position management
        if symbol in portfolio:
            pos = portfolio[symbol]
            new_trail = price - params['trailing_atr'] * atr
            if new_trail > pos.trailing_stop:
                pos.trailing_stop = new_trail
            if (price <= pos.stop_loss or price >= pos.take_profit or price <= pos.trailing_stop):
                exec_price = price
                profit = (exec_price - pos.entry_price) * pos.qty
                capital += profit
                fees = abs(exec_price * pos.qty) * 0.001
                fees_paid_total += fees
                del portfolio[symbol]
                if profit < 0:
                    last_loss_time[symbol] = now_ts
                equity = capital
        else:
            breakout = price > ma + params['k_break'] * atr
            if breakout and atr > 0 and not np.isnan(atr) and not np.isnan(ma):
                risk_amt = params['risk_trade'] * capital
                qty = risk_amt / (params['sl_atr'] * atr)
                if qty <= 0:
                    continue
                entry = price
                fees = entry * qty * 0.001
                fees_paid_total += fees
                capital -= entry * qty + fees
                portfolio[symbol] = Trade(
                    entry_price=entry,
                    qty=qty,
                    stop_loss=entry - params['sl_atr'] * atr,
                    take_profit=entry + params['tp_atr'] * atr,
                    trailing_stop=entry - params['trailing_atr'] * atr,
                )
                equity = capital

        unrealized = 0.0
        if symbol in portfolio:
            pos = portfolio[symbol]
            unrealized = (price - pos.entry_price) * pos.qty
        equity = capital + unrealized

        if equity < last_equity_high * (1 - params['max_drawdown']):
            if symbol in portfolio:
                del portfolio[symbol]
            capital = equity
            break

        last_equity_high = max(last_equity_high, equity)
        equity_curve.append(equity)
        timestamps.append(now_ts)

    return BacktestResult(params=params, equity_curve=equity_curve, timestamps=timestamps, fees_total=fees_paid_total)


def grid_search(df: pd.DataFrame, param_grid: Dict[str, List[Any]], top_n: int = 3) -> List[BacktestResult]:
    keys = list(param_grid.keys())
    import itertools
    results: List[BacktestResult] = []
    for values in itertools.product(*param_grid.values()):
        params = dict(zip(keys, values))
        df['ma'] = SMA(df['close'], params['ma_len'])
        df['atr'] = ATR(df, params['atr_len'])
        res = run_backtest(df, params)
        results.append(res)
        print('Tested', params, 'Final equity', res.equity_curve[-1])
    results.sort(key=lambda r: r.equity_curve[-1], reverse=True)
    return results[:top_n]


def plot_result(result: BacktestResult, filename: str):
    plt.figure(figsize=(10, 6))
    plt.plot(result.timestamps, result.equity_curve)
    plt.title('Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Equity (€)')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    start_date = '2021-01-01'
    timeframe = '30m'
    symbol = 'BTC/USDT'
    since = int(pd.Timestamp(start_date).timestamp() * 1000)
    df = fetch_ohlcv(symbol, timeframe, since)
    param_grid = {
        'capital_init': [100.0],
        'risk_trade': [0.05, 0.08],
        'atr_len': [14, 20],
        'ma_len': [14, 20],
        'k_break': [1.5, 2.0],
        'sl_atr': [1.0, 1.2],
        'tp_atr': [2.0, 2.5],
        'trailing_atr': [0.5, 0.8],
        'max_drawdown': [0.2, 0.25],
        'cooldown_h': [1, 3],
    }
    top_results = grid_search(df, param_grid, top_n=3)
    for idx, res in enumerate(top_results, 1):
        filename = f'equity_curve_{idx}.png'
        plot_result(res, filename)
        print(f'Result {idx}: final equity {res.equity_curve[-1]:.2f}€, fees {res.fees_total:.2f}€, plot saved to {filename}')
        print('Params:', res.params)

if __name__ == '__main__':
    main()
