import io
import zipfile
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
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


def download_binance_klines(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    """Download historical klines from data.binance.vision.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. ``'BTCUSDT'``.
    timeframe : str
        Interval such as ``'30m'``.
    start : str
        Start date in ``YYYY-MM`` or ``YYYY-MM-DD`` format.
    end : str
        End date in ``YYYY-MM`` or ``YYYY-MM-DD`` format.
    """
    start_p = pd.Period(start, freq="M")
    end_p = pd.Period(end, freq="M")
    months = pd.period_range(start_p, end_p, freq="M")

    frames = []
    for p in months:
        url = (
            f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{timeframe}/"
            f"{symbol}-{timeframe}-{p.year}-{p.month:02d}.zip"
        )
        r = requests.get(url)
        if r.status_code != 200:
            print(f"Warning: failed to download {url} ({r.status_code})")
            continue
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            name = z.namelist()[0]
            with z.open(name) as f:
                df_month = pd.read_csv(
                    f,
                    header=None,
                    names=[
                        "open_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "quote_volume",
                        "trades",
                        "taker_buy_base",
                        "taker_buy_quote",
                        "ignore",
                    ],
                )
        frames.append(df_month)

    if not frames:
        raise RuntimeError("No data downloaded")

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


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
        symbol = 'BTCUSDT'

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
                fees = abs(exec_price * pos.qty) * 0.001
                fees_paid_total += fees
                capital += profit - fees
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
                capital -= fees
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
    start_date = '2024-01-01'
    end_date = '2024-06-30'
    timeframe = '30m'
    symbol = 'BTCUSDT'

    df = download_binance_klines(symbol, timeframe, start_date, end_date)
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
