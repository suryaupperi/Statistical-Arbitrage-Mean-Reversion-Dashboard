import yfinance as yf
import numpy as np
import pandas as pd
import itertools
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import seaborn as sns

user_input = input("Enter ticker symbols separated by commas (or press Enter for default list): ").strip()
if user_input:
    tickers = [t.strip().upper() for t in user_input.split(",")]
else:
    tickers = ['PNC','TFC','FITB','KEY', 'JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'USB']

start_input = input("Enter start date (YYYY-MM-DD, or press Enter for 2020-01-01): ").strip()
end_input   = input("Enter end date (YYYY-MM-DD, or press Enter for 2025-01-01): ").strip()

start = start_input if start_input else '2020-01-01'
end   = end_input if end_input else '2025-01-01'

z_entry = 1.0
z_exit  = 0.25
lookback = 60

lookback_range = [30, 60, 90]
z_entry_range = [0.8, 1.0, 1.2]
z_exit_range  = [0.2, 0.25, 0.3]
transaction_cost = 0.0005

print(f"\nDownloading data for: {tickers}")
df = yf.download(tickers, start=start, end=end, auto_adjust=False)


if 'Adj Close' in df.columns.levels[0]:
    px = df.xs('Adj Close', axis=1, level=0)
else:
    px = df.xs('Close', axis=1, level=0)


available_tickers = [t for t in tickers if t in px.columns]
missing_tickers = set(tickers) - set(available_tickers)

if missing_tickers:
    print(f"\n Warning: The following tickers failed to download and will be skipped: {missing_tickers}")

px = px[available_tickers].dropna(how='any')

if px.empty:
    raise ValueError("No valid tickers with price data available. Exiting program.")


def test_pair(stock1, stock2, px, lookback, z_entry, z_exit):
    s1 = np.log(px[stock1])
    s2 = np.log(px[stock2])

    
    score, pvalue, _ = coint(s1, s2)
    if pvalue > 0.05:
        return None

    
    beta = np.polyfit(s2, s1, 1)[0]
    spread = s1 - beta * s2

    
    roll_mean = spread.rolling(lookback).mean()
    roll_std  = spread.rolling(lookback).std()
    z = (spread - roll_mean) / roll_std

    
    long_sig  = z < -z_entry
    short_sig = z > z_entry
    exit_sig  = z.abs() < z_exit

    positions = pd.Series(0, index=px.index)
    positions[long_sig]  =  1
    positions[short_sig] = -1
    positions = positions.replace(0, np.nan).ffill().fillna(0)
    positions[exit_sig] = 0
    positions = positions.replace(0, np.nan).ffill().fillna(0)

    
    rets = px.pct_change()
    port_rets = positions.shift(1) * (rets[stock1] - beta*rets[stock2])
    port_rets -= transaction_cost * positions.diff().abs()

    
    equity = (1 + port_rets.fillna(0)).cumprod()
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    ann_factor = 252
    ann_ret = (equity.iloc[-1])**(ann_factor / len(equity)) - 1
    ann_vol = port_rets.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

   
    downside_rets = port_rets[port_rets < 0]
    sortino = ann_ret / (downside_rets.std() * np.sqrt(ann_factor)) if len(downside_rets) > 0 else np.nan
    cagr = (equity.iloc[-1])**(ann_factor/len(equity)) - 1

    return {
        'pair': (stock1, stock2),
        'pvalue': pvalue,
        'beta': beta,
        'equity': equity,
        'total_return': equity.iloc[-1] - 1,
        'annual_return': ann_ret,
        'annual_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'cagr': cagr,
        'max_drawdown': max_dd
    }


results = []
for stock1, stock2 in itertools.combinations(available_tickers, 2):
    try:
        res = test_pair(stock1, stock2, px, lookback, z_entry, z_exit)
        if res is not None:
            results.append(res)
            print(f"Cointegrated pair found: {stock1}-{stock2}, p-value={res['pvalue']:.4f}, Sharpe={res['sharpe']:.2f}")
    except Exception as e:
        print(f"⚠️ Error testing pair {stock1}-{stock2}: {e}")

if not results:
    raise ValueError("No cointegrated pairs found with given parameters.")


results_sorted = sorted(results, key=lambda x: x['sharpe'], reverse=True)
top_n = min(5, len(results_sorted))

for res in results_sorted[:top_n]:
    plt.figure(figsize=(12,6))
    s1 = np.log(px[res['pair'][0]])
    s2 = np.log(px[res['pair'][1]])
    spread = s1 - res['beta'] * s2
    roll_mean = spread.rolling(lookback).mean()
    roll_std  = spread.rolling(lookback).std()
    z = (spread - roll_mean) / roll_std

    plt.plot(spread, label='Spread')
    plt.plot(roll_mean, label='Rolling Mean', linestyle='--')
    plt.fill_between(z.index, roll_mean - z_entry*roll_std, roll_mean + z_entry*roll_std, color='lightgray', alpha=0.5)
    plt.title(f'Spread and Signals: {res["pair"][0]}-{res["pair"][1]}, Sharpe={res["sharpe"]:.2f}')
    plt.legend()
    plt.show()


top_pairs = results_sorted[:top_n]
portfolio_rets = pd.Series(0, index=px.index)
for res in top_pairs:
    pair_rets = res['equity'].pct_change().fillna(0)
    weight = 1 / pair_rets.std() if pair_rets.std() > 0 else 0
    portfolio_rets += weight * pair_rets
portfolio_equity = (1 + portfolio_rets).cumprod()


plt.figure(figsize=(12,6))
for res in results_sorted[:top_n]:
    plt.plot(res['equity'], label=f"{res['pair'][0]}-{res['pair'][1]}")
plt.plot(portfolio_equity, label='Portfolio', color='black', linewidth=2)
plt.title('Cointegrated Pair Trading Equity Curves')
plt.legend()
plt.show()


plt.figure(figsize=(10,8))
sns.heatmap(px.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Pairwise Price Correlations')
plt.show()


pvals = pd.DataFrame(index=available_tickers, columns=available_tickers)
for stock1, stock2 in itertools.combinations(available_tickers, 2):
    try:
        s1 = np.log(px[stock1])
        s2 = np.log(px[stock2])
        _, pval, _ = coint(s1, s2)
        pvals.loc[stock1, stock2] = pval
        pvals.loc[stock2, stock1] = pval
    except Exception:
        pvals.loc[stock1, stock2] = 1.0
        pvals.loc[stock2, stock1] = 1.0
pvals.fillna(1.0, inplace=True)

plt.figure(figsize=(10,8))
sns.heatmap(pvals.astype(float), annot=True, fmt=".3f", cmap="viridis")
plt.title('Pairwise Cointegration p-values')
plt.show()
