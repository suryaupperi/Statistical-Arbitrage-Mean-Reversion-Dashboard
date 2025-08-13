import yfinance as yf
import numpy as np
import pandas as pd
import itertools
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
tickers = ['PNC','TFC','FITB','KEY', 'JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'USB']
start = '2020-01-01'
end = '2025-01-01'

z_entry = 1.0
z_exit  = 0.25
lookback = 60

lookback_range = [30, 60, 90]
z_entry_range = [0.8, 1.0, 1.2]
z_exit_range  = [0.2, 0.25, 0.3]
transaction_cost = 0.0005

# Data
df = yf.download(tickers, start=start, end=end)

# Select adjusted close prices
if 'Adj Close' in df.columns.levels[0]:
    px = df.xs('Adj Close', axis=1, level=0).dropna(how='any')
else:
    px = df.xs('Close', axis=1, level=0).dropna(how='any')

# Function for Pair Trading
def test_pair(stock1, stock2, px, lookback, z_entry, z_exit):
    s1 = np.log(px[stock1])
    s2 = np.log(px[stock2])

    # Engle-Granger cointegration test
    score, pvalue, _ = coint(s1, s2)
    if pvalue > 0.05:
        return None

    # Calculate spread with a fixed beta
    beta = np.polyfit(s2, s1, 1)[0]
    spread = s1 - beta * s2

    # Rolling z-score
    roll_mean = spread.rolling(lookback).mean()
    roll_std  = spread.rolling(lookback).std()
    z = (spread - roll_mean) / roll_std

    # Generate signals
    long_sig  = z < -z_entry
    short_sig = z > z_entry
    exit_sig  = z.abs() < z_exit

    positions = pd.Series(0, index=px.index)
    positions[long_sig]  =  1
    positions[short_sig] = -1
    positions = positions.replace(0, np.nan).ffill().fillna(0)
    positions[exit_sig] = 0
    positions = positions.replace(0, np.nan).ffill().fillna(0)

    # Calculate returns
    rets = px.pct_change()
    port_rets = positions.shift(1) * (rets[stock1] - beta*rets[stock2])
    port_rets -= transaction_cost * positions.diff().abs()

    # Equity & risk 
    equity = (1 + port_rets.fillna(0)).cumprod()
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    ann_factor = 252
    ann_ret = (equity.iloc[-1])**(ann_factor / len(equity)) - 1
    ann_vol = port_rets.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    # Sortino ratio
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
for stock1, stock2 in itertools.combinations(tickers, 2):
    res = test_pair(stock1, stock2, px, lookback, z_entry, z_exit)
    if res is not None:
        results.append(res)
        print(f"Cointegrated pair found: {stock1}-{stock2}, p-value={res['pvalue']:.4f}, Sharpe={res['sharpe']:.2f}")

# Spread Plots
results_sorted = sorted(results, key=lambda x: x['sharpe'], reverse=True)
top_n = 5 

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

# Portfolio
top_pairs = results_sorted[:5]  # top 5 Sharpe pairs
portfolio_rets = pd.Series(0, index=px.index)
for res in top_pairs:
    pair_rets = res['equity'].pct_change().fillna(0)
    weight = 1 / pair_rets.std()  # inverse volatility weighting
    portfolio_rets += weight * pair_rets
portfolio_equity = (1 + portfolio_rets).cumprod()

# Equity Curves
plt.figure(figsize=(12,6))
for res in results_sorted[:top_n]:
    plt.plot(res['equity'], label=f"{res['pair'][0]}-{res['pair'][1]}")
plt.plot(portfolio_equity, label='Portfolio', color='black', linewidth=2)
plt.title('Cointegrated Pair Trading Equity Curves')
plt.legend()
plt.show()

#            -HEATMAPS-
# Correlation Price Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(px.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Pairwise Price Correlations')
plt.show()

# Cointegration p-value heatmap
pvals = pd.DataFrame(index=tickers, columns=tickers)
for stock1, stock2 in itertools.combinations(tickers, 2):
    s1 = np.log(px[stock1])
    s2 = np.log(px[stock2])
    _, pval, _ = coint(s1, s2)
    pvals.loc[stock1, stock2] = pval
    pvals.loc[stock2, stock1] = pval
pvals.fillna(1.0, inplace=True)

plt.figure(figsize=(10,8))
sns.heatmap(pvals.astype(float), annot=True, fmt=".3f", cmap="viridis")
plt.title('Pairwise Cointegration p-values')
plt.show()
