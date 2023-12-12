#!/usr/bin/env python
# coding: utf-8

# In[348]:


import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')

from time import time
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from datetime import datetime

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from talib import RSI, BBANDS, MACD, ATR
import sys, os
import pandas_datareader.data as web

DATA_STORE = Path('assets.h5')


# In[218]:


df1 = pd.read_csv('NASDAQ.csv')
df2 = pd.read_csv('AMEX.csv')
df = pd.concat([df1, df2]).dropna(how='all', axis=1)
df = df.rename(columns=str.lower).set_index('symbol')
df = df[~df.index.duplicated()]
print(df.info())


# In[219]:


df = df.rename(columns = {'last sale': 'lastsale', 'market cap': 'marketcap', 'ipo year': 'ipoyear'})

mcap = df[['marketcap']].dropna()
df['marketcap'] = mcap.marketcap
df.marketcap.describe(percentiles=np.arange(.1, 1, .1).round(1)).apply(lambda x: f'{int(x):,d}')


# In[220]:


df = pd.read_csv('us_equities_meta_data.csv')
df.info()


# In[221]:


with pd.HDFStore(DATA_STORE) as store:
    store.put('us_equities/stocks', df.set_index('ticker'))


# In[222]:


# Alpha Factors and Features

# average number of active days in a month = MONTH
MONTH = 21
YEAR = 12 * MONTH

START = '2013-01-01'
END = '2017-12-31'

sns.set_style('whitegrid')
idx = pd.IndexSlice


# In[223]:


ohlcv = ['adj_open', 'adj_close', 'adj_low', 'adj_high', 'adj_volume']


# In[224]:


with pd.HDFStore(DATA_STORE) as store:
    prices = (store['quandl/wiki/prices']
              .loc[idx[START:END, :], ohlcv]
              .rename(columns=lambda x: x.replace('adj_', ''))
              .assign(volume=lambda x: x.volume.div(1000))
              .swaplevel()
              .sort_index())

    stocks = (store['us_equities/stocks']
              .loc[:, ['marketcap', 'ipoyear', 'sector']])


# In[225]:


#Remove stocks with few observations
# want at least 2 years of data
min_obs = 2 * YEAR

# have this much per ticker 
nobs = prices.groupby(level='ticker').size()

# keep those that exceed the limit
keep = nobs[nobs > min_obs].index

prices = prices.loc[idx[keep, :], :]


# In[226]:


# Aligning price and meta data
stocks = stocks[~stocks.index.duplicated() & stocks.sector.notnull()]
stocks.sector = stocks.sector.str.lower().str.replace(' ', '_')
stocks.index.name = 'ticker'


# In[227]:


shared = (prices.index.get_level_values('ticker').unique()
          .intersection(stocks.index))
stocks = stocks.loc[shared, :]
prices = prices.loc[idx[shared, :], :]


# In[228]:


prices.info(show_counts=True)


# In[229]:


stocks.info(show_counts=True)


# In[230]:


stocks.sector.value_counts()


# In[231]:


# compute dollar volume to determine universe
prices['dollar_vol'] = prices[['close', 'volume']].prod(axis=1)


# In[232]:


prices['dollar_vol_1m'] = (prices.dollar_vol.groupby('ticker')
                           .rolling(window=21)
                           .mean()).values


# In[233]:


prices['dollar_vol_rank'] = (prices.groupby('date')
                             .dollar_vol_1m
                             .rank(ascending=False))


# In[234]:


prices.info(show_counts=True)


# In[235]:


## Adding Basic Factors
# Computing Relative Strength Index

prices['rsi'] = prices.groupby(level='ticker').close.apply(RSI)


# In[236]:


ax = sns.distplot(prices.rsi.dropna())
ax.axvline(30, ls='--', lw=1, c='k')
ax.axvline(70, ls='--', lw=1, c='k')
ax.set_title('RSI Distribution with Signal Threshold')
plt.tight_layout();


# In[237]:


## Computing Bollinger Bands

def compute_bb(close):
    high, mid, low = BBANDS(close, timeperiod=20)
    return pd.DataFrame({'bb_high': high, 'bb_low': low}, index=close.index)


# In[238]:


prices = (prices.join(prices
                      .groupby(level='ticker')
                      .close
                      .apply(compute_bb)))


# In[239]:


prices['bb_high'] = prices.bb_high.sub(prices.close).div(prices.bb_high).apply(np.log1p)
prices['bb_low'] = prices.close.sub(prices.bb_low).div(prices.close).apply(np.log1p)


# In[240]:


fig, axes = plt.subplots(ncols=2, figsize=(15, 5))
sns.distplot(prices.loc[prices.dollar_vol_rank<100, 'bb_low'].dropna(), ax=axes[0])
sns.distplot(prices.loc[prices.dollar_vol_rank<100, 'bb_high'].dropna(), ax=axes[1])
plt.tight_layout();


# In[241]:


# Compute Average True Range

def compute_atr(stock_data):
    df = ATR(stock_data.high, stock_data.low, 
             stock_data.close, timeperiod=14)
    return df.sub(df.mean()).div(df.std())


# In[242]:


prices['atr'] = (prices.groupby('ticker', group_keys=False)
                 .apply(compute_atr))


# In[243]:


sns.distplot(prices[prices.dollar_vol_rank<50].atr.dropna());


# In[244]:


# Compute Moving Average Convergence/Divergence

def compute_macd(close):
    macd = MACD(close)[0]
    return (macd - np.mean(macd))/np.std(macd)


# In[245]:


prices['macd'] = (prices
                  .groupby('ticker', group_keys=False)
                  .close
                  .apply(compute_macd))


# In[246]:


prices.macd.describe(percentiles=[.001, .01, .02, .03, .04, .05, .95, .96, .97, .98, .99, .999]).apply(lambda x: f'{x:,.1f}')


# In[247]:


sns.distplot(prices[prices.dollar_vol_rank<100].macd.dropna());


# In[248]:


# Lagged Returns Computation

lags = [1, 5, 10, 21, 42, 63]

returns = prices.groupby(level='ticker').close.pct_change()
percentiles=[.0001, .001, .01]
percentiles+= [1-p for p in percentiles]
returns.describe(percentiles=percentiles).iloc[2:].to_frame('percentiles').style.format(lambda x: f'{x:,.2%}')


# In[249]:


# Winsorization of outliers

q = 0.0001

for lag in lags:
    prices[f'return_{lag}d'] = (prices.groupby(level='ticker').close
                                .pct_change(lag)
                                .pipe(lambda x: x.clip(lower=x.quantile(q),
                                                       upper=x.quantile(1 - q)))
                                .add(1)
                                .pow(1 / lag)
                                .sub(1)
                                )


# In[250]:


# Shifting Lagged Returns

for t in [1, 2, 3, 4, 5]:
    for lag in [1, 5, 10, 21]:
        prices[f'return_{lag}d_lag{t}'] = (prices.groupby(level='ticker')
                                           [f'return_{lag}d'].shift(t * lag))


# In[251]:


# Computing Forward Returns

for t in [1, 5, 10, 21]:
    prices[f'target_{t}d'] = prices.groupby(level='ticker')[f'return_{t}d'].shift(-t)


# In[252]:


# Combining Price and Metadata
# Running once

prices = prices.join(stocks[['sector']])


# In[253]:


# Creating Time and Sector Dummy Variables

prices['year'] = prices.index.get_level_values('date').year
prices['month'] = prices.index.get_level_values('date').month


# In[254]:


prices.info(null_counts=True)


# In[255]:


prices.assign(sector=pd.factorize(prices.sector, sort=True)[0]).to_hdf('data.h5', 'model_data/no_dummies')


# In[256]:


prices = pd.get_dummies(prices,
                        columns=['year', 'month', 'sector'],
                        prefix=['year', 'month', ''],
                        prefix_sep=['_', '_', ''],
                        drop_first=True)


# In[257]:


prices.info(null_counts=True)


# In[258]:


prices.to_hdf('data.h5', 'model_data')


# In[259]:


## DATA SUMMARY EXPLORATION ##


# In[260]:


target = 'target_5d'
top100 = prices[prices.dollar_vol_rank<100].copy()


# In[261]:


## RSI

top100.loc[:, 'rsi_signal'] = pd.cut(top100.rsi, bins=[0, 30, 70, 100])


# In[262]:


top100.groupby('rsi_signal')['target_5d'].describe()


# In[263]:


## Bollinger Bands

metric = 'bb_low'
j=sns.jointplot(x=metric, y=target, data=top100)

df = top100[[metric, target]].dropna()
r, p = spearmanr(df[metric], df[target])
print(f'{r:,.2%} ({p:.2%})')


# In[264]:


metric = 'bb_high'
j=sns.jointplot(x=metric, y=target, data=top100)

df = top100[[metric, target]].dropna()
r, p = spearmanr(df[metric], df[target])
print(f'{r:,.2%} ({p:.2%})')


# In[265]:


## ATR

metric = 'atr'
j=sns.jointplot(x=metric, y=target, data=top100)

df = top100[[metric, target]].dropna()
r, p = spearmanr(df[metric], df[target])
print(f'{r:,.2%} ({p:.2%})')


# In[266]:


## MACD

metric = 'macd'
j=sns.jointplot(x=metric, y=target, data=top100)

df = top100[[metric, target]].dropna()
r, p = spearmanr(df[metric], df[target])
print(f'{r:,.2%} ({p:.2%})')


# In[267]:


## STOCK PREDICTION (LINEAR REGRESSION) ##


# In[268]:


sns.set_style('darkgrid')
idx = pd.IndexSlice


# In[269]:


YEAR = 252


# In[270]:


## LOAD DATA

with pd.HDFStore('data.h5') as store:
    data = (store['model_data']
            .dropna()
            .drop(['open', 'close', 'low', 'high'], axis=1))


# In[271]:


data.index.names = ['symbol', 'date']


# In[272]:


data = data.drop([c for c in data.columns if 'lag' in c], axis=1)


# In[273]:


## SELECT INVESTMENT PORTFOLIO RANGE

data = data[data.dollar_vol_rank<100]


# In[274]:


data.info(null_counts=True)


# In[275]:


## Creating Model Data

y = data.filter(like='target')
X = data.drop(y.columns, axis=1)
X = X.drop(['dollar_vol', 'dollar_vol_rank', 'volume', 'consumer_durables'], axis=1)


# In[276]:


class MultipleTimeSeriesCV:
    """Generates tuples of train_idx, test_idx pairs
    Assumes the MultiIndex contains levels 'symbol' and 'date'
    purges overlapping outcomes"""

    def __init__(self,
                 n_splits=3,
                 train_period_length=126,
                 test_period_length=21,
                 lookahead=None,
                 shuffle=False):
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle

    def split(self, X, y=None, groups=None):
        unique_dates = X.index.get_level_values('date').unique()
        days = sorted(unique_dates, reverse=True)

        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            split_idx.append([train_start_idx, train_end_idx,
                              test_start_idx, test_end_idx])

        dates = X.reset_index()[['date']]
        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = dates[(dates.date > days[train_start])
                              & (dates.date <= days[train_end])].index
            test_idx = dates[(dates.date > days[test_start])
                             & (dates.date <= days[test_end])].index
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            yield train_idx, test_idx

    def get_n_splits(self, X, y, groups=None):
        return self.n_splits


# In[277]:


## Check that it works

train_period_length = 63
test_period_length = 10
n_splits = int(3 * YEAR/test_period_length)
lookahead =1 

cv = MultipleTimeSeriesCV(n_splits=n_splits,
                          test_period_length=test_period_length,
                          lookahead=lookahead,
                          train_period_length=train_period_length)


# In[278]:


i = 0
for train_idx, test_idx in cv.split(X=data):
    train = data.iloc[train_idx]
    train_dates = train.index.get_level_values('date')
    test = data.iloc[test_idx]
    test_dates = test.index.get_level_values('date')
    df = train.reset_index().append(test.reset_index())
    n = len(df)
    assert n== len(df.drop_duplicates())
    print(train.groupby(level='symbol').size().value_counts().index[0],
          train_dates.min().date(), train_dates.max().date(),
          test.groupby(level='symbol').size().value_counts().index[0],
          test_dates.min().date(), test_dates.max().date())
    i += 1
    if i == 10:
        break


# In[ ]:





# In[279]:


## Prediction versus actual scatter plot

def plot_preds_scatter(df, ticker=None):
    if ticker is not None:
        idx = pd.IndexSlice
        df = df.loc[idx[ticker, :], :]
    j = sns.jointplot(x='predicted', y='actuals',
                      robust=True, ci=None,
                      line_kws={'lw': 1, 'color': 'k'},
                      scatter_kws={'s': 1},
                      data=df,
                      kind='reg')
    j.ax_joint.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    j.ax_joint.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
    j.ax_joint.set_xlabel('Predicted')
    j.ax_joint.set_ylabel('Actuals')


# In[280]:


## Daily IC Distribution

def plot_ic_distribution(df, ax=None):
    if ax is not None:
        sns.distplot(df.ic, ax=ax)
    else:
        ax = sns.distplot(df.ic)
    mean, median = df.ic.mean(), df.ic.median()
    ax.axvline(0, lw=1, ls='--', c='k')
    ax.text(x=.05, y=.9,
            s=f'Mean: {mean:8.2f}\nMedian: {median:5.2f}',
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlabel('Information Coefficient')
    sns.despine()
    plt.tight_layout()


# In[281]:


## Rolling Daily IC

def plot_rolling_ic(df):
    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(14, 8))
    rolling_result = df.sort_index().rolling(21).mean().dropna()
    mean_ic = df.ic.mean()
    rolling_result.ic.plot(ax=axes[0],
                           title=f'Information Coefficient (Mean: {mean_ic:.2f})',
                           lw=1)
    axes[0].axhline(0, lw=.5, ls='-', color='k')
    axes[0].axhline(mean_ic, lw=1, ls='--', color='k')

    mean_rmse = df.rmse.mean()
    rolling_result.rmse.plot(ax=axes[1],
                             title=f'Root Mean Squared Error (Mean: {mean_rmse:.2%})',
                             lw=1,
                             ylim=(0, df.rmse.max()))
    axes[1].axhline(df.rmse.mean(), lw=1, ls='--', color='k')
    sns.despine()
    plt.tight_layout()


# In[282]:


## Linear Regression using sklearn ##
# Cross-validation setup

train_period_length = 63
test_period_length = 10
n_splits = int(3 * YEAR / test_period_length)
lookahead = 1

cv = MultipleTimeSeriesCV(n_splits=n_splits,
                          test_period_length=test_period_length,
                          lookahead=lookahead,
                          train_period_length=train_period_length)


# In[283]:


get_ipython().run_cell_magic('time', '', "target = f'target_{lookahead}d'\nlr_predictions, lr_scores = [], []\nlr = LinearRegression()\nfor i, (train_idx, test_idx) in enumerate(cv.split(X), 1):\n    X_train, y_train, = X.iloc[train_idx], y[target].iloc[train_idx]\n    X_test, y_test = X.iloc[test_idx], y[target].iloc[test_idx]\n    lr.fit(X=X_train, y=y_train)\n    y_pred = lr.predict(X_test)\n\n    preds = y_test.to_frame('actuals').assign(predicted=y_pred)\n    preds_by_day = preds.groupby(level='date')\n    scores = pd.concat([preds_by_day.apply(lambda x: spearmanr(x.predicted,\n                                                               x.actuals)[0] * 100)\n                        .to_frame('ic'),\n                        preds_by_day.apply(lambda x: np.sqrt(mean_squared_error(y_pred=x.predicted,\n                                                                                y_true=x.actuals)))\n                        .to_frame('rmse')], axis=1)\n\n    lr_scores.append(scores)\n    lr_predictions.append(preds)\n\nlr_scores = pd.concat(lr_scores)\nlr_predictions = pd.concat(lr_predictions)")


# In[284]:


## Persist Results

lr_scores.to_hdf('data.h5', 'lr/scores')
lr_predictions.to_hdf('data.h5', 'lr/predictions')


# In[285]:


lr_scores = pd.read_hdf('data.h5', 'lr/scores')
lr_predictions = pd.read_hdf('data.h5', 'lr/predictions')


# In[286]:


## Evaluation of Results

lr_r, lr_p = spearmanr(lr_predictions.actuals, lr_predictions.predicted)
print(f'Information Coefficient (overall): {lr_r:.3%} (p-value: {lr_p:.4%})')


# In[287]:


# Scatter Plot: Predictions vs Actuals

plot_preds_scatter(lr_predictions)


# In[288]:


plot_ic_distribution(lr_scores)


# In[289]:


plot_rolling_ic(lr_scores)


# In[290]:


## Loading Price Data Again

with pd.HDFStore('data.h5') as store:
    data = (store['model_data']
            .dropna()
            .drop(store['model_data'].iloc[:, 4:69], axis=1))


# In[291]:


#lr_predictions
jk = lr_predictions.loc['MSFT','actuals' :'predicted']
jk_actuals = jk[jk.actuals > 0.03]
jk_predicted = jk[jk.predicted > 0.01]
pred_dates = lr_predictions.index.get_level_values('date').unique()
#dates = sorted(unique_dates, reverse=True)
pred_dates = pred_dates[::-1]
pred_dates = sorted(pred_dates)


test = pd.concat([jk, jk_predicted], axis = 1)
no_nans = test[~test.isnull().any(axis=1)]
test.columns = ['actuals', 'predicted', 'act_threshold', 'pred_threshold']
test = pd.merge(test, data.loc['AAPL'], left_index=True, right_index = True)


# In[292]:


budget = 10000
pct = 0.02
i = 0
a = 0
pred_dates = pd.DataFrame(pred_dates)
for index in pred_dates.iterrows():
    abc = test.iloc[i, 3]
    buy = test.iloc[i, 4]
    sell = test.iloc[i, 5]
    if np.isnan(abc) == False :
        invest = budget
        #invest = pct * budget
        budget = budget - invest
        shares = invest / buy
        result = shares * sell
        profit = result - invest
        budget = budget + result
        print(a+1, 'For ', index)
        print('Budget: ', budget)
        print('Profit: ', profit)
        print('Shares: ', shares)
        a = a + 1
    i += 1


#unique_dates = X.index.get_level_values('date').unique()
#sorted(unique_dates, reverse=True)


# In[ ]:


with pd.HDFStore('data.h5') as store:
    data = (store['model_data']
            .dropna()
            .drop(store['model_data'].iloc[:, 4:69], axis=1))


# In[ ]:


data


# In[ ]:


test.loc['2015-08-27',:]


# In[ ]:


np.isnan(test.iloc[:,3])


# In[ ]:


test


# In[ ]:


## WORKFLOW BACKTESTING ##


# In[293]:


df = (pd.read_csv('wiki_prices.csv',
                 parse_dates=['date'],
                 index_col=['date', 'ticker'],
                 infer_datetime_format=True)
     .sort_index())

print(df.info(null_counts=True))
with pd.HDFStore(DATA_STORE) as store:
    store.put('quandl/wiki/prices', df)


# In[294]:


## Ridge Regression ##

ridge_alphas = np.logspace(-4, 4, 9)
ridge_alphas = sorted(list(ridge_alphas) + list(ridge_alphas * 5))


# In[295]:


n_splits = int(3 * YEAR/test_period_length)
train_period_length = 63
test_period_length = 10
lookahead = 1

cv = MultipleTimeSeriesCV(n_splits=n_splits,
                          test_period_length=test_period_length,
                          lookahead=lookahead,
                          train_period_length=train_period_length)


# In[296]:


# Ridge regression cross-validation

target = f'target_{lookahead}d'

X = X.drop([c for c in X.columns if 'year' in c], axis=1)


# In[297]:


get_ipython().run_cell_magic('time', '', "ridge_coeffs, ridge_scores, ridge_predictions = {}, [], []\n\nfor alpha in ridge_alphas:\n    print(alpha, end=' ', flush=True)\n    start = time()\n    model = Ridge(alpha=alpha,\n                  fit_intercept=False,\n                  random_state=42)\n\n    pipe = Pipeline([\n        ('scaler', StandardScaler()),\n        ('model', model)])\n\n    coeffs = []\n    for i, (train_idx, test_idx) in enumerate(cv.split(X), 1):\n        X_train, y_train, = X.iloc[train_idx], y[target].iloc[train_idx]\n        X_test, y_test = X.iloc[test_idx], y[target].iloc[test_idx]\n\n        pipe.fit(X=X_train, y=y_train)\n        y_pred = pipe.predict(X_test)\n\n        preds = y_test.to_frame('actuals').assign(predicted=y_pred)\n        preds_by_day = preds.groupby(level='date')\n        scores = pd.concat([preds_by_day.apply(lambda x: spearmanr(x.predicted,\n                                                                   x.actuals)[0] * 100)\n                            .to_frame('ic'),\n                            preds_by_day.apply(lambda x: np.sqrt(mean_squared_error(y_pred=x.predicted,\n                                                                                    y_true=x.actuals)))\n                            .to_frame('rmse')], axis=1)\n\n        ridge_scores.append(scores.assign(alpha=alpha))\n        ridge_predictions.append(preds.assign(alpha=alpha))\n\n        coeffs.append(pipe.named_steps['model'].coef_)\n    ridge_coeffs[alpha] = np.mean(coeffs, axis=0)\n\nprint('\\n')")


# In[298]:


# Persist Results

ridge_scores = pd.concat(ridge_scores)
ridge_scores.to_hdf('data.h5', 'ridge/scores')

ridge_coeffs = pd.DataFrame(ridge_coeffs, index=X.columns).T
ridge_coeffs.to_hdf('data.h5', 'ridge/coeffs')

ridge_predictions = pd.concat(ridge_predictions)
ridge_predictions.to_hdf('data.h5', 'ridge/predictions')


# In[299]:


ridge_scores = pd.read_hdf('data.h5', 'ridge/scores')
ridge_coeffs = pd.read_hdf('data.h5', 'ridge/coeffs')
ridge_predictions = pd.read_hdf('data.h5', 'ridge/predictions')


# In[ ]:


## Lasso cross-validation ##


# In[300]:


lasso_alphas = np.logspace(-10, -3, 8)


# In[301]:


train_period_length = 63
test_period_length = 10
YEAR = 252
n_splits = int(3 * YEAR / test_period_length) # three years
lookahead = 1


# In[302]:


cv = MultipleTimeSeriesCV(n_splits=n_splits,
                          test_period_length=test_period_length,
                          lookahead=lookahead,
                          train_period_length=train_period_length)


# In[ ]:


## Running cross-validation with Lasso regression ##


# In[303]:


target = f'target_{lookahead}d'

scaler = StandardScaler()
X = X.drop([c for c in X.columns if 'year' in c], axis=1)


# In[304]:


get_ipython().run_cell_magic('time', '', "\nlasso_coeffs, lasso_scores, lasso_predictions = {}, [], []\nfor alpha in lasso_alphas:\n    print(alpha, end=' ', flush=True)\n    model = Lasso(alpha=alpha,\n                  fit_intercept=False,  # StandardScaler centers data\n                  random_state=42,\n                  tol=1e-3,\n                  max_iter=1000,\n                  warm_start=True,\n                  selection='random')\n\n    pipe = Pipeline([\n        ('scaler', StandardScaler()),\n        ('model', model)])\n    coeffs = []\n    for i, (train_idx, test_idx) in enumerate(cv.split(X), 1):\n        t = time()\n        X_train, y_train, = X.iloc[train_idx], y[target].iloc[train_idx]\n        X_test, y_test = X.iloc[test_idx], y[target].iloc[test_idx]\n\n        pipe.fit(X=X_train, y=y_train)\n        y_pred = pipe.predict(X_test)\n\n        preds = y_test.to_frame('actuals').assign(predicted=y_pred)\n        preds_by_day = preds.groupby(level='date')\n        scores = pd.concat([preds_by_day.apply(lambda x: spearmanr(x.predicted,\n                                                                   x.actuals)[0] * 100)\n                            .to_frame('ic'),\n                            preds_by_day.apply(lambda x: np.sqrt(mean_squared_error(y_pred=x.predicted,\n                                                                                    y_true=x.actuals)))\n                            .to_frame('rmse')],\n                           axis=1)\n\n        lasso_scores.append(scores.assign(alpha=alpha))\n        lasso_predictions.append(preds.assign(alpha=alpha))\n\n        coeffs.append(pipe.named_steps['model'].coef_)\n\n    lasso_coeffs[alpha] = np.mean(coeffs, axis=0)")


# In[305]:


## Persist lasso results
lasso_scores = pd.concat(lasso_scores)
lasso_scores.to_hdf('data.h5', 'lasso/scores')

lasso_coeffs = pd.DataFrame(lasso_coeffs, index=X.columns).T
lasso_coeffs.to_hdf('data.h5', 'lasso/coeffs')

lasso_predictions = pd.concat(lasso_predictions)
lasso_predictions.to_hdf('data.h5', 'lasso/predictions')


# In[332]:


best_alpha = lasso_scores.groupby('alpha').ic.mean().idxmax()
preds = lasso_predictions[lasso_predictions.alpha==best_alpha]

lasso_r, lasso_p = spearmanr(preds.actuals, preds.predicted)
print(f'Information Coefficient (overall): {lasso_r:.3%} (p-value: {lasso_p:.4%})')


# In[333]:


lasso_scores.groupby('alpha').ic.agg(['mean', 'median'])


# In[ ]:


## Logistic Regression ##


# In[311]:


sys.path.insert(1, os.path.join(sys.path[0], '..'))
#from utils import MultipleTimeSeriesCV
YEAR = 252


# In[312]:


## Loading data

with pd.HDFStore('data.h5') as store:
    data = (store['model_data']
            .dropna()
            .drop(['open', 'close', 'low', 'high'], axis=1))
data = data.drop([c for c in data.columns if 'year' in c or 'lag' in c], axis=1)


# In[313]:


# Select investment universe
data = data[data.dollar_vol_rank<100]


# In[314]:


# creating model data
y = data.filter(like='target')
X = data.drop(y.columns, axis=1)
X = X.drop(['dollar_vol', 'dollar_vol_rank', 'volume', 'consumer_durables'], axis=1)


# In[315]:


# Defining cross-valiation parameters
train_period_length = 63
test_period_length = 10
lookahead =1
n_splits = int(3 * YEAR/test_period_length)

cv = MultipleTimeSeriesCV(n_splits=n_splits,
                          test_period_length=test_period_length,
                          lookahead=lookahead,
                          train_period_length=train_period_length)


# In[316]:


target = f'target_{lookahead}d'


# In[317]:


y.loc[:, 'label'] = (y[target] > 0).astype(int)
y.label.value_counts()


# In[318]:


Cs = np.logspace(-5, 5, 11)
cols = ['C', 'date', 'auc', 'ic', 'pval']


# In[328]:


## Running cross-validation

#%%time
log_coeffs, log_scores, log_predictions = {}, [], []
for C in Cs:
    print(C)
    model = LogisticRegression(C=C,
                               fit_intercept=True,
                               random_state=42,
                               n_jobs=-1)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)])
    ics = aucs = 0
    start = time()
    coeffs = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        X_train, y_train, = X.iloc[train_idx], y.label.iloc[train_idx]
        pipe.fit(X=X_train, y=y_train)
        X_test, y_test = X.iloc[test_idx], y.label.iloc[test_idx]
        actuals = y[target].iloc[test_idx]
        if len(y_test) < 10 or len(np.unique(y_test)) < 2:
            continue
        y_score = pipe.predict_proba(X_test)[:, 1]
       
        auc = roc_auc_score(y_score=y_score, y_true=y_test)
        actuals = y[target].iloc[test_idx]
        ic, pval = spearmanr(y_score, actuals)

        log_predictions.append(y_test.to_frame('labels').assign(
            predicted=y_score, C=C, actuals=actuals))
        date = y_test.index.get_level_values('date').min()
        log_scores.append([C, date, auc, ic * 100, pval])
        coeffs.append(pipe.named_steps['model'].coef_)
        ics += ic
        aucs += auc
        if i % 10 == 0:
            print(f'\t{time()-start:5.1f} | {i:03} | {ics/i:>7.2%} | {aucs/i:>7.2%}')

    log_coeffs[C] = np.mean(coeffs, axis=0).squeeze()


# In[329]:


## Results Evaluation

log_scores = pd.DataFrame(log_scores, columns=cols)
log_scores.to_hdf('data.h5', 'logistic/scores')

log_coeffs = pd.DataFrame(log_coeffs, index=X.columns).T
log_coeffs.to_hdf('data.h5', 'logistic/coeffs')

log_predictions = pd.concat(log_predictions)
log_predictions.to_hdf('data.h5', 'logistic/predictions')


# In[330]:


log_scores = pd.read_hdf('data.h5', 'logistic/scores')


# In[ ]:


## Workflows ##


# In[337]:


with pd.HDFStore('data.h5') as store:
    lr_predictions = store['lr/predictions']
    lasso_predictions = store['lasso/predictions']
    lasso_scores = store['lasso/scores']
    ridge_predictions = store['ridge/predictions']
    ridge_scores = store['ridge/scores']


# In[362]:


pd.set_option('display.expand_frame_repr', False)
np.random.seed(42)

PROJECT_DIR = Path('..', '..')

DATA_DIR = PROJECT_DIR / 'data'


def get_backtest_data(predictions='lasso/predictions'):
    """Combine chapter 7 lr/lasso/ridge regression predictions
        with adjusted OHLCV Quandl Wiki data"""
    with pd.HDFStore(DATA_DIR / 'assets.h5') as store:
        prices = (store['quandl/wiki/prices']
                  .filter(like='adj')
                  .rename(columns=lambda x: x.replace('adj_', ''))
                  .swaplevel(axis=0))

    with pd.HDFStore('data.h5') as store:
        print(store.info())
        predictions = store[predictions]

    best_alpha = predictions.groupby('alpha').apply(lambda x: spearmanr(x.actuals, x.predicted)[0]).idxmax()
    predictions = predictions[predictions.alpha == best_alpha]
    predictions.index.names = ['ticker', 'date']
    tickers = predictions.index.get_level_values('ticker').unique()
    start = predictions.index.get_level_values('date').min().strftime('%Y-%m-%d')
    stop = (predictions.index.get_level_values('date').max() + pd.DateOffset(1)).strftime('%Y-%m-%d')
    idx = pd.IndexSlice
    prices = prices.sort_index().loc[idx[tickers, start:stop], :]
    predictions = predictions.loc[predictions.alpha == best_alpha, ['predicted']]
    return predictions.join(prices, how='right')


df = get_backtest_data('lasso/predictions')
print(df.info())
df.to_hdf('backtest.h5', 'data')


# In[363]:


## Vectorized Backtesting
DATA_DIR = Path('..', 'data')
data = pd.read_hdf('backtest.h5', 'data')
data.info()


# In[364]:


## SP500 benchmark

sp500 = web.DataReader('SP500', 'fred', '2014', '2018').pct_change()


# In[350]:


sp500.info()


# In[365]:


## Computing Forward Returns

daily_returns = data.open.unstack('ticker').sort_index().pct_change()
daily_returns.info()


# In[366]:


fwd_returns = daily_returns.shift(-1)


# In[367]:


## Generating signals
predictions = data.predicted.unstack('ticker')
predictions.info()


# In[368]:


N_LONG = N_SHORT = 15


# In[405]:


long_signals = ((predictions
                .where(predictions > -0.01)
                .rank(axis=1, ascending=False) > N_LONG)
                .astype(int))
short_signals = ((predictions
                  .where(predictions < -0.01)
                  .rank(axis=1) > N_SHORT)
                 .astype(int))


# In[406]:


long_returns = long_signals.mul(fwd_returns).mean(axis=1)
short_returns = short_signals.mul(-fwd_returns).mean(axis=1)
strategy = long_returns.add(short_returns).to_frame('Strategy')


# In[407]:


fig, axes = plt.subplots(ncols=2, figsize=(14,5))
strategy.join(sp500).add(1).cumprod().sub(1).plot(ax=axes[0], title='Cumulative Return')
sns.distplot(strategy.dropna(), ax=axes[1], hist=False, label='Strategy')
sns.distplot(sp500, ax=axes[1], hist=False, label='SP500')
axes[1].set_title('Daily Standard Deviation')
axes[0].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
axes[1].xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
sns.despine()
fig.tight_layout();


# In[385]:


res = strategy.join(sp500).dropna()


# In[386]:


res.std()


# In[374]:


predictions

