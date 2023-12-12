#!/usr/bin/env python
# coding: utf-8

# In[3]:


import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
import os
import datetime as dt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.ticker import FuncFormatter

import pytz
import pandas_datareader.data as web
import pymc3 as pm
import arviz
from scipy import stats
from matplotlib import gridspec


import yfinance as yf
import datetime as dt
import pandas_datareader.data as web
import pytz

from pathlib import Path
import requests
from io import BytesIO
from zipfile import ZipFile, BadZipFile
from sklearn.datasets import fetch_openml

pd.set_option('display.expand_frame_repr', False)


# In[4]:


import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

np.random.seed(42)
sns.set_style('dark')


# In[ ]:


def format_plot(axes, i, p, y, trials, success, true_p, tmle, tmap=None):
    fmt = FuncFormatter(lambda x, _: f'{x:.0%}')
    if i >= 6:
        axes[i].set_xlabel("$p$, Success Probability")
        axes[i].xaxis.set_major_formatter(fmt)
    else:
        axes[i].axes.get_xaxis().set_visible(False)
    if i % 3 == 0:
        axes[i].set_ylabel("Posterior Probability")
    axes[i].set_yticks([])

    axes[i].plot(p, y, lw=1, c='k')
    axes[i].fill_between(p, y, color='darkblue', alpha=0.4)
    axes[i].vlines(true_p, 0, max(10, np.max(y)), color='k', linestyle='--', lw=1)
    axes[i].set_title(f'Trials: {trials:,d} - Success: {success:,d}')
    if i > 0:
        smle = r"$\theta_{{\mathrm{{MLE}}}}$ = {:.2%}".format(tmle)
        axes[i].text(x=.02, y=.85, s=smle, transform=axes[i].transAxes)
        smap = r"$\theta_{{\mathrm{{MAP}}}}$ = {:.2%}".format(tmap)
        axes[i].text(x=.02, y=.75, s=smap, transform=axes[i].transAxes)    
    return axes[i]


# In[ ]:


n_trials = [0, 1, 3, 5, 10, 25, 50, 100, 500]
outcomes = stats.bernoulli.rvs(p=0.5, size=n_trials[-1])
p = np.linspace(0, 1, 100)
# uniform (uninformative) prior
a = b = 1


# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 7), sharex=True)
axes = axes.flatten()
fmt = FuncFormatter(lambda x, _: f'{x:.0%}')
for i, trials in enumerate(n_trials):
    successes = outcomes[:trials]
    theta_mle = np.mean(successes)
    heads = sum(successes)
    tails = trials - heads
    update = stats.beta.pdf(p, a + heads , b + tails)
    theta_map = pd.Series(update, index=p).idxmax()
    axes[i] = format_plot(axes, i, p, update, trials=trials, success=heads, 
                          true_p=.5, tmle=theta_mle, tmap=theta_map)

title = 'Bayesian Probabilities: Updating the Posterior'
fig.suptitle(title,  y=1.02, fontsize=14)
fig.tight_layout()


# In[ ]:


## 4.Redoing Above logic with open S&P 500 prices over the last 61 trading days

sp500_returns = pd.read_csv('HistoricalPrices.csv')
#sp500_returns.iloc[:, 1]
sp500_binary = (sp500_returns.iloc[:, 1].pct_change().dropna() > 0).astype(int)


# In[ ]:


n_days = [0, 1, 3, 5, 10, 25, 50]
# random sample of trading days
# outcomes = sp500_binary.sample(n_days[-1])

# initial 500 trading days
outcomes = sp500_binary.iloc[:n_days[-1]]
p = np.linspace(0, 1, 100)

# uniform (uninformative) prior
a = b = 1

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 7), sharex=True)
axes = axes.flatten()
for i, days in enumerate(n_days):
    successes = outcomes.iloc[:days]
    theta_mle = successes.mean()
    up = successes.sum()
    down = days - up
    update = stats.beta.pdf(p, a + up , b + down)
    theta_map = pd.Series(update, index=p).idxmax()
    axes[i] = format_plot(axes, i, p, update, trials=days, success=up, 
                          true_p=sp500_binary.mean(), tmle=theta_mle, tmap=theta_map)

title = 'Bayesian Probabilities: Updating the Posterior'
fig.suptitle(title,  y=1.02, fontsize=14)
fig.tight_layout()


# In[ ]:


benchmark = web.DataReader('SP500', data_source='fred', start=2010)
benchmark.columns = ['benchmark']


# In[ ]:


DATA_STORE = Path('assets.h5')

df = (pd.read_csv('wiki_prices.csv',
                 parse_dates=['date'],
                 index_col=['date', 'ticker'],
                 infer_datetime_format=True)
     .sort_index())

print(df.info(null_counts=True))
with pd.HDFStore(DATA_STORE) as store:
    stock = store['quandl/wiki/prices'].adj_close.unstack()['AMZN'].to_frame('stock')


# In[ ]:


data = stock.join(benchmark).pct_change().dropna().loc['2010':]


# In[ ]:


data.info()


# In[ ]:


mean_prior = data.stock.mean()
std_prior = data.stock.std()
std_low = std_prior / 1000
std_high = std_prior * 1000

with pm.Model() as sharpe_model:
    mean = pm.Normal('mean', mu=mean_prior, sd=std_prior)
    std = pm.Uniform('std', lower=std_low, upper=std_high)

    nu = pm.Exponential('nu_minus_two', 1 / 29, testval=4) + 2.
    returns = pm.StudentT('returns', nu=nu, mu=mean, sd=std, observed=data.stock)

    sharpe = returns.distribution.mean / returns.distribution.variance ** .5 * np.sqrt(252)
    pm.Deterministic('sharpe', sharpe)


# In[ ]:


sharpe_model.model


# In[2]:


#pm.model_to_graphviz(model=sharpe_model)


# In[15]:


tune = 2000
draws = 200
with sharpe_model:
    trace = pm.sample(tune=tune, 
                      draws=draws, 
                      chains=4, 
                      cores=1)


# In[19]:


trace_df = pm.trace_to_dataframe(trace).assign(chain=lambda x: x.index // draws)
trace_df.info()


# In[17]:


arviz.plot_trace(data=trace);


# In[18]:


## Continuing the sampling ##


# In[ ]:


draws = 25000
with sharpe_model:
    trace = pm.sample(draws=draws, 
                      trace=trace, 
                      chains=4, 
                      cores=1)


# In[1]:


pm.trace_to_dataframe(trace).shape


# In[ ]:


df = pm.trace_to_dataframe(trace).iloc[400:].reset_index(drop=True).assign(chain=lambda x: x.index // draws)
trace_df = pd.concat([trace_df.assign(run=1),
                      df.assign(run=2)])
trace_df.info()  


# In[ ]:


trace_df_long = pd.melt(trace_df, id_vars=['run', 'chain'])
trace_df_long.info()


# In[ ]:


g = sns.FacetGrid(trace_df_long, col='variable', row='run', hue='chain', sharex='col', sharey=False)
g = g.map(sns.distplot, 'value', hist=False, rug=False)


# In[ ]:


arviz.plot_trace(data=trace);


# In[ ]:


arviz.plot_posterior(data=trace)

