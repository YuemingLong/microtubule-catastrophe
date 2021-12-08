# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 23:03:41 2021

@author: danny

Looks at the catastrophe times for different concentrations of tubulin and
calculates maximum likelihood estimates for the gamma-distributed model
for each concentration.
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import bokeh.io
import bokeh.layouts
import iqplot
import bebi103
import scipy.optimize
import scipy.special

# need later
rg = np.random.default_rng(1344)

# load data
df = pd.read_csv('../data/gardner_mt_catastrophe_only_tubulin.csv', skiprows=9)
df = df.melt(var_name='concentration', value_name='time').dropna()

def log_param_pdf(log_param,n):
    '''
    Computes the log likelihood
    given the log of the gamma distribution parameters and data
    '''
    log_alpha, log_beta = log_param
    
    alpha = np.exp(log_alpha)
    beta = np.exp(log_beta)
    return np.sum(scipy.stats.gamma.logpdf(n, alpha, scale = 1/beta))

def mle_log_param(n):
    '''Computes the MLEs for the gamma distributed model'''
    res = scipy.optimize.minimize(
        fun=lambda log_param, n: -log_param_pdf(log_param, n),
        x0=np.array([2, 0.1]),
        args=(n),
        method='Nelder-Mead'
    )
    return res.x

def draw_bs(data):
    return rg.choice(data, size = len(data))

def bs_mle_rep(mle_fun, data, size = 1):
    return np.exp(np.array([mle_fun(draw_bs(data)) for _ in range(size)]))

def compute_ci(data):
    bs_reps = bs_mle_rep(mle_log_param, data, size=1000)
    return np.percentile(bs_reps, [2.5, 97.5], axis=0)


# extract data
data = []
data.append((7, df[df['concentration'] == '7 uM']['time'].to_numpy()))
data.append((9, df[df['concentration'] == '9 uM']['time'].to_numpy()))
data.append((10, df[df['concentration'] == '10 uM']['time'].to_numpy()))
data.append((12, df[df['concentration'] == '12 uM']['time'].to_numpy()))
data.append((14, df[df['concentration'] == '14 uM']['time'].to_numpy()))

# calculate MLEs and confidence intervals
alphas = []
betas = []
alpha_lows = []
alpha_highs = []
beta_lows = []
beta_highs = []
for concentration in data:
    print(f'{concentration[0]} uM:')
    params = np.exp(mle_log_param(concentration[1]))
    alphas.append(params[0])
    betas.append(params[1])
    conf_int = compute_ci(concentration[1])
    alpha_lows.append(conf_int[0,0])
    alpha_highs.append(conf_int[1,0])
    beta_lows.append(conf_int[0,1])
    beta_highs.append(conf_int[1,1])
    print(f'\talpha: {params[0]}')
    print(f'\talpha 95% CI: {conf_int[:,0]}')
    print(f'\tbeta: {params[1]}')
    print(f'\tbeta 95% CI: {conf_int[:,1]}')
    
# plot results
df_alphas = pd.DataFrame({'Concentration':[7, 9, 10, 12, 14],
                          'Alpha':alphas,
                          'Alpha Low':alpha_lows,
                          'Alpha High':alpha_highs})
df_betas = pd.DataFrame({'Concentration':[7, 9, 10, 12, 14],
                          'Beta':betas,
                          'Beta Low':beta_lows,
                          'Beta High':beta_highs})
df_alphas = df_alphas.melt(id_vars='Concentration', value_name='alpha')
df_betas = df_betas.melt(id_vars='Concentration', value_name='beta')
p2 = iqplot.strip(df_alphas, q='alpha', cats='Concentration', 
                  color_column='variable', q_axis='y', x_axis_label='concentration')
p3 = iqplot.strip(df_betas, q='beta', cats='Concentration', 
                  color_column='variable', q_axis='y', x_axis_label='concentration')
bokeh.io.show(bokeh.layouts.row(p2, p3))