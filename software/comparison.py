# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 22:59:50 2021

@author: danny

Looks at the data with a 12 uM tubulin concentration to perform model 
comparison between the gamma-distributed model and the two-step model
using Q-Q plots, predictive ECDFs, and the Akaike Information Criterion.
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

# load data
df = pd.read_csv('../data/gardner_mt_catastrophe_only_tubulin.csv', skiprows=9)
df = df.melt(var_name='concentration', value_name='time').dropna()

# exploratory ECDF
p1 = iqplot.ecdf(df, q='time', cats='concentration', conf_int=True,
                 order=['7 uM', '9 uM', '10 uM', '12 uM', '14 uM'],
                 marker_kwargs={'size':2})
bokeh.io.show(p1)

# extract 12 uM data
df12 = df[df['concentration'] == '12 uM']
d12 = df12['time'].to_numpy()

def generate_sample(model, p1, p2, size):
    '''
    Draws a sample of size "size" from the given model (gamma = 1 or two-step = 2),
    parameterized by the given parameters p1 and p2.
    For gamma, p1 = alpha, p2 = beta
    For two-step, p1 = beta1, p2 = beta2
    '''
    rg = np.random.default_rng()
    if model == 1:
        return rg.gamma(p1, 1/p2, size)
    return rg.exponential(1/p1, size) + rg.exponential(1/p2, size)

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

def log_beta_param(log_beta,n):
    '''
    Computes the log likelihood for the two-step model with
    the given logs of the parameters and the data
    '''
    log_beta1,log_beta2 = log_beta
    beta1 = np.exp(log_beta1)
    beta2 = np.exp(log_beta2)
    
    if beta1 == beta2:
        log_alpha, log_beta1 = log_beta
        return np.sum(scipy.stats.gamma.logpdf(n, 2, scale = 1/beta1))
    
    elif beta1 > beta2:
        return -np.inf
    
    else:
        beta_array = [-beta1 * n, -beta2 * n]
        
        b_array = []
        for _ in range(len(n)):
                       b_array.append([1,-1])
        b_final = np.array(b_array).T
        
        return np.sum(np.log((beta1 * beta2)/(beta2 - beta1)) + scipy.special.logsumexp(beta_array, b = b_final))

def beta_mle(n):
    '''Computes MLEs for beta1 and beta2'''
    res = scipy.optimize.minimize(
        fun=lambda log_param, n: -log_beta_param(log_param, n),
        x0=np.array([0.05, 0.05]),
        args=(n),
        method='Nelder-Mead'
    )
    return res.x

# compute MLEs for parameters
gamma_params = np.exp(mle_log_param(d12))
two_step_params = np.exp(beta_mle(d12))
alpha = gamma_params[0]
beta = gamma_params[1]
beta1 = two_step_params[0]
beta2 = two_step_params[1]
print(f'Gamma parameters: {gamma_params}')
print(f'Two-step parameters: {two_step_params}')

# make Q-Q plot
gamma_sample = np.array([generate_sample(1, alpha, beta, d12.size) for i in range(10000)])
qq_gamma = bebi103.viz.qqplot(gamma_sample, d12,
                              x_axis_label='time', y_axis_label='time')
two_step_sample = np.array([generate_sample(2, beta1, beta2, d12.size) for i in range(10000)])
qq_two_step = bebi103.viz.qqplot(two_step_sample, d12,
                                 x_axis_label='time', y_axis_label='time')
bokeh.io.show(bokeh.layouts.row(qq_gamma, qq_two_step))

# make predictive ECDFs
pred_ecdf_gamma = bebi103.viz.predictive_ecdf(gamma_sample, d12, 
                                              discrete=True, x_axis_label='time')
pred_ecdf_two_step = bebi103.viz.predictive_ecdf(two_step_sample, d12,
                                                 discrete=True, x_axis_label='time')
bokeh.io.show(bokeh.layouts.row(pred_ecdf_gamma, pred_ecdf_two_step))

# calculate AICs
AIC_gamma = -2*log_param_pdf(np.log(gamma_params), d12) + 4
AIC_two_step = -2*log_beta_param(np.log(two_step_params), d12) + 4
print(f'Gamma model AIC: {AIC_gamma}')
print(f'Two-step model AIC: {AIC_two_step}')

# calculate Akaike weights
weight_gamma = 1 / (1 + np.exp((AIC_gamma - AIC_two_step) / 2))
weight_two_step = np.exp((AIC_gamma - AIC_two_step) / 2) / (1 + np.exp((AIC_gamma - AIC_two_step) / 2))
print(f'Gamma model weight: {weight_gamma}')
print(f'Two-step model weight: {weight_two_step}')
