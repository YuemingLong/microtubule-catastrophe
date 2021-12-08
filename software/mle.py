# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 22:13:16 2021

@author: danny

Takes code written for BE/Bi 103a hw 7.1 and puts it into a python script
so it can be used in the website for hw 10.3. Looks at the labeled tubulin 
data to compute the maximum likelihood estimates with confidence intervals
for the parameters of the gamma-distributed model and the two-step model.
"""

import pandas as pd
import numpy as np
import scipy.stats
import scipy.special

# load data
df_1 = pd.read_csv('../data/gardner_time_to_catastrophe_dic_tidy.csv')
df = df_1.loc[df_1['labeled'] == True]
n = df['time to catastrophe (s)'].values

# need later
rg = np.random.default_rng(1344)

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
    '''Draws a bootstrap sample of the data'''
    return rg.choice(data, size = len(data))

def bs_mle_rep(mle_fun, data, size = 1):
    '''Computes bootstrap replicates of the parameters'''
    return np.exp(np.array([mle_fun(draw_bs(data)) for _ in range(size)]))

# calculate MLEs with confidence intervals for gamma model
bs_reps = bs_mle_rep(mle_log_param, n, size=1000)
conf_int = np.percentile(bs_reps, [2.5, 97.5], axis=0).transpose()
mle = np.exp(mle_log_param(n))
print('alpha:', mle[0], conf_int[0])
print('beta:', mle[1], conf_int[1])

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
    
# calculate MLEs for the two-steo model
res = scipy.optimize.minimize(
        fun=lambda log_param, n: -log_beta_param(log_param, n),
        x0=np.array([0.05, 0.06]),
        args=(n),
        method='Nelder-Mead'
)
beta_val = np.exp(res.x)    
print('beta1:', beta_val[0])
print('beta2:', beta_val[1])


def beta_mle(n):
    '''Computes MLEs for beta1 and beta2'''
    res = scipy.optimize.minimize(
        fun=lambda log_param, n: -log_beta_param(log_param, n),
        x0=np.array([0.05, 0.05]),
        args=(n),
        method='Nelder-Mead'
    )
    return res.x

# calculate confidence intervals for two-step model
beta_rep = bs_mle_rep(beta_mle, n, size=1000)
beta_conf_int = np.percentile(beta_rep, [2.5, 97.5], axis=0).transpose()
print('beta1:', beta_val[0], beta_conf_int[0])
print('beta2:', beta_val[1], beta_conf_int[1])