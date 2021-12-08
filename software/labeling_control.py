# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 21:52:40 2021

@author: danny

Takes code written for BE/Bi 103a hw 7.1 and puts it into a python script
so it can be used in the website for hw 10.3. Looks at the data for 
microtubule catastrophe with labeled and unlabeled tubulin to see if there
is a difference in their distributions by plotting their ECDFs,
constructing confidence intervals for the mean time to catastrophe,
and performing a permutation test.
"""
import pandas as pd
import numpy as np
import bokeh.io
import holoviews as hv
import iqplot

hv.extension('bokeh')

# read in data
datapath = '../data/gardner_time_to_catastrophe_dic_tidy.csv'
df = pd.read_csv(datapath)

# plot ECDFs with confidence intervals
p1 = iqplot.ecdf(data = df, q = 'time to catastrophe (s)', cats = 'labeled', conf_int = True)
bokeh.io.show(p1)

def draw_bs_sample(data):
    '''Draw a bootstrap sample from data'''
    return np.random.choice(data, size=len(data))

def bs_mean(data, size = 1):
    '''Calculate the mean of a bootstrap sample from data, repeated 'size' times'''
    out = np.empty(size)
    for i in range(size):
        out[i] = np.mean(draw_bs_sample(data))
    return out

# extract time data
lab_array = df[df['labeled'] == True]['time to catastrophe (s)']
unlab_array = df[df['labeled'] == False]['time to catastrophe (s)']

# get bootstrap replicates of the mean
lab_bs_means = bs_mean(lab_array, size = 100)
unlab_bs_means = bs_mean(unlab_array, size = 100)

# find confidence intervals
lab_mean_ci = np.percentile(lab_bs_means, [2.5, 97.5])
unlab_mean_ci = np.percentile(unlab_bs_means, [2.5, 97.5])
print('Confidence interval for mean of labeled sample:[{0:.2f}, {1:.2f}]\nConfidence interval for mean of unlabeled sample:[{2:.2f}, {3:.2f}]'
      .format(*(tuple(lab_mean_ci) + tuple(unlab_mean_ci))))

def draw_perm_sample(x, y):
    '''
    Randomly reassigns each data point into one of two 
    arrays of the same size as the input arrays
    '''
    concat_data = np.concatenate((x, y))
    np.random.shuffle(concat_data)
    return concat_data[:len(x)], concat_data[len(y):]

def diff_mean_replicates(x, y, size=1):
    '''
    Calculates the difference in mean for two permuted samples
    of x and y repeated 'size' times.
    '''
    out = np.empty(size)
    for i in range(size):
        x_perm, y_perm = draw_perm_sample(x, y)
        out[i] = np.mean(x_perm) - np.mean(y_perm)

    return out

# actual difference in means
diff_mean = np.mean(lab_array) - np.mean(unlab_array)
print(f'Observed difference in means: {diff_mean}')

# perform permutation test
perm_reps = diff_mean_replicates(lab_array, unlab_array, size=1000000)
p_val = np.sum(perm_reps >= diff_mean) / len(perm_reps)
print('p-value =', p_val)
