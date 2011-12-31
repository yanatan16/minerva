'''
Created on Dec 4, 2011

@author: Eisen
'''
import numpy as np
import pywt

def identity(data, params):
    return data

def mean(data, params):
    mdata = [np.mean(d) for d in data]
    return np.array(mdata)

def stdev(data, params):
    stdata = [np.std(d) for d in data]
    return np.array(stdata)

def max(data, params):
    maxdata = [np.max(d) for d in data]
    return np.array(maxdata)

def min(data, params):
    mindata = [np.min(d) for d in data]
    return np.array(mindata)

def dwt(data, params):
    dwtdata = []
    if params['wavelet'] == '':
        wavelet = 'db1'
    else:
        wavelet = params['wavelet']
    for d in data:
        (approx, detail) = pywt.dwt(d, wavelet)
        a = approx.reshape(approx.size)
        b = detail.reshape(detail.size)
        dwtdata += np.concatenate(a,b)
    return np.array(dwtdata)