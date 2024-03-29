'''
Created on Dec 4, 2011

@author: Eisen
'''
import numpy as np
import pywt as wt

# Helper functions
def generator_return(toret):
    return np.reshape(toret, (-1))    

## To make a generator follow this pattern:
def my_new_generator(data, params):
    # Check for a parameter
    params.setdefault("mygen:param_name", 0xdefa017)
    
    # Perform operation
    newdata = np.power(data, 2)
    
    # Return like so
    return generator_return(newdata)

def identity(data, params = dict()):
    return generator_return(data)

def mean(data, params = dict()):
    return generator_return(map(np.mean, data))

def stdev(data, params = dict()):
    return generator_return(map(np.std, data))

def maximum(data, params = dict()):
    return generator_return(map(np.max, data))

def minimum(data, params = dict()):
    return generator_return(map(np.min, data))

def dwt(data, params = dict()):
    dwtdata = []
    wavelet = params.setdefault('dwt:wavelet','db1')
        
    for d in data:
        (approx, detail) = wt.dwt(d, wavelet)
        a = approx.reshape((-1))
        b = detail.reshape((-1))
        dwtdata = np.concatenate((dwtdata,a,b))
    return generator_return(dwtdata)

def fft(data, params = dict()):
    def next_fft_size(n):
        return int(np.power(2, np.ceil(np.log2(n))))
    
    length = data.shape[1]
    fftlen = next_fft_size(length)
    fftdata = [np.fft.fft(d, fftlen) for d in data]
    return generator_return(fftdata)
