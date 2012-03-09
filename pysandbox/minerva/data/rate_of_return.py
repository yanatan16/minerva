'''
Created on Mar 9, 2012

@author: jon
'''
import numpy as np

def rateOfReturn(data):
    return np.diff(np.float64(data)) / data[0:-1]
