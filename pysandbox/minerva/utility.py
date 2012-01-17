'''
Created on Jan 8, 2012

@author: jon
'''

import numpy as np

def close_enough(a, b, epsilon=1.0e-5):
    diff = np.array(a) - np.array(b)
    return (diff < epsilon).all()

def get_parameter(params, name, default):
    if params.has_key(name):
        return params[name]
    else:
        return default