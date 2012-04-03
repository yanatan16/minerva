'''
Created on Mar 9, 2012

@author: jon
'''
import numpy as np
from types import ListType, TupleType, IntType

def rateOfReturn(data, divisor=None):
    diffAndDivide = lambda toDiff, toDiv: np.diff(np.float64(toDiff)) / toDiv
    
    if divisor is None:
        return diffAndDivide(data, data[0:-1])
    else:
        assert len(divisor) == len(data) - 1, 'len(divisor) must be len(data) - 1'
        return diffAndDivide(data, divisor)
    
def normalizeVolume(data):
    return data / np.mean(data)

def normalizeData(data, ror_divisor_row=None, volume_row=None):
    if type(data[0]) == np.ndarray or type(data[0]) == ListType:
        if volume_row is None:
            if ror_divisor_row is None:
                return np.array([rateOfReturn(d) for d in data])
            else:
                return np.array([rateOfReturn(d, data[ror_divisor_row][0:-1]) for d in data])
        else:
            if ror_divisor_row is None:
                return np.array([rateOfReturn(d) for d in data[0:volume_row]] +
                                [normalizeVolume(data[volume_row][1:])] +
                                [rateOfReturn(d) for d in data[volume_row+1:]])
            else:
                return np.array([rateOfReturn(d, data[ror_divisor_row][0:-1]) for d in data[0:volume_row]] +
                                [normalizeVolume(data[volume_row][1:])] +
                                [rateOfReturn(d, data[ror_divisor_row][0:-1]) for d in data[volume_row+1:]])
            
    else:
        if volume_row == 0:
            return normalizeVolume(data)
        else:
            return rateOfReturn(data)
    
class DataNormalizer(object):    
    def __init__(self, ror_divisor_row=None, volume_row=None):
        '''Make a function to normalize data'''
        self.ror_row = ror_divisor_row
        self.vol_row = volume_row
    def __call__(self, data):
        return normalizeData(data, self.ror_row, self.vol_row)
    def __eq__(self, other):
        return type(self) == type(other) and self.ror_row == other.ror_row and self.vol_row == other.vol_row