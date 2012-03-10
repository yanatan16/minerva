'''
Created on Mar 9, 2012

@author: jon
'''
import numpy as np
from types import ListType, TupleType, IntType

def rateOfReturn(data, selected_divisors=None):
    diffAndDivide = lambda toDiff, toDiv: np.diff(np.float64(toDiff)) / toDiv
    
    if type(np.array(data)[0]) == np.ndarray:
        if selected_divisors is None:
            return [diffAndDivide(d, d[0:-1]) for d in data]
        elif type(selected_divisors) == IntType:
            return [diffAndDivide(d, data[selected_divisors][0:-1]) for d in data]
        else:
            assert len(selected_divisors) == len(data), 'Selected divisors should be the same length as data'
            return [diffAndDivide(d, data[sd][0:-1]) for d, sd in zip(data, selected_divisors)]
    else:
        return diffAndDivide(data, data[0:-1])
    
