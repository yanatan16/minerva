'''
Created on Jan 8, 2012

@author: jon
'''

import numpy as np

def close_enough(a, b, epsilon=1.0e-5):
    diff = np.array(a) - np.array(b)
    return (diff < epsilon).all()

class RandomWalk(object):
    def __init__(self, 
                 start=np.random.rand(), 
                 inc=lambda: np.random.randn(),  
                 walk_mean=0,
                 walk_stdev=1,
                 n = np.Infinity):
        self.current = start;
        self.length = n
        self.walk = lambda cur: cur + inc() * walk_stdev + walk_mean
        
    def __iter__(self):
        return self
    
    def next(self):
        self.length -= 1
        if self.length < 0:
            raise StopIteration
        else:
            self.current = self.walk(self.current)
            return self.current