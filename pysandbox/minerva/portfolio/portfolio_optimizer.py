'''
Created on Jan 8, 2012

@author: Jon Eisen
'''

class OptimizationException(Exception):
    def __init__(self, reason):
        self.str = reason
    def __str__(self):
        return self.str
        

class PortfolioOptimizer(object):
    '''
    classdocs
    
    An interface for any implementation of a financial portfolio optimizer
    '''

    def optimize(self, current_allocations, possible_investments):
        '''
        Optimize a portfolio's allocations given the current allocations and information
        on the possible investments. Returns the optimized allocations.
        
        Current allocations is an array of weights that sum to 1. The weights represent
        the amount invested in each possible investments. 
        
        Possible investments is information on each possible investment. Its length should
        be the same as current allocations
        
        This function returns an array of new allocation weights that sum to 1.
        
        There is an optional return of the mean and variance of the newly allocated weights
        '''
        assert len(current_allocations) == len(possible_investments)