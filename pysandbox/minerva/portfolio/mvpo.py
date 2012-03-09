'''
Created on Jan 15, 2012

@author: Jon Eisen
'''
from portfolio_optimizer import PortfolioOptimizer, OptimizationException
import numpy as np
from scipy.optimize import fmin_slsqp

class MeanVariancePortfolioOptimizer(PortfolioOptimizer):
    '''
    classdocs
    
    Optimize a portfolio using Modern Portfolio Theory. This class does not use a
    transaction cost, and does not use shorting.
    '''

    def __init__(self, correlation_coefficients, 
                 risk_tolerance = 1, 
                 allow_shorting = False, 
                 verbose = False):
        '''
        Constructor
        
        Construct the MPT optimizer with a set of correlation coefficients. This should
        be a real symmetric matrix of values from -1 to 1, which represent the correlation
        coefficients of two financial investments (represented only as indices i and j).
        The diagnol of the matrix should be 1s.
        '''
        self.corr_coefs = np.array(correlation_coefficients)
        assert self.corr_coefs.shape[0] == self.corr_coefs.shape[1], "Correlation coefficient matrix must be square"
        assert self.corr_coefs.shape[0] > 0, "Correlation Coefficient matrix must not be empty"
        
        assert risk_tolerance >= 0, 'Risk Tolerance cannot be negative' 
        self.q = risk_tolerance
        
        self.short = allow_shorting
        self.verbose = verbose
        
    def optimize(self, current_allocations, investment_predicted_stats):
        '''
        MPT Optimizer. This function will use the correlation coefficients and the information
        passed in possible investments to generate new optimal allocations. 
        
        Current allocations is the weight vector of the current allocation of investments. 
        
        Investment predicted statistics should contain two values for each investment:
        predicted mean and variance of the rate of return of the investment. 
        
        This function returns 3 things: an array of new allocation weights that sum to 1, the mean and variance of that portfolio
        '''
        assert len(current_allocations) == len(investment_predicted_stats), "Must have an allocation for all possible investments"
        assert sum(current_allocations) <= 1.0 + 1.0e-5, "Current Allocations must sum to less than 1"
        assert len(investment_predicted_stats) == self.corr_coefs.shape[0], "Must keep # of investments the same as the correlation coefficient matrix"
        assert len(investment_predicted_stats[0]) == 2, "Expected two stats per investment: mean and variance"

        # First we must construct the covariance matrix and return vector
        returns = np.matrix(investment_predicted_stats[:,0])
        variances = np.array(investment_predicted_stats[:,1])
        covariances = np.matrix(np.array(variances.T * variances) * self.corr_coefs)
        
        Var = lambda w: (w * covariances).dot(w)
        Ret = lambda w: returns.dot(w)
        
        # Function to minimize
        F = lambda w, *args: Var(w) - self.q * Ret(w)
        Fp = lambda w, *args: np.array(2 * w * covariances - self.q * returns)[0]
        
        # Constraint = 0
        C = lambda w, *args: sum(w) - 1
        Cp = lambda w, *args: np.ones((1,len(w)))
        
        # Initial guess
        w0 = np.array(current_allocations);
        
        if (self.short):
            # Unbounded 
            bounds = []
        else:
            # Bounded
            bounds = np.zeros((len(w0),2))
            bounds[:,1] = 1
        
        verb = 1 * self.verbose
        wf, unused_fx, unused_its, imode, smode = fmin_slsqp(
                        F, w0, 
                        fprime=Fp, 
                        bounds=bounds, 
                        eqcons=[C], 
                        fprime_eqcons=Cp, 
                        full_output=True, 
                        iprint=verb);
                       
        if (imode != 0):
            raise OptimizationException('Error in optimization: ' + smode)

        #We have the new weights. Calculate the mean and variance
        return wf, Var(wf), Ret(wf)
        
        