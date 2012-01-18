'''
Created on Jan 16, 2012

@author: Jon Eisen
'''
import unittest
import numpy as np
from minerva.portfolio import *
from minerva.utility import close_enough


class MeanVariancePortfolioOptimizerTestCase(unittest.TestCase):
    
    ninvest = 20

    def testOptimizeExecution(self):
        '''Test MVPO execution'''
        corr_coeffs = 2 * np.random.rand(self.ninvest, self.ninvest) - 1
        mvpo = MeanVariancePortfolioOptimizer(corr_coeffs)
         
        current_alloc = np.arange(self.ninvest,dtype=np.float64) / np.arange(self.ninvest).sum()
        predicted_stats = np.random.randn(self.ninvest,2)
        predicted_stats[:,0] = predicted_stats[:,0] / 3
        predicted_stats[:,1] = abs(predicted_stats[:,1]) * 3
        new_alloc, unused_var, unused_ret = mvpo.optimize(current_alloc, predicted_stats)
        
        assert close_enough(sum(new_alloc),1), "New allocations are not equal to 1"
        
    def testOptimizeInEasyLargeCase(self):
        '''Test MVPO in an easy larger case'''
        cc = np.eye(self.ninvest)
        mvpo = MeanVariancePortfolioOptimizer(cc)
        
        current_alloc = np.zeros(self.ninvest)
        current_alloc[0] = 1
        predicted_stats = np.random.randn(self.ninvest,2)
        predicted_stats[:,0] = predicted_stats[:,0] / 3
        predicted_stats[:,1] = abs(predicted_stats[:,1]) * 3
        predicted_stats[0,0] = -.5
        predicted_stats[0,1] = 20
        
        unused_new_alloc, var, ret = mvpo.optimize(current_alloc, predicted_stats)
        
        assert var < 20, 'Variance did not go down in toy case'
        assert ret > -.5, 'Return did not go up in toy case'

    def testOptimizeWithSmallRiskTolerance(self):
        '''Test MVPO with small risk tolerance'''
        cc = np.eye(2)
        mvpo = MeanVariancePortfolioOptimizer(cc, risk_tolerance=0)
        
        current_alloc = np.array([1,0])
        predicted_stats = np.array([[.5,20],[.01,0]])
        
        new_alloc, var, ret = mvpo.optimize(current_alloc, predicted_stats)
        
        assert close_enough(new_alloc,[0,1]), 'Allocations in no risk tolerance case did not take lower risk'
        assert close_enough(var,0), 'Variance was not calculated correctly'
        assert close_enough(ret,0.01), 'Returns were not calculated correctly'
        
    def testOptimizeWithEqualChoices(self):
        '''Test MVPO with equal choices'''
        cc = np.eye(2)
        mvpo = MeanVariancePortfolioOptimizer(cc)
        
        current_alloc = np.array([1,0])
        predicted_stats = np.array([[.1,1],[.1,1]])
        
        new_alloc, var, ret = mvpo.optimize(current_alloc, predicted_stats)
        
        assert close_enough(new_alloc,[.5,.5]), 'Allocations in equal investment case did not diversify properly'
        assert close_enough(var,2), 'Variance was not calculated correctly'
        assert close_enough(ret,.1), 'Returns were not calculated correctly'
        
    def testOptimizeWithFullRiskTolerance(self):
        '''Test MVPO with large risk tolerance'''
        cc = np.eye(2)
        mvpo = MeanVariancePortfolioOptimizer(cc, risk_tolerance=1000)
        
        current_alloc = np.array([.5,.5])
        predicted_stats = np.array([[.1,1],[.5,10]])
        
        new_alloc, var, ret = mvpo.optimize(current_alloc, predicted_stats)
        
        assert close_enough(new_alloc,[0,1]), 'Allocations in full risk tolerance case did not choose the higher return: new_alloc=' + str(new_alloc)
        assert close_enough(var,1000), 'Variance was not calculated correctly'
        assert close_enough(ret,.5), 'Returns were not calculated correctly'

    def testOptimizeWithShorting(self):
        '''Test MVPO with shorting allowed'''
        cc = np.eye(self.ninvest)
        mvpo = MeanVariancePortfolioOptimizer(cc, allow_shorting=True, risk_tolerance=1)
        
        current_alloc = np.random.rand(self.ninvest)
        current_alloc /= sum(current_alloc)
        predicted_stats = np.random.rand(self.ninvest,2)
        predicted_stats[:,0] = predicted_stats[:,0] / 2 - .25
        predicted_stats[:,1] = predicted_stats[:,1] * 10
        predicted_stats[0,0] = .01
        predicted_stats[0,1] = 0
        
        new_alloc, unused_var, unused_ret = mvpo.optimize(current_alloc, predicted_stats)
        
        assert close_enough(sum(new_alloc),1), 'New allocations with shorting are not equal to 1'
        assert (np.array(new_alloc) < 0).any(), 'At least one of these allocations should be negative'

if __name__ == "__main__":
    import sys;sys.argv = ['', 'PyMinerva.MeanVariancePortfolioOptimizer']
    unittest.main()