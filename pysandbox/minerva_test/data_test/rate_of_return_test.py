'''
Created on Mar 9, 2012

@author: jon
'''
import unittest
import numpy as np
from minerva.data.rate_of_return import rateOfReturn

class RateOfReturnTest(unittest.TestCase):

    def testSingleDim(self):
        '''Test the rate of return function with single dimension data'''
        data = np.arange(1,10)
        ror = rateOfReturn(data)
        exp_ror = [1, 1/2., 1/3., 1/4., 1/5., 1/6., 1/7., 1/8.]
        
        assert np.shape(ror) == np.shape(exp_ror), 'Rate of return has incorrect size'
        assert (ror == exp_ror).all(), 'Rate of return calculated incorrectly'
        
        data = np.random.rand(1000)
        ror = rateOfReturn(data)
        exp_ror = np.diff(data)/data[0:-1]
        
        assert np.shape(ror) == np.shape(exp_ror), 'Rate of return has incorrect size'
        assert (ror == exp_ror).all(), 'Rate of return calculated incorrectly'

    def testMultidim(self):
        '''Test the rate of return function with multidimension data'''
        data = np.random.rand(5,100)
        ror = rateOfReturn(data)
        assert np.shape(ror) == (5,99), 'Rate of return has incorrect size'
        
        data = [[1,2,3],[2,3,4],[3,4,5]]
        ror = rateOfReturn(data)
        assert np.shape(ror) == (3,2), 'Rate of return has incorrect size'
        assert (np.array(ror) == np.array([[1,1/2.],[1/2.,1/3.],[1/3.,1/4.]])).all(), \
                'Rate of return is wrong!'
                
    def testMultidimWithSelectedDivisors(self):
        '''Test the rate of return function with multidimension data and selected divisors'''
        data = np.random.rand(5,100)
        ror = rateOfReturn(data, selected_divisors=1)
        assert np.shape(ror) == (5,99), 'Rate of return has incorrect size'
        
        ror = rateOfReturn(data, selected_divisors=(1,1,1,1,2))
        assert np.shape(ror) == (5,99), 'Rate of return has incorrect size'
        
        try:
            ror = rateOfReturn(data,selected_divisors=(1,2,3))
            assert False, 'Rate of return function did not throw error for incorrect selecte divisors'
        except AssertionError:
            pass

        data = [[1,2,3],[2,3,4],[3,4,5]]
        ror = rateOfReturn(data, selected_divisors=0)
        assert np.shape(ror) == (3,2), 'Rate of return has incorrect size'
        assert (np.array(ror) == np.array([[1,1/2.],[1,1/2.],[1.,1/2.]])).all(), \
                'Rate of return with selected divisor 1 is wrong!'
        
        ror = rateOfReturn(data, selected_divisors=(0,1,1))
        assert np.shape(ror) == (3,2), 'Rate of return has incorrect size'
        assert (np.array(ror) == np.array([[1,1/2.],[1/2.,1/3.],[1/2.,1/3.]])).all(), \
                'Rate of return with selected divisors (1,2,2) is wrong!'
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()