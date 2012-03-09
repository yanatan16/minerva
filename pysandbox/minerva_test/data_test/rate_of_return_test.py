'''
Created on Mar 9, 2012

@author: jon
'''
import unittest
import numpy as np
from minerva.data.rate_of_return import rateOfReturn

class RateOfReturnTest(unittest.TestCase):

    def testRange(self):
        '''Use a range to test the rate of return function'''
        data = np.arange(1,10)
        ror = rateOfReturn(data)
        exp_ror = [1, 1/2., 1/3., 1/4., 1/5., 1/6., 1/7., 1/8.]
        
        assert np.shape(ror) == np.shape(exp_ror), 'Rate of return has incorrect size'
        assert (ror == exp_ror).all(), 'Rate of return calculated incorrectly'
        
    def testLargeRandomData(self):
        '''Use a large set of random data to test rate of return function'''
        data = np.random.rand(1000)
        ror = rateOfReturn(data)
        exp_ror = np.diff(data)/data[0:-1]
        
        assert np.shape(ror) == np.shape(exp_ror), 'Rate of return has incorrect size'
        assert (ror == exp_ror).all(), 'Rate of return calculated incorrectly'


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()