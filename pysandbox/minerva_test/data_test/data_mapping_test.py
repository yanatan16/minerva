'''
Created on Mar 9, 2012

@author: jon
'''
import unittest
import numpy as np
from minerva.data.data_mapping import rateOfReturn, normalizeVolume, normalizeData, makeDataNormalizer
from minerva.utility import close_enough
from itertools import izip

class DataMappingTest(unittest.TestCase):

    def testRateOfReturnNoDivisor(self):
        '''Test the rate of return function without a divisor'''
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

    def testRateOfReturnWithDivisor(self):
        '''Test the rate of return function with a specified divisor'''
        data = [1,2,2.5,2]
        div = [2,2,2]
        bad_div = [2,2,2,2]
        ror = rateOfReturn(data, divisor=div)
        assert np.shape(ror) == (3,), 'Rate of return has incorrect size'
        assert (np.array(ror) == np.array([.5,.25,-.25])).all(), 'Rate of return is incorrect'
        
        try:
            ror = rateOfReturn(data,divisor=bad_div)
            assert False, 'Rate of return did not assert error for incorrect sized divisor'
        except AssertionError:
            pass
        
        data = np.random.rand(100)
        div = np.random.rand(99) + 100
        ror = rateOfReturn(data, div)
        assert np.shape(ror) == (99,), 'Rate of return has incorrect size'
        assert close_enough(ror, np.ones((99,))), 'Rate of return is wrong!'
                
    def testNormalizeVolume(self):
        '''Test the normalize volume function'''
        data = range(1,10)
        norm_data = normalizeVolume(data)
        assert len(norm_data) == 9, 'NormalizeVolume changed length of vector'
        assert close_enough(norm_data, np.arange(.2,2,.2)), 'NormalizeVolume gave incorrect output' 
    
    def testNormalizeDataBasic(self):
        '''Test the normalize data function with basic inputs'''
        data = np.random.rand(3,100)
        ndata = normalizeData(data)
        assert np.shape(ndata) == (3,99), 'Normalize Data gave incorrect shape'
        for nd, d in izip(ndata, data):
            assert close_enough(nd, rateOfReturn(d)), \
                'Normalize Data did not use Rate of Return function correctly'
    
    def testNormalizeDataWithSelectedDivisor(self):
        '''Test the normalize data function with a selected RateOfReturn divisor row'''
        data = np.random.rand(3,100)
        data[2] += 100
        ndata = normalizeData(data, ror_divisor_row=2)
        assert np.shape(ndata) == (3,99), 'Normalize Data gave incorrect shape'
        for nd, d in izip(ndata, data):
            assert close_enough(nd, rateOfReturn(d,data[2][0:-1])), \
        'Normalize Data did not use Rate of Return with divisor function correctly'
        
    def testNormalizeDataWithVolume(self):
        '''Test the normalize data function with a selected RateOfReturn divisor row'''
        data = np.random.rand(3,100)
        data = np.array(data.tolist() + np.random.randint(100,10000,(1,100)).tolist())
        ndata = normalizeData(data, volume_row=3)
        assert np.shape(ndata) == (4,99), 'Normalize Data gave incorrect shape'
        for nd, d in izip(ndata[0:3], data[0:3]):
            assert close_enough(nd, rateOfReturn(d)), \
                'Normalize Data did not use Rate of Return correctly'
        assert close_enough(ndata[3], normalizeVolume(data[3][1:])), \
                'Normalize Data did not use Normalize Volume correctly'
        
        
    def testNormalizeDataWithSelectedDivisorAndVolume(self):
        '''Test the normalize data function with a selected RateOfReturn divisor row and Volume'''
        data = np.random.rand(3,100)
        data = np.array(data.tolist() + np.random.randint(100,10000,(1,100)).tolist())
        ndata = normalizeData(data, volume_row=3, ror_divisor_row=2)
        assert np.shape(ndata) == (4,99), 'Normalize Data gave incorrect shape'
        for nd, d in izip(ndata[0:3], data[0:3]):
            assert close_enough(nd, rateOfReturn(d, data[2][0:-1])), \
                'Normalize Data did not use Rate of Return correctly'
        assert close_enough(ndata[3], normalizeVolume(data[3][1:])), \
                'Normalize Data did not use Normalize Volume correctly'
    
    def testMakeNormalizeDataGenerator(self):
        '''Test the normalize data generator function'''
        data = np.random.rand(3,100)
        simple_norm = makeDataNormalizer()
        ndata = simple_norm(data)
        assert np.shape(ndata) == (3,99), 'Normalize Data gave incorrect shape'
        for nd, d in izip(ndata, data):
            assert close_enough(nd, rateOfReturn(d)), \
                'Normalize Data did not use Rate of Return function correctly'
        
        data = np.array(data.tolist() + np.random.randint(100,10000,(1,100)).tolist())
        complex_norm = makeDataNormalizer(1, 3)
        ndata = complex_norm(data)
        assert np.shape(ndata) == (4,99), 'Normalize Data gave incorrect shape'
        for nd, d in izip(ndata[0:3], data[0:3]):
            assert close_enough(nd, rateOfReturn(d, data[1][0:-1])), \
                'Normalize Data did not use Rate of Return correctly'
        assert close_enough(ndata[3], normalizeVolume(data[3][1:])), \
                'Normalize Data did not use Normalize Volume correctly'
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()