'''
Created on Mar 10, 2012

@author: jon
'''
import unittest
import numpy as np
from minerva.utility import RandomWalk
from minerva.experiment import RegressionExperiment
from types import IntType
from minerva.data import DataNormalizer

class RegressionExperimentMultidimTest(unittest.TestCase):
    def createTimeSeriesData(self, n1, n2, n3=None):
        if n3 is None and type(n2) == IntType:
            return np.array([[x for x in RandomWalk(n=n2, walk_mean=mean/10,
                                                    walk_stdev=stdev/10)] 
                             for mean, stdev in zip(np.random.randn(n1),np.random.randn(n1))])
        if n3 is None:
            assert len(n2) == n1, 'n2 must be of length n1 if its a tuple'
            return np.array([[x for x in RandomWalk(n=y, walk_mean=mean/10,
                                                    walk_stdev=stdev/10)] 
                             for y, mean, stdev in zip(n2, np.random.randn(n1),np.random.randn(n1))])
        elif type(n3) == IntType:
            assert type(n2) == IntType, 'n2 Must be an int if n3 is defined'
            assert n2 > 1, 'Must have more than 1 on second dimension'
            return np.array([[[x for x in RandomWalk(n=n3, walk_mean=mean/10, walk_stdev=stdev/10)] 
                              for unused_2 in range(n2-1)] + [np.random.randint(100,10000,(n3,))]
                             for mean, stdev in zip(np.random.randn(n1),np.random.randn(n1))])
        else:
            assert type(n2) == IntType, 'n2 Must be an int if n3 is defined'
            assert n2 > 1, 'Must have more than 1 on second dimension'
            assert len(n3) == n1, 'n3 must be of length n1 if its a tuple'
            return np.array([[[x for x in RandomWalk(n=y)] 
                              for unused in range(n2-1)] + [np.random.randint(100,10000,(y,))] 
                             for y, mean, stdev in zip(n3, np.random.randn(n1),np.random.randn(n1))])
    

    def testRegressionWithMultidimData(self):
        '''Run a single experiment with multidimension data'''
        data = self.createTimeSeriesData(150, 3, 50)
        custom_vars = dict({'output_fncs': [lambda ds: np.mean(ds[1])]})
        exp = RegressionExperiment()
        val = exp.run(data, graph=False, disp=False, static_variables=custom_vars)
        assert type(val) == np.float64, 'Run did not return a number.'
        
    def testRegressionWithNonAlignedData(self):
        '''Run a single experiment with nonaligned data'''
        data = self.createTimeSeriesData(10, [np.random.randint(100,1000) for unused in range(10)])
        exp = RegressionExperiment()
        val = exp.run(data, graph=False, disp=False)
        assert type(val) == np.float64, 'Run did not return a number.'
    
    def testRegressionWithNonAlignedMultidimData(self):
        '''Run a single experiment with non-aligned multidimension data'''
        data = self.createTimeSeriesData(10, 3, [np.random.randint(100,1000) for unused in range(10)])
        custom_vars = dict({'output_fncs': [lambda ds: np.mean(ds[0])]})
        exp = RegressionExperiment()
        val = exp.run(data, graph=False, disp=False, static_variables=custom_vars)
        assert type(val) == np.float64, 'Run did not return a number.'
    
    def testRegressionWithRorSelectedMultidimData(self):
        '''Run a single experiment with multidimension data'''
        data = self.createTimeSeriesData(150, 3, 50)
        custom_vars = dict({'output_fncs': [lambda ds: np.mean(ds[1])]})
        exp = RegressionExperiment()
        val = exp.run(data, graph=False, disp=False, static_variables=custom_vars)
        assert type(val) == np.float64, 'Run did not return a number.'
    
    def testRegressionWithRorSelectedNonalignedMultidimData(self):
        '''Run a single experiment with multidimension data'''
        data = self.createTimeSeriesData(10, 3, [np.random.randint(100,1000) for unused in range(10)])
        custom_vars = dict({'output_fncs': [lambda ds: np.mean(ds[1])],
                            'data_mapping': DataNormalizer(0, 2)})
        exp = RegressionExperiment()
        val = exp.run(data, graph=False, disp=False, static_variables=custom_vars)
        assert type(val) == np.float64, 'Run did not return a number.'


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()