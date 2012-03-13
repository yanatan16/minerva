'''
Created on Jan 30, 2012

@author: Jon Eisen
'''
import unittest
import numpy as np
from minerva.experiment import RegressionExperiment
from minerva.features import generators as gens
from minerva.regression import *
from minerva.utility import RandomWalk

class RegressionExperimentTest(unittest.TestCase):
    
    def createTimeSeriesData(self, n1, n2):
        return np.array([[x for x in RandomWalk(n=n2, walk_mean=np.random.randn()/10,
                                                walk_stdev=np.random.randn()/10 )] 
                         for unused in range(n1)])
        
    def testNoExperimentDefaultVars(self, repeats=0):
        '''Test a regression experiment with default vars.'''
        exp = RegressionExperiment()
        data_set = self.createTimeSeriesData(100,200)
        val = exp.run(data_set, graph=False, disp=False, repeats=repeats)
        assert val * 0 == 0, 'Run did not return a number.'

    def testNoExperimentCustomVars(self, repeats=0):
        '''Test a regression experiment with custom vars.'''
        exp = RegressionExperiment()
        data_set = self.createTimeSeriesData(100,200)
        custom_vars = dict(
            {'output_fncs': [np.min],
             'data_mapping': None,
             'seg:predictor_length': 10,
             'seg:predictee_length': 1,
             'seg:allowable_overlap': 1,
             'seg:validation_split': .3,
             'fg:generators': [gens.identity,gens.maximum,gens.minimum],
             'fg:threshold': .05,
             'reg:constructor': SupportVectorRegressor,
             'reg:training_params': dict({'kernel_type': 1, 'degree': 2})
            })
        val = exp.run(data_set, graph=False, disp=False, static_variables=custom_vars, repeats=repeats)
        assert type(val) == np.float64, 'Run did not return a number.'
        
    def testNoExperimentWithRepeats(self):
        '''Run a two-dimensional experiment with repeats'''
        self.testNoExperimentCustomVars(2)

    def testOneDimExperiment(self, repeats=0):
        '''Test a one-dimensional regression experiment'''
        exp = RegressionExperiment()
        data_set = self.createTimeSeriesData(100,200)
        custom_vars = dict(
            {
             'seg:validation_split': .3,
             'fg:generators': [gens.identity,gens.maximum,gens.minimum],
            })
        indep_vars = dict({ 'seg:allowable_overlap': range(7,9) })
        val = exp.run(data_set, graph=False, disp=False,
                      variables_under_test=indep_vars,
                      static_variables=custom_vars,
                      repeats=repeats)
        assert type(val) == np.ndarray, 'Run did not return a np.ndarray.'
        assert len(val) == len(indep_vars.values()[0]), 'Run did not return a correct length array.'
        assert type(val[0]) == np.float64, 'Run did not return a list of floats.'
        
    def testOneDimExperimentWithRepeats(self):
        '''Run a 1-dimensional experiment with repeats'''
        self.testOneDimExperiment(2)
        
    def testTwoDimExperiment(self, repeats=0):
        '''Test a two-dimensional regression experiment'''
        exp = RegressionExperiment()
        data_set = self.createTimeSeriesData(100,200)
        custom_vars = dict(
            {
             'seg:validation_split': .3,
             'fg:generators': [gens.identity,gens.maximum,gens.minimum],
            })
        indep_vars = dict({ 'seg:allowable_overlap': range(1,3),
                            'seg:predictor_length': [15,20] })
        val = exp.run(data_set, graph=False, disp=False,
                      variables_under_test=indep_vars,
                      static_variables=custom_vars,
                      repeats=repeats)
        assert type(val) == np.ndarray, 'Run did not return a np.ndarray.'
        assert np.shape(val) == (len(indep_vars.values()[0]), len(indep_vars.values()[1])) \
                or np.shape(np.transpose(val)) == (len(indep_vars.values()[0]), len(indep_vars.values()[1])), \
                    'Run did not return a correct size array.'
        assert type(val[0][0]) == np.float64, 'Run did not return an array of floats.'
        
    def testTwoDimExperimentWithRepeats(self):
        '''Run a two-dimensional experiment with repeats'''
        self.testTwoDimExperiment(1)
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()