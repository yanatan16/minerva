'''
Created on Jan 30, 2012

@author: Jon Eisen
'''
import unittest
import numpy as np
from minerva.experiment import RegressionExperiment
from minerva.features import generators as gens
from minerva.regression import SupportVectorRegressor

def createTimeSeriesData(nObs, nSamp):
    starting_vals = np.random.rand(nObs)
    walk_stdevs = np.random.rand(nObs) * .4
    walk_means = np.random.randn(nObs) * .1 + .05  
    random_data = np.random.randn(nObs, nSamp)
    indices = range(nObs)
    random_walk = np.array([random_data[i] * walk_stdevs[i] + walk_means[i] for i in indices])
    time_series_data = np.cumsum(random_walk, axis=1)
    return np.array([starting_vals[i] + time_series_data[i] for i in indices])
    
class RegressionExperimentTest(unittest.TestCase):

    def testNoExperimentDefaultVars(self):
        exp = RegressionExperiment()
        data_set = createTimeSeriesData(100,200)
        val = exp.run(data_set, graph=False)
        assert val * 0 == 0, 'Run did not return a number.'

    def testNoExperimentCustomVars(self):
        exp = RegressionExperiment()
        data_set = createTimeSeriesData(100,200)
        custom_vars = dict(
            {'output_fncs': [np.min],
             'seg:predictor_length': 10,
             'seg:predictee_length': 1,
             'seg:allowable_overlap': 1,
             'seg:validation_split': .3,
             'fg:generators': [gens.identity,gens.maximum,gens.minimum],
             'fg:threshold': .05,
             'reg:constructor': SupportVectorRegressor,
             'reg:training_params': dict({'kernel_type': 1, 'degree': 2})
            })
        val = exp.run(data_set, graph=False, static_variables=custom_vars)
        assert val * 0 == 0, 'Run did not return a number.'

#TODO add more

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()