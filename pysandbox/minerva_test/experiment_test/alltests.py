'''
Created on Jan 15, 2012

@author: Jon Eisen
'''

import unittest
from regression_experiment_test import RegressionExperimentTest
from base_experiment_test import BaseExperimentTest
from regression_expmt_multidim_test import RegressionExperimentMultidimTest
        
if __name__ == "__main__":
    import sys;sys.argv = ['', 'PyMinerva.DataTests']
    unittest.main()