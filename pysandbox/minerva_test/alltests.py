'''
Created on Jan 15, 2012

@author: Jon Eisen
'''

import unittest
from regressor_test import RegressorTestCase
from feature_generator_test import FeatureGeneratorTestCase
from portfolio_test.alltests import *
from data_test.alltests import *
from experiment_test.alltests import *
        
if __name__ == "__main__":
    import sys;sys.argv = ['', 'PyMinerva.AllTests']
    unittest.main()