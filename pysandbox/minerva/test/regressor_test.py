'''
Created on Jan 7, 2012

@author: jon
'''
import unittest

from minerva.regressor import Regressor
from minerva.regressor_impl import svr
from svmutil import svm_parameter
import numpy as np

def generateOutputData(in_data, fncs):
    return np.transpose(map(lambda fn: map(fn, in_data), fncs))

class RegressorSVRTest(unittest.TestCase):
    reg = []
    input_vecs = 10
    input_length = 5
    fncs = [np.linalg.norm, np.sum, np.mean]
    reg_impl_choice = "svr"
    input_data = np.random.rand(input_vecs, input_length)
    params = dict({"quiet": ''})
    
    def setUp(self):
        self.reg = Regressor(len(self.fncs), self.reg_impl_choice, self.params)
    
    def testConstruction(self):
        '''Test the construction of the regressor'''
        assert len(self.reg.impl) == len(self.fncs)
        for regi in self.reg.impl:
            assert type(regi) == svr
            assert type(regi.params) == svm_parameter

    def testTrain(self):
        '''Test the regressor's training'''
        self.reg.train(self.input_data,
                       generateOutputData(self.input_data, self.fncs))
        for regi in self.reg.impl:
            assert regi.model != []
        
    def testRegress(self):
        '''Test the regressor's regression ability'''
        self.reg.train(self.input_data,
                       generateOutputData(self.input_data, self.fncs))
        test_vecs = 50
        test_data = np.random.rand(test_vecs, self.input_length)
        output = self.reg.regress(test_data)
        assert output.shape == (test_vecs, self.input_length)
    
    def testEvaluate(self):
        '''Test the regressor's evaluation ability'''
        self.reg.train(self.input_data,
                       generateOutputData(self.input_data, self.fncs))
        test_vecs = 50
        test_data = np.random.rand(test_vecs, self.input_length)
        test_output_data = generateOutputData(test_data, self.fncs)
        
        eval_tuple = self.reg.evaluate(test_data, test_output_data)
        assert len(eval_tuple) == 3
        for val in eval_tuple:
            assert type(val) == float

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()