'''
Created on Jan 7, 2012

@author: jon
'''
import unittest

from minerva.regression.svr import svr, svr_single
from svmutil import svm_parameter
import numpy as np
from minerva.regression.regressor import saveRegressor, loadRegressor 

def generateOutputData(in_data, fncs):
    return np.transpose(map(lambda fn: map(fn, in_data), fncs))

def testRegressor(regressor, inputData, outputFncs):
    nInVecs = inputData.shape[0]
    nInFeatures = inputData.shape[1]
    nOutputs = len(outputFncs)
    nTestVecs = int(nInVecs / 2)
        
    '''Test the regressor's training'''
    regressor.train(inputData,
                   generateOutputData(inputData, outputFncs))
    
    '''Test the regressor's regression ability'''
    test_data = np.random.rand(nTestVecs, nInFeatures)
    test_output_data = regressor.regress(test_data)
    assert test_output_data.shape == (nTestVecs, nOutputs)
    
    # Put back into evaluate. This should give us a 0 error
    eval_tuple = regressor.evaluate(test_data, test_output_data)
    assert eval_tuple[0] < 0.0001
    assert eval_tuple[1] < 0.0001
    assert eval_tuple[2] < 0.0001
    
    '''Test the regressor's evaluation ability'''
    test_data = np.random.rand(nTestVecs, nInFeatures)
    test_output_data = generateOutputData(test_data, outputFncs)
    
    eval_tuple = regressor.evaluate(test_data, test_output_data)
    assert len(eval_tuple) == 3
    for val in eval_tuple:
        assert val * 0 == 0 # Make sure its a number
        

class RegressorTestCase(unittest.TestCase):
    input_vecs = 10
    input_length = 5
    fncs = [np.linalg.norm, np.sum, np.mean]
    
    def testSVR(self):
        '''Test the Support Vector Regressor'''
        params = dict({"quiet": ''})
        regressor = svr(len(self.fncs), params)
        input_data = np.random.rand(self.input_vecs, self.input_length)
        
        assert len(regressor.regressors) == len(self.fncs)
        for regi in regressor.regressors:
            assert type(regi) == svr_single
            assert type(regi.params) == type("")
        
        testRegressor(regressor, input_data, self.fncs)
        
    def testSavingLoading(self):
        ''' Test saving and loading a regressor '''
        params = dict({"quiet": ''})
        regressor = svr(len(self.fncs), params)
        input_data = np.random.rand(self.input_vecs, self.input_length)
        regressor.train(input_data,
                   generateOutputData(input_data, self.fncs))
        nTestVecs = 50
        test_data = np.random.rand(nTestVecs, self.input_vecs)
        test_output_data = regressor.regress(test_data)
        
        # Now save and load it back
        saveRegressor(regressor, "test")
        pickledReg = loadRegressor("test")
        
        # Self evaluate data from the first regressor
        eval_tuple = pickledReg.evaluate(test_data, test_output_data)
        
        # All values should be 0
        assert eval_tuple[0] < 0.0001
        assert eval_tuple[1] < 0.0001
        assert eval_tuple[2] < 0.0001
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()