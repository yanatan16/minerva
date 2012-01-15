'''
Created on Jan 7, 2012

@author: jon
'''
import unittest

from minerva.regression import *
import numpy as np 

def generateOutputData(in_data, fncs):
    return np.transpose(map(lambda fn: map(fn, in_data), fncs))

def testRegressor(regressor, inputData, outputFncs, params = dict()):
    nInVecs = inputData.shape[0]
    nInFeatures = inputData.shape[1]
    nOutputs = len(outputFncs)
    nTestVecs = int(nInVecs / 2)
    
    # Regress before training
    error = False
    try:
        regressor.regress(inputData)
    except:
        error = True
    assert error, "Regressor should return error if not trained."
        
    #Test the regressor's training
    regressor.train(inputData,
                   generateOutputData(inputData, outputFncs), params)
    
    #Test the regressor's regression ability
    test_data = np.random.rand(nTestVecs, nInFeatures)
    test_output_data = regressor.regress(test_data)
    assert test_output_data.shape == (nTestVecs, nOutputs)
    
    eval_tuple = regressor.evaluate(test_data, test_output_data)
    
    # All values should be 0
    assert eval_tuple[0] < 0.0001, "Regressor should be deterministic"
    assert eval_tuple[1] < 0.0001, "Regressor should be deterministic"
    assert eval_tuple[2] < 0.0001, "Regressor should be deterministic"
    
    # Now save and load it back
    saveRegressor(regressor, "test")
    pickledReg = loadRegressor("test")
    
    # Self evaluate data from the first regressor
    eval_tuple = pickledReg.evaluate(test_data, test_output_data)
    
    # All values should be 0
    assert eval_tuple[0] < 0.0001, "Regressor load failed"
    assert eval_tuple[1] < 0.0001, "Regressor load failed"
    assert eval_tuple[2] < 0.0001, "Regressor load failed"    

class RegressorTestCase(unittest.TestCase):
    input_vecs = 10
    input_length = 5
    fncs = [np.linalg.norm, np.sum, np.mean]
    
    def testSupportVectorRegressor(self):
        '''Test the Support Vector Regressor'''
        params = dict({"quiet": ''})
        regressor = SupportVectorRegressor(len(self.fncs))
        input_data = np.random.rand(self.input_vecs, self.input_length)
        
        assert len(regressor.regressors) == len(self.fncs)
        
        testRegressor(regressor, input_data, self.fncs, params)
        
    def testLinearRegressor(self):
        '''Test the Linear Regressor'''
        regressor = LinearRegressor()
        input_data = np.random.rand(self.input_vecs, self.input_length)
        
        testRegressor(regressor, input_data, self.fncs)
        
    def testFeedforwardNeuralNetworkRegressor(self):
        '''Test the Feedforward Neural Network Regressor'''
        regressor = FeedforwardNeuralNetworkRegressor()
        input_data = np.random.rand(self.input_vecs, self.input_length)
        
        testRegressor(regressor, input_data, self.fncs)
        
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()