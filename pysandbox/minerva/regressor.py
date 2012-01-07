'''
Created on Jan 2, 2012

@author: jon
'''
import pickle
from regressor_impl import svr
import numpy as np

# This is a list of the regressor implementations to choose from
regressor_list = { "svr": svr }

class Regressor(object):
    '''
    classdocs
    
    The regressor is a pattern recognition tool that learns a mapping of 
    n-dimensional real data to m-dimensional real data.
    '''
    impl = [] # Base implementation of the Regressor

    def __init__(self, output_len, impl_choice, params = dict()):
        '''
        Constructor
        '''
        # Construct the implementation with params
        constructor = regressor_list[impl_choice]
        self.impl = []
        for i in range(output_len):
            self.impl.append(constructor(params))
        
    def train(self, input_vectors, expected_output_vectors):
        '''
        Train trains the regressors to model the function represented by the input
        and output vectors
        '''
        exp_out = np.array(expected_output_vectors)
        for i, r in enumerate(self.impl):
            r.train_impl(input_vectors, exp_out[:,i])
    
    def regress(self, input_vectors):
        '''
        Regress predicts input_vectors and returns predicted output_vectors as a numpy array.
        An optional second return values is probability of correctness (or sureness)
        '''
        return np.concatenate(map(lambda r: [r.regress_impl(input_vectors)], self.impl), axis=1)
    
    def evaluate(self, test_input_vectors, test_output_vectors):
        '''
        Evaluate regresses the test input vectors and compares the test outputs with the actual outputs.
        It returns a tuple of (mean error, mean squared error, and standard deviation)
        '''
        actual = self.regress(test_input_vectors)
        test = (actual - test_output_vectors) / test_output_vectors
        return (np.mean(test), np.mean(test**2), np.std(test))
        
def saveRegressor(regressor, name):
    fn = name + ".reg"
    pickle.dump(regressor, fn)
    
def loadRegressor(name):
    fn = name + ".reg"
    return pickle.load(fn)
    