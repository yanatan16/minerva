'''
Created on Jan 2, 2012

@author: jon
'''
import pickle
import numpy as np

class Regressor(object):
    '''
    classdocs
    
    The regressor is a pattern recognition tool that learns a mapping of 
    n-dimensional real data to m-dimensional real data.
    '''
    
    def __init__(self):
        '''
        Constructor
        '''
        pass
        
    def train(self, input_vectors, expected_output_vectors, params = dict()):
        '''
        Train trains the regressors to model the function represented by the input
        and output vectors.
        
        input_vectors: 2-d array of reals
        expected_output_vectors: 2-d array of reals
        
        This method should be overloaded by an implementing regressor.
        '''
        pass
    
    def regress(self, input_vectors):
        '''
        Regress predicts input_vectors and returns predicted output_vectors as a numpy array.
        An optional second return values is probability of correctness (or sureness)
        
        input_vectors: 2-d array of reals. An array of input vectors
        returns: 2-d array of reals
    
        This method should be overloaded by an implementing regressor.
        '''
        pass
    
    def evaluate(self, test_input_vectors, test_output_vectors):
        '''
        Evaluate regresses the test input vectors and compares the test outputs with the actual outputs.
        It returns a tuple of (mean error, mean squared error, and standard deviation)
        
        test_input_vectors: 2-d array of reals
        test_output_vectors: 2-d array of reals
        returns: tuple of mean error, mean squared error, and standard deviation of error
        '''
        actual = self.regress(test_input_vectors)
        test = (actual - test_output_vectors) / test_output_vectors
        return (np.mean(test), np.mean(test**2), np.std(test))
    
    def prepareForSave(self, name):
        '''
        Prepare the Regressor for a save operation.
        Save off any non-picklable objects by their own method
        '''
        pass
    def recoverFromLoad(self, name):
        '''
        Prepare the Regressor from a load operation.
        Recover any non-picklable objects by their own method
        '''
        pass
    
    
class SingleOutputRegressor(Regressor):
    '''
    classdocs
    
    The single-output regressor is a pattern recognition tool that learns a mapping of 
    n-dimensional real data to 1-dimensional real data.
    '''
    
    def __init__(self):
        '''
        Constructor
        '''
        pass
        
    def train(self, input_vectors, expected_output_values):
        '''
        Train trains the regressors to model the function represented by the input
        and output vectors.
        
        input_vectors: 2-d array of reals
        expected_output_values: 1-d array of reals
        
        This method should be overloaded by an implementing regressor.
        '''
        pass
    
    def regress(self, input_vectors):
        '''
        Regress predicts input_vectors and returns predicted output_values as a numpy array.
        An optional second return values is probability of correctness (or sureness)
        
        input_vectors: 2-d array of reals
        returns: 1-d array of reals
        
        This method should be overloaded by an implementing regressor.
        '''
        pass

class SingleOutputExtensionRegressor(Regressor):
    '''
    classdocs
    
    The single output extension regressor is a wrapper class for single-output
    regressors to be used for multiple output regression application
    '''
    regressors = [] # Base implementation of the Regressor

    def __init__(self, output_len, regressor_constructor):
        '''
        Constructor
        '''
        # Construct the implementation with params
        self.regressors = []
        for i in range(output_len):
            self.regressors.append(regressor_constructor())
        
    def train(self, input_vectors, expected_output_vectors, params = dict()):
        '''
        Train trains the regressors to model the function represented by the input
        and output vectors
        '''
        exp_out = np.array(expected_output_vectors)
        for i, r in enumerate(self.regressors):
            r.train(input_vectors, exp_out[:,i], params)
    
    def regress(self, input_vectors):
        '''
        Regress predicts input_vectors and returns predicted output_vectors as a numpy array.
        An optional second return values is probability of correctness (or sureness)
        '''
        return np.transpose(map(lambda r: r.regress(input_vectors), self.regressors))
    
    def prepareForSave(self, name):
        '''
        Prepare the Regressor for a save operation.
        Save off any non-picklable objects by their own method
        '''
        for i, regi in enumerate(self.regressors):
            regi.prepareForSave(name + "." + str(i))
            
    def recoverFromLoad(self, name):
        '''
        Prepare the Regressor from a load operation.
        Recover any non-picklable objects by their own method
        '''
        for i, regi in enumerate(self.regressors):
            regi.recoverFromLoad(name + "." + str(i))
        
def saveRegressor(regressor, name):
    fn = name + ".reg"
    fid = open(fn,'wb')
    regressor.prepareForSave(name)
    pickle.dump(regressor, fid)
    fid.close()
    
def loadRegressor(name):
    fn = name + ".reg"
    fid = open(fn,'rb')
    reg = pickle.load(fid)
    reg.recoverFromLoad(name)
    fid.close()
    return reg
    