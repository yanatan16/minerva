'''
Created on Jan 8, 2012

@author: jon
'''
from regressor import Regressor
import numpy as np

class LinearRegressor(Regressor):
    '''
    classdocs
    
    This is a n to m linear regressor
    '''
    application_matrix = []
            
    def train(self, input_vectors, expected_output_vectors):
        '''
        Train the linear regressor by solving the system created by the
        input matrix and expected outputs
        
        input_vectors: 2-d array of reals
        expected_output_vectors: 2-d array of reals
        '''
        self.application_matrix, unused_residues, unused_rank, unused_shape = \
            np.linalg.lstsq(input_vectors, expected_output_vectors, 1e-10) 
    
    def regress(self, input_vectors):
        '''
        Regress uses the matrix solved for in train() to predit outputs
        
        input_vectors: 2-d array of reals
        returns: 2-d array of reals
        '''
        return np.array(np.matrix(input_vectors) * self.application_matrix)
    