'''
Created on Dec 4, 2011

@author: Eisen
'''

import numpy as np
from minerva.features import FeatureGenerator
from minerva.features import generators as gens

import unittest

class MultiDimFeatureGeneratorTestCase(unittest.TestCase):
    def testGenerate(self):
        '''Test the generation of features with multidim generators'''
        data = np.random.rand(200,3,10)
        gs = [[gens.minimum], [gens.maximum], [gens.mean]]
        fg = FeatureGenerator(data, gs)
        fg.generated_features = [] # Reset the generator
        fg.generate()
        assert len(fg.generated_features) > 0, 'Bad generate() output'
        assert len(fg.generated_features.shape) == 2, 'Generated features is not 2-d'
        assert fg.generated_features.shape[0] == fg.datas.shape[0], \
                   'Number of observations is not held constant during generate()' 
        assert fg.generated_features.shape[1] == 3, \
                   'Number of generated features is not appropriate'
                   
    def testGenerateDiff2ndDim(self):
        '''Test the generation of features with multidim generators with oddly shaped generator dimensions'''
        data = np.random.rand(200,3,10)
        gs = [[gens.minimum], [gens.maximum, gens.mean], [gens.mean, gens.identity]]
        fg = FeatureGenerator(data, gs)
        fg.generated_features = [] # Reset the generator
        fg.generate()
        assert len(fg.generated_features) > 0, 'Bad generate() output'
        assert len(fg.generated_features.shape) == 2, 'Generated features is not 2-d'
        assert fg.generated_features.shape[0] == fg.datas.shape[0], \
                   'Number of observations is not held constant during generate()' 
        assert fg.generated_features.shape[1] == 14, \
                   'Number of generated features is not appropriate'
        
    def testGenerateMultidimAccuracyDiff2ndDim(self):
        '''Test the accuracy of generation of features with multidim generators with oddly shaped generator dimensions'''
        data = np.array([[[5,4,5],[1,2,3],[0,-1,-2]]])
        gs = [[gens.minimum], [gens.maximum, gens.mean, gens.minimum], [gens.mean]]
        fg = FeatureGenerator(data, gs)
        fg.generate()
        expected_genfeat = np.array([[4,3,2,1,-1]])
        assert np.shape(fg.generated_features) == np.shape(expected_genfeat), 'Generate features produced an incorrect number of features'
        assert (fg.generated_features == expected_genfeat).all(), 'Generated features are not correct'
        
    def testGenerateMultidimAccuracy(self):
        '''Test the accuracy of generation of features with multidim generators'''
        data = np.array([[[1,2,3],[1,2,3],[1,2,3]]])
        gs = [[gens.minimum], [gens.maximum], [gens.mean]]
        fg = FeatureGenerator(data, gs)
        fg.generate()
        expected_genfeat = np.array([[1,3,2]])
        assert np.shape(fg.generated_features) == np.shape(expected_genfeat), 'Generate features produced an incorrect number of features'
        assert (fg.generated_features == expected_genfeat).all(), 'Generated features are not correct'
        
    def testMultiDimGeneratorsWithNones(self):
        '''Test the generation of features with multidim generators with none generators'''
        data = np.random.rand(200,3,10)
        gs = [[], [gens.maximum], []]
        fg = FeatureGenerator(data, gs)
        fg.generate()
        
        assert len(fg.generated_features) > 0, 'Bad generate() output'
        assert len(fg.generated_features.shape) == 2, 'Generated features is not 2-d'
        assert fg.generated_features.shape[0] == fg.datas.shape[0], \
                   'Number of observations is not held constant during generate()' 
        assert fg.generated_features.shape[1] == 1, \
                   'Number of generated features is not appropriate'
    
    def testMultiDimGeneratorsWithNonesAccuracy(self):
        '''Test the accuracy of generation of features with multidim generators with none generators'''
        
        data = np.array([[[1,2,3],[1,2,3],[1,2,3]]])
        gs = [[], [gens.maximum], []]
        fg = FeatureGenerator(data, gs)
        fg.generate()
        expected_genfeat = np.array([[3]])
        assert np.shape(fg.generated_features) == np.shape(expected_genfeat), 'Generate features produced an incorrect number of features'
        assert (fg.generated_features == expected_genfeat).all(), 'Generated features are not correct'
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'PyMinerva.FeatureGeneratorTest']
    unittest.main()
