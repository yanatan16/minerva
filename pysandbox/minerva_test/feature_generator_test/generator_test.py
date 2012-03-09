'''
Created on Dec 4, 2011

@author: Eisen
'''

import numpy as np
from minerva.features import FeatureGenerator
from minerva.features import generators as gens

import unittest

class GeneratorTestCase(unittest.TestCase):
    def setUp(self):
        data = np.random.rand(200,3,10)
        self.fg = FeatureGenerator(data)
                       
    def testDwtGenerator(self):
        '''Test the use of the dwt generator'''
        # Run as default
        self.fg.generators = [gens.identity, gens.minimum, gens.dwt]
        self.fg.generate()
        
        # Test the use of a bad wavelet
        self.fg.generator_params['dwt:wavelet'] = 'wrong'
        expected_error = False
        try:
            self.fg.generate()
        except ValueError:
            expected_error = True
        assert expected_error, 'Bad wavelet parameter did not produce error'
        
        # Test the use of a non-default wavelet
        self.fg.generator_params['dwt:wavelet'] = 'haar'
        self.fg.generate()
        
        assert len(self.fg.generated_features) > 0, 'Bad generate() output with dwt'
        assert len(self.fg.generated_features.shape) == 2, 'Generated features is not 2-d'
        assert self.fg.generated_features.shape[0] == self.fg.datas.shape[0], \
                   'Number of observations is not held constant during generate()'
        
    def testFftGenerator(self):
        '''Test the use of the fft generator'''
        # Run as default
        self.fg.generators = [gens.identity, gens.minimum, gens.fft]
        self.fg.generate()
               
        assert len(self.fg.generated_features) > 0, 'Bad generate() output with fft'
        assert len(self.fg.generated_features.shape) == 2, 'Generated features is not 2-d'
        assert self.fg.generated_features.shape[0] == self.fg.datas.shape[0], \
                   'Number of observations is not held constant during generate()'
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'PyMinerva.FeatureGeneratorTest']
    unittest.main()
