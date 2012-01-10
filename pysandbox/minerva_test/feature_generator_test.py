'''
Created on Dec 4, 2011

@author: Eisen
'''

import numpy as np
import minerva.feature_generator as fgen
import minerva.generators as gens

import unittest

class FeatureGeneratorTestCase(unittest.TestCase):
    def setUp(self):
        data = np.random.rand(200,3,10)
        self.fg = fgen.FeatureGenerator(data)
        
    def testGenerate(self):
        '''Test the generation of features using the basic generators'''
        gs = [gens.identity, gens.minimum, gens.maximum, gens.mean, gens.stdev]
        self.fg.generators = gs;
        self.fg.generated_features = [] # Reset the generator
        self.fg.generate()
        assert len(self.fg.generated_features) > 0, 'Bad generate() output'
        assert len(self.fg.generated_features.shape) == 2, 'Generated features is not 2-d'
        assert self.fg.generated_features.shape[0] == self.fg.datas.shape[0], \
                   'Number of observations is not held constant during generate()' 
        
        self.fg.generators = [gens.identity, gens.maximum, gens.minimum, gens.mean]
        self.fg.datas = np.array([[[1,2,3]]])
        self.fg.generate()
        expected_genfeat = np.array([[1,2,3,3,1,2]])
        assert (self.fg.generated_features == expected_genfeat).all(), 'Generated features are not aligned correctly'
                    
    def testGenerateScaling(self):
        self.fg.generators = [gens.identity]
        self.fg.generator_params['scaling'] = False
        self.fg.datas = np.array([[np.arange(10),np.arange(10)-5]],dtype='float')
        self.fg.generate()
        
        assert len(self.fg.scaling[0]) > 0, 'Scaling not generated'
        assert len(self.fg.scaling[0]) == len(self.fg.scaling[1]), 'Scaling parameters sized inappropriately'
        assert len(self.fg.scaling[0]) == self.fg.datas.shape[1], 'Scaling parameters sized incorrectly'
        assert self.fg.scaling[0].max() == 1 and self.fg.scaling[0].min() == 1, \
                    'Scaling "off" did not produce correct multipliers'
        assert self.fg.scaling[1].max() == 0 and self.fg.scaling[1].min() == 0, \
                    'Scaling "off" did not produce correct summands'
        assert self.fg.generated_features.max() > 1, 'Scaling applied when off'
        
        self.fg.generator_params['scaling'] = True
        self.fg.generate()
        assert len(self.fg.scaling[0]) > 0, 'Scaling not generated with scaling on'
        assert len(self.fg.scaling[0]) == len(self.fg.scaling[1]), 'Scaling parameters sized inappropriately with scaling on'
        assert len(self.fg.scaling[0]) == self.fg.datas.shape[1], 'Scaling parameters sized incorrectly with scaling on'
        assert not (self.fg.scaling[0].max() == 1 and self.fg.scaling[0].min() == 1), \
                    'Scaling "on" did not produce correct multipliers'
        assert not (self.fg.scaling[1].max() == 0 and self.fg.scaling[1].min() == 0), \
                    'Scaling "on" did not produce correct summands'
        assert self.fg.generated_features.max() == 1, 'Scaling (maximum) not applied correctly'
        assert self.fg.generated_features.min() == -1, 'Scaling (minimum) not applied correctly'
        assert self.fg.generated_features[0,0:10].min() == 0, 'Scaling on a per vector basis not applied correctly'
    
                   
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
        
    def testSelect(self):
        '''Test the selection of generated features'''
        self.fg.generators = [gens.identity]
        self.fg.generate()
        selret = self.fg.select()
        assert len(self.fg.selected_weights) > 0, 'Bad selected weights output'
        assert len(self.fg.selected_fracs) > 0, 'Bad selected fracs output'
        assert (selret == self.fg.selected_fracs).all(), 'Select() did not return selected fracs appropriately'
        assert len(self.fg.selected_fracs) == self.fg.selected_weights.shape[0], \
                        'Selected fracs and weights do not have equal first dimension'
        
    def testSelectThreshold(self):
        '''Test the effect of the threshold parameter on feature selection'''
        self.fg.generators = [gens.identity]
        self.fg.generate()
        self.fg.select(0.00001)
        n01 = len(self.fg.selected_fracs)
        self.fg.select(0.5)
        n5 = len(self.fg.selected_fracs)
        sum5 = sum(self.fg.selected_fracs)
        assert n5 <= n01, 'Threshold has no effect on select()'
        assert sum5 < .5, 'Threshold does not bound eigenvalues correctly'
        
    def testApplication(self):
        '''Test the application of the generated weights on data'''
        self.fg.generators = [gens.identity]
        self.fg.generate()
        self.fg.select()
        
        applied_base = self.fg.apply_weights()
        
        assert len(applied_base.shape) == 2, 'apply_weights() returns non-2d shape for [] input'
        assert applied_base.shape[0] == self.fg.datas.shape[0], \
                        'apply_weights() doesnt keep first dimension constant on [] input'
        assert applied_base.shape[1] == len(self.fg.selected_fracs), \
                        'apply_weights() does not keep the same number of features as selected'
                    
        new_data = np.random.rand(int(self.fg.datas.shape[0]/2), self.fg.datas.shape[1], self.fg.datas.shape[2])
        applied_rand = self.fg.apply_weights(new_data)
        
        assert len(applied_rand.shape) == 2, 'apply_weights() returns non-2d shape for new data input'
        assert applied_rand.shape[0] == new_data.shape[0], \
                        'apply_weights() doesnt keep first dimension constant on new data input'
        assert applied_rand.shape != applied_base.shape, 'New input data had no effect on return from apply_weights()' 
        

 
class FeatureGeneratorTestSuite(unittest.TestSuite):
    def __init__(self):
        unittest.TestSuite.__init__(self,map(FeatureGeneratorTestCase,
                                                     ("testGenerate",
                                                      "testGeneratorScaling",
                                                      "testDwtGenerator",
                                                      "testFftGenerator"
                                                      "testSelect",
                                                      "testSelectThreshold",
                                                      "testApplication"
                                                      )))
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
