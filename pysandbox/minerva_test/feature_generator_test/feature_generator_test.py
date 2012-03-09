'''
Created on Dec 4, 2011

@author: Eisen
'''

import numpy as np
from minerva.features import FeatureGenerator
from minerva.features import generators as gens

import unittest

class FeatureGeneratorTestCase(unittest.TestCase):
    def testGenerate(self):
        '''Test the generation of features using the basic generators'''
        data = np.random.rand(200,3,10)
        fg = FeatureGenerator(data)
        gs = [gens.identity, gens.minimum, gens.maximum, gens.mean, gens.stdev]
        fg.generators = gs;
        fg.generated_features = [] # Reset the generator
        fg.generate()
        assert len(fg.generated_features) > 0, 'Bad generate() output'
        assert len(fg.generated_features.shape) == 2, 'Generated features is not 2-d'
        assert fg.generated_features.shape[0] == fg.datas.shape[0], \
                   'Number of observations is not held constant during generate()' 
        
        fg.generators = [gens.identity, gens.maximum, gens.minimum, gens.mean]
        fg.datas = np.array([[[1,2,3]]])
        fg.generate()
        expected_genfeat = np.array([[1,2,3,3,1,2]])
        assert (fg.generated_features == expected_genfeat).all(), 'Generated features are not aligned correctly'
                    
    def testGenerateScaling(self):
        '''Test feature scaling'''
        data = np.array([[np.arange(10),np.arange(10)-5]],dtype='float')
        params = dict({'scalaing':False})
        generators = [gens.identity]
        fg = FeatureGenerator(data, generators=generators, gen_params=params)
        fg.generate()
        
        assert len(fg.scaling[0]) > 0, 'Scaling not generated'
        assert len(fg.scaling[0]) == len(fg.scaling[1]), 'Scaling parameters sized inappropriately'
        assert len(fg.scaling[0]) == fg.datas.shape[1], 'Scaling parameters sized incorrectly'
        assert fg.scaling[0].max() == 1 and fg.scaling[0].min() == 1, \
                    'Scaling "off" did not produce correct multipliers'
        assert fg.scaling[1].max() == 0 and fg.scaling[1].min() == 0, \
                    'Scaling "off" did not produce correct summands'
        assert fg.generated_features.max() > 1, 'Scaling applied when off'
        
        fg.generator_params['scaling'] = True
        fg.generate()
        assert len(fg.scaling[0]) > 0, 'Scaling not generated with scaling on'
        assert len(fg.scaling[0]) == len(fg.scaling[1]), 'Scaling parameters sized inappropriately with scaling on'
        assert len(fg.scaling[0]) == fg.datas.shape[1], 'Scaling parameters sized incorrectly with scaling on'
        assert not (fg.scaling[0].max() == 1 and fg.scaling[0].min() == 1), \
                    'Scaling "on" did not produce correct multipliers'
        assert not (fg.scaling[1].max() == 0 and fg.scaling[1].min() == 0), \
                    'Scaling "on" did not produce correct summands'
        assert fg.generated_features.max() == 1, 'Scaling (maximum) not applied correctly'
        assert fg.generated_features.min() == -1, 'Scaling (minimum) not applied correctly'
        assert fg.generated_features[0,0:10].min() == 0, 'Scaling on a per vector basis not applied correctly'
        
    def testSelect(self):
        '''Test the selection of generated features'''
        data = np.random.rand(200,3,10)
        fg = FeatureGenerator(data)
        fg.generators = [gens.identity]
        fg.generate()
        selret = fg.select()
        assert len(fg.selected_weights) > 0, 'Bad selected weights output'
        assert len(fg.selected_fracs) > 0, 'Bad selected fracs output'
        assert (selret == fg.selected_fracs).all(), 'Select() did not return selected fracs appropriately'
        assert len(fg.selected_fracs) == fg.selected_weights.shape[0], \
                        'Selected fracs and weights do not have equal first dimension'
        
    def testSelectThreshold(self):
        '''Test the effect of the threshold parameter on feature selection'''
        data = np.random.rand(200,3,10)
        fg = FeatureGenerator(data)
        fg.generators = [gens.identity]
        fg.generate()
        fg.select(0.00001)
        n01 = len(fg.selected_fracs)
        fg.select(0.5)
        n5 = len(fg.selected_fracs)
        sum5 = sum(fg.selected_fracs)
        assert n5 <= n01, 'Threshold has no effect on select()'
        assert sum5 < .75, 'Threshold does not bound eigenvalues correctly'
        
    def testApplication(self):
        '''Test the application of the generated weights on data'''
        data = np.random.rand(200,3,10)
        fg = FeatureGenerator(data)
        fg.generators = [gens.identity]
        fg.generate()
        fg.select()
        
        applied_base = fg.apply_weights()
        
        assert len(applied_base.shape) == 2, 'apply_weights() returns non-2d shape for [] input'
        assert applied_base.shape[0] == fg.datas.shape[0], \
                        'apply_weights() doesnt keep first dimension constant on [] input'
        assert applied_base.shape[1] == len(fg.selected_fracs), \
                        'apply_weights() does not keep the same number of features as selected'
                    
        new_data = np.random.rand(int(fg.datas.shape[0]/2), fg.datas.shape[1], fg.datas.shape[2])
        applied_rand = fg.apply_weights(new_data)
        
        assert len(applied_rand.shape) == 2, 'apply_weights() returns non-2d shape for new data input'
        assert applied_rand.shape[0] == new_data.shape[0], \
                        'apply_weights() doesnt keep first dimension constant on new data input'
        assert applied_rand.shape != applied_base.shape, 'New input data had no effect on return from apply_weights()' 
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'PyMinerva.FeatureGeneratorTest']
    unittest.main()
