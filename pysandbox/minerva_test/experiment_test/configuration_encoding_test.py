'''
Created on Apr 2, 2012

@author: jon
'''
import unittest
from minerva.experiment import deserializeConfiguration, serializeConfiguration
from minerva.features import generators as gens
from minerva.data import DataNormalizer, OutputFunction
from minerva.regression import SupportVectorRegressor
import numpy as np

class ConfigurationEncodingTest(unittest.TestCase):
    fn = 'test.cfg'

    def _serializeAndDeserialize(self, static, dynamic=dict()):
        if dynamic == dict():
            serializeConfiguration(self.fn,static)
        else:
            serializeConfiguration(self.fn, static, dynamic)
        (new_static, new_dynamic) = deserializeConfiguration(self.fn)
        
        assert static == new_static, 'Static Config not (de)serialized correctly.'
        assert dynamic == new_dynamic, 'Dynamic Config not (de)serialized correctly.'
        
    def testBasic(self):
        static = dict({
                       'aaa': 123,
                       'bbb': 'yoyoyo',
                       'happy_days': [1, 2, 3],
                       'sad_days': dict({'days?': 'yoyo'})
                   })
        self._serializeAndDeserialize(static)
        
    def testAdvanced(self):
        static = dict({
                    'fg:generators': [[gens.identity],
                                      [gens.identity,gens.mean,gens.stdev],
                                      [gens.identity],
                                      [gens.identity],
                                      [gens.identity,gens.maximum,gens.minimum]],
                    
                    'data_mapping': DataNormalizer(ror_divisor_row=1, volume_row=4),
                    'output_fncs': [OutputFunction(np.mean,1), OutputFunction(np.std,1)],
                    'reg:constructor': SupportVectorRegressor,
                    'reg:training_params': dict({'use_shrinking': '0'})
                   })
        self._serializeAndDeserialize(static)
        
    def testAdvancedWithDynamic(self):
        static = dict({
                    'fg:generators': [[gens.identity],
                                      [gens.identity,gens.mean,gens.stdev],
                                      [gens.identity],
                                      [gens.identity],
                                      [gens.identity,gens.maximum,gens.minimum]],
                    
                    'data_mapping': DataNormalizer(ror_divisor_row=1, volume_row=4),
                    'output_fncs': [OutputFunction(np.mean,1), OutputFunction(np.std,1)],
                    'reg:constructor': SupportVectorRegressor,
                    'reg:training_params': dict({'use_shrinking': '0'})
                   })
        dynamic = dict({'seg:predictor_length': range(10,50,2)})
        self._serializeAndDeserialize(static, dynamic)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()