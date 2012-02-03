'''
Created on Jan 31, 2012

@author: Jon Eisen
'''
import unittest
from minerva.experiment import BaseExperiment
import numpy as np

class BaseExperimentTest(unittest.TestCase):
    
    def testNoExperiment(self):
        '''Testing a zero-dimensional experiment'''
        exp = BaseExperiment()
        runner = lambda: 87
        val = exp.run(runner, None, graph=False)
        assert val != None, 'Run did not return a value.'
        assert type(val) == int, 'Run did not return an integer.'
        assert val == runner(), 'Run did not return the proper value.'
        
    def testOneDimExperimentWithArray(self):
        '''Testing a one-dimensional experiment with array input'''
        exp = BaseExperiment()
        values = [range(10)]
        runner = lambda x: x**2 - x*10 + 1
        vals = exp.run(runner, values, graph=False)
        assert vals != None, 'Run did not return a value.'
        assert type(vals) == list, 'Run did not return an list.'
        assert len(vals) == len(values[0]), 'Run did not return a list of proper length.'
        assert type(vals[0]) == int, 'Run did not return a list of integers.'
        assert (vals == np.array(map(runner, values[0]))).all(), 'Run did not return the proper values.'
        
    def testTwoDimExperiment(self):
        '''Testing a two-dimensional experiment'''
        exp = BaseExperiment()
        values = (np.arange(10.0, dtype=float), np.arange(10.0, dtype=float)) # Use floats
        runner = lambda x, y: x**2-3*y**2+x*y-2*x+1.2
        vals = exp.run(runner, values, graph=False)
        assert vals != None, 'Run did not return a value.'
        assert type(vals) == list, 'Run did not return an list.'
        assert np.shape(vals) == (len(values[0]), len(values[1])), 'Run did not return an array of proper shape.'
        assert type(vals[0][0]) == np.float64, 'Run did not return an array of floats.'
        assert (vals == np.array([[runner(v1,v2) 
                          for v1 in values[0]]
                         for v2 in values[1]])).all(), 'Run did not return the proper values.'
                  
    #******** Note - True graphing capabilities must be tested manually. ******
class ManualExperimentTest(object): #TODO - Graphing doesn't work
    def testOneDimGraphOn(self):
        '''Verifying Graphing on doesn't break a one-dimensional experiment'''
        exp = BaseExperiment()
        values = range(10)
        runner = lambda x: x**2 - x*10 + 1
        vals = exp.run(runner, values, graph=True)
        assert (vals == np.array(map(runner, values))).all(), 'Run did not return the proper values.'
        
    def testTwoDimGraphOn(self):
        '''Verifying Graphing on doesn't break a two-dimensional experiment'''
        exp = BaseExperiment()
        values = (np.arange(10.0, dtype=float), np.arange(10.0, dtype=float)) # Use floats
        runner = lambda x, y: x**2-3*y**2+x*y-2*x+1.2
        vals = exp.run(runner, values, graph=True)
        assert (vals == np.array([[runner(v1,v2) 
                          for v1 in values[0]]
                         for v2 in values[1]])).all(), 'Run did not return the proper values.'

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'BaseExperimentTest.noExperimentTest']
    #unittest.main()
    
    print('Testing One Dimensional Graphing.')
    # Now run the manual Graph Test
    test = ManualExperimentTest()
    test.testOneDimGraphOn()
    test.testTwoDimGraphOn()
    
    