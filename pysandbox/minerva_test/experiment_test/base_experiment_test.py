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
        val = exp.run(runner, None, graph=False, disp=False)
        assert val != None, 'Run did not return a value.'
        assert type(val) == np.float64, 'Run did not return a float.'
        assert val == runner(), 'Run did not return the proper value.'
        
    def testOneDimExperiment(self, repeats=0):
        '''Testing a one-dimensional experiment with array input'''
        exp = BaseExperiment()
        values = [range(10)]
        runner = lambda x: x**2 - x*10 + 1
        vals = exp.run(runner, values, graph=False, disp=False, repeats=repeats)
        assert vals != None, 'Run did not return a value.'
        assert type(vals) == np.ndarray, 'Run did not return an np.ndarray.'
        assert len(vals) == len(values[0]), 'Run did not return a list of proper length.'
        assert type(vals[0]) == np.float64, 'Run did not return a list of floats.'
        assert (vals == np.array(map(runner, values[0]))).all(), 'Run did not return the proper values.'
        
    def testTwoDimExperiment(self, repeats=0):
        '''Testing a two-dimensional experiment'''
        exp = BaseExperiment()
        values = (np.arange(10.0, dtype=float), np.arange(10.0, dtype=float)) # Use floats
        runner = lambda x, y: x**2-3*y**2+x*y-2*x+1.2
        vals = exp.run(runner, values, graph=False, disp=False, repeats=repeats)
        assert vals != None, 'Run did not return a value.'
        assert type(vals) == np.ndarray, 'Run did not return an np.ndarray.'
        assert np.shape(vals) == (len(values[0]), len(values[1])), 'Run did not return an array of proper shape.'
        assert type(vals[0][0]) == np.float64, 'Run did not return an array of floats.'
        assert ((vals - np.array([[runner(v1,v2) 
                          for v1 in values[0]]
                         for v2 in values[1]])) < 0.0001).all(), 'Run did not return the proper values.'
    
    def test0DimWithRepeats(self):
        '''Run the 0-dimension test with repeats'''
        exp = BaseExperiment()
        runner = lambda: np.random.randn()
        val = exp.run(runner, None, graph=False, disp=False, repeats=100)
        assert val != None, 'Run did not return a value.'
        assert type(val) == np.float64, 'Run did not return a float.'
        assert np.abs(val) < 1, 'Run did not return the proper value.'
        
    def test1DimWithRepeats(self):
        '''Run the 1-dimension test with repeats'''
        self.testOneDimExperiment(20);
        
    def test2DimWithRepeats(self):
        '''Run the 2-dimension test with repeats'''
        self.testTwoDimExperiment(5);
    
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
    
    