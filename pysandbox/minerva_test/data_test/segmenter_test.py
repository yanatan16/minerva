'''
Created on Jan 17, 2012

@author: jon
'''
import unittest
from minerva.data import timeSeriesSegmenter
import numpy as np

class TimeSeriesSegmenterTestCase(unittest.TestCase):

    def testBasic(self):
        data = np.random.rand(5,20)
        segment_parts = (3,2)
        train_data, valid_data = timeSeriesSegmenter(data, segment_parts)
        
        assert len(train_data) == 15, 'Training data should be 75% of 20'
        assert len(valid_data) == 5, 'Validation data should be 25% of 20'
        assert len(train_data[0]) == 2, 'Each data tuple should have length equal to segment_parts'
        assert len(valid_data[0]) == 2, 'Each data tuple should have length equal to segment_parts'
        assert len(train_data[0][0]) == 3, 'The first segment array should be 3'
        assert len(train_data[0][1]) == 2, 'The second segment array should be 2'
        
    def testOverlap(self):
        data = np.random.rand(5,20)
        segment_parts = (3,2)
        train_data, valid_data = timeSeriesSegmenter(data, 
             segment_parts, allowable_overlap=2, validation_split=0)
        
        assert len(valid_data) == 0, 'Validation Data should be empty because split is 0'
        assert len(train_data) == 30, 'Overlap wasn\'t calculated correctly'
        
if __name__ == "__main__":
    import sys;sys.argv = ['', 'PyMinerva.TimeSeriesSegmenter']
    unittest.main()