'''
Created on Jan 17, 2012

@author: jon
'''
import unittest
from minerva.data import timeSeriesSegmenter
import numpy as np

class TimeSeriesSegmenterTestCase(unittest.TestCase):

    def testBasic(self):
        '''Test the Time Series Segmenter basically'''
        data = np.random.rand(5,20)
        segment_parts = (3,2)
        train_data, valid_data = timeSeriesSegmenter(data, segment_parts)

        assert len(train_data[0]) == 2, 'Each data tuple should have length equal to segment_parts'
        assert len(valid_data[0]) == 2, 'Each data tuple should have length equal to segment_parts'
        assert len(train_data[0][0]) == 3, 'The first segment array should be 3'
        assert len(train_data[0][1]) == 2, 'The second segment array should be 2'
        
    def testOverlap(self):
        '''Test the Time Series Segmenter overlap'''
        data = np.random.rand(5,20)
        segment_parts = (3,2)
        train_data, valid_data = timeSeriesSegmenter(data, 
             segment_parts, allowable_overlap=2, validation_split=0)
        
        assert len(train_data) == 30, 'Training data should be 100% of 30'
        assert len(valid_data) == 0, 'Validation data should be 0% of 30'
           
    def testSegmentOverlap(self):
        '''Test the Time Series Segmenter segment overlap'''
        data = np.random.rand(5,20)
        segment_parts = (3,3)
        train_data, valid_data = timeSeriesSegmenter(data, 
             segment_parts, allowable_overlap=0, validation_split=0, segment_overlap=(1,))
        
        assert len(train_data[0]) == 2, 'Each data tuple should have length equal to segment_parts'
        assert len(train_data[0][0]) == 3, 'The first segment array should be 3'
        assert len(train_data[0][1]) == 3, 'The second segment array should be 3'
        
        train_data = np.array(train_data)
        assert (train_data[:,0,2] == train_data[:,1,0]).all(), "Segment overlap did not occur"
        
    def testNonAlignedArray(self):
        '''Test the segmenter using an unaligned array'''
        data = np.array(map(np.random.rand,(20,23,15)))
        segment_parts = (3,2)
        train_data, valid_data = timeSeriesSegmenter(data, segment_parts, validation_split=.5)
        
        assert len(valid_data) > 0, 'Validation Data shouldn\'t be empty'
        assert len(train_data) > 0, 'Training data shouldn\'t be empty'
        
    def testMultidimensionData(self):
        '''Test the Time Series Segmenter using multi-dimension data'''
        data = np.random.rand(5,6,20)
        segment_parts = (3,2)
        train_data, valid_data = timeSeriesSegmenter(data, segment_parts)
        
        assert len(train_data[0]) == 2, 'Each data tuple should have length equal to segment_parts'
        assert len(valid_data[0]) == 2, 'Each data tuple should have length equal to segment_parts'
        assert len(train_data[0][0]) == 6, 'The first segment array should be of high dimension 6'
        assert len(train_data[0][0][0]) == 3, 'The first segment array should be of low dimension 3'
        assert len(valid_data[0][0]) == 6, 'The first segment array should be of high dimension 6'
        assert len(valid_data[0][0][0]) == 3, 'The first segment array should be of low dimension 3'
        assert len(train_data[0][1]) == 6, 'The second segment array should be of high dimension6'
        assert len(train_data[0][1][0]) == 2, 'The second segment array should be of low dimension 2'
        assert len(valid_data[0][1]) == 6, 'The second segment array should be of high dimension 6'
        assert len(valid_data[0][1][0]) == 2, 'The second segment array should be of low dimension 2'
        
    def testNonAlignedMultiDimArray(self):
        '''Test the segmenter using a multidimensional unaligned array'''
        doubleRand = lambda n: np.random.rand(2, n)
        data = np.array(map(doubleRand,(20,23,15)))
        segment_parts = (3,2)
        train_data, valid_data = timeSeriesSegmenter(data, segment_parts, validation_split=.5)
        
        assert len(valid_data) > 0, 'Validation Data shouldn\'t be empty'
        assert len(train_data) > 0, 'Training data shouldn\'t be empty'  
        assert len(train_data[0]) == 2, 'Each data tuple should have length equal to segment_parts'
        assert len(valid_data[0]) == 2, 'Each data tuple should have length equal to segment_parts'
        assert len(train_data[0][0]) == 2, 'The first segment array should be of high dimension 2'
        assert len(train_data[0][0][0]) == 3, 'The first segment array should be of low dimension 3'
        assert len(valid_data[0][0]) == 2, 'The first segment array should be of high dimension 2'
        assert len(valid_data[0][0][0]) == 3, 'The first segment array should be of low dimension 3'
        assert len(train_data[0][1]) == 2, 'The second segment array should be of high dimension 2'
        assert len(train_data[0][1][0]) == 2, 'The second segment array should be of low dimension 2'
        assert len(valid_data[0][1]) == 2, 'The second segment array should be of high dimension 2'
        assert len(valid_data[0][1][0]) == 2, 'The second segment array should be of low dimension 2'
        
    def testNonAlignedMultiDimWithSegOverlapArray(self):
        '''Test the segmenter using a multidimensional unaligned array with segment overlap'''
        doubleRand = lambda n: np.random.rand(2, n)
        data = np.array(map(doubleRand,(20,23,15)))
        segment_parts = (3,3)
        train_data, valid_data = timeSeriesSegmenter(data, segment_parts, segment_overlap=(1,))
        
        assert len(valid_data) > 0, 'Validation Data shouldn\'t be empty'
        assert len(train_data) > 0, 'Training data shouldn\'t be empty'  
        assert len(train_data[0]) == 2, 'Each data tuple should have length equal to segment_parts'
        assert len(valid_data[0]) == 2, 'Each data tuple should have length equal to segment_parts'
        assert len(train_data[0][0]) == 2, 'The first segment array should be of high dimension 2'
        assert len(train_data[0][0][0]) == 3, 'The first segment array should be of low dimension 3'
        assert len(valid_data[0][0]) == 2, 'The first segment array should be of high dimension 2'
        assert len(valid_data[0][0][0]) == 3, 'The first segment array should be of low dimension 3'
        assert len(train_data[0][1]) == 2, 'The second segment array should be of high dimension 2'
        assert len(train_data[0][1][0]) == 3, 'The second segment array should be of low dimension 3'
        assert len(valid_data[0][1]) == 2, 'The second segment array should be of high dimension 2'
        assert len(valid_data[0][1][0]) == 3, 'The second segment array should be of low dimension 3'
        
        assert train_data[0][0][0][2] == train_data[0][1][0][0], "Segment overlap did not occur"
        
    def testLargeSegmentation(self):
        '''Test the Time Series Segmenter with a large dataset (overlap, segment overlap, multidim, and unaligned)'''
        data = [np.random.rand(4,np.random.randint(100,2000)) for unused in range(200)]
        segment_parts = (30,5)
        segment_overlap = (1,)
        allowable_overlap = 1
        train_data, valid_data = timeSeriesSegmenter(data, segment_parts, allowable_overlap=allowable_overlap, segment_overlap=segment_overlap)
        
        assert len(train_data[0]) == 2, 'Each data tuple should have length equal to segment_parts'
        assert len(valid_data[0]) == 2, 'Each data tuple should have length equal to segment_parts'
        assert len(train_data[0][0]) == 4, 'The first dimension of a segment should be the number of data types'
        assert len(train_data[0][0][0]) == 30, 'The first segment array should be 30'
        assert len(train_data[0][1]) == 4, 'The first dimension of a segment should be the number of data types'
        assert len(train_data[0][1][0]) == 5, 'The second segment array should be 5'
        assert len(train_data) > len(valid_data), 'Training Data should be much larger than validation data'
        
    def testMultiPartSegment(self):
        '''Use a 4-part segment to segment the data'''
        data = np.random.rand(5,27)
        segment_parts = (3,2,3,1)
        train_data, valid_data = timeSeriesSegmenter(data, segment_parts)

        assert len(train_data) + len(valid_data) == 15, 'Total segments doesn\'t match'
        assert len(train_data[0]) == 4, 'Each data tuple should have length equal to segment_parts'
        assert len(train_data[0][0]) == 3, 'The first segment array should be 3'
        assert len(train_data[0][1]) == 2, 'The second segment array should be 2'
        assert len(train_data[0][2]) == 3, 'The third segment array should be 3'
        assert len(train_data[0][3]) == 1, 'The fourth segment array should be 1'
        
    def testMultiPartSegmentWithOverlap(self):
        '''Use a 4-part segment with overlaps to segment the data'''
        data = np.random.rand(5,31)
        segment_parts = (3,2,3,1)
        allowable_overlap = 1
        segment_overlap = (2,-1,1)
        train_data, valid_data = timeSeriesSegmenter(data, segment_parts)

        assert len(train_data) + len(valid_data) == 20, 'Total segments doesn\'t add up'
        assert len(train_data[0]) == 4, 'Each data tuple should have length equal to segment_parts'
        assert len(train_data[0][0]) == 3, 'The first segment array should be 3'
        assert len(train_data[0][1]) == 2, 'The second segment array should be 2'
        assert len(train_data[0][2]) == 3, 'The third segment array should be 3'
        assert len(train_data[0][3]) == 1, 'The fourth segment array should be 1'
            
    def testNoSegmentsError(self):
        '''Error case: No segment parts'''
        try:
            data = [[1,2]]
            segment_parts = []
            train, valid = timeSeriesSegmenter(data, segment_parts)
            assert False, 'Error was not raised for an empty segment'
        except AssertionError:
            # This is good
            pass
        
if __name__ == "__main__":
    import sys;sys.argv = ['', 'PyMinerva.TimeSeriesSegmenter']
    unittest.main()