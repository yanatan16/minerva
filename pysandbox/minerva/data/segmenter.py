'''
Created on Jan 6, 2012

@author: Jon Eisen
'''
import numpy as np
from types import ListType

class TimeSeriesSegmentMasterGenerator(object):
    ''' Segment a matrix or tensor of time series data into data segments ignoring original observation'''
    
    def __init__(self,
                 data,
                 segment_parts,
                 allowable_overlap=0,
                 segment_overlap=[]):
        assert len(segment_parts) > 0, 'Segment parts must be nonempty'
        assert (len(segment_overlap) == 0) or (len(segment_parts) == len(segment_overlap) + 1), \
                    'If segment overlap is defined, it must be of length parts minus one.'
                    
        increment = np.sum(segment_parts) - allowable_overlap - np.sum(segment_overlap)
        if len(segment_overlap) > 0:
            starts = [0] + [s - o for s, o in zip(segment_parts[0:-1],segment_overlap)]
        else:
            starts = [0] + list(segment_parts[0:-1])
        lengths = segment_parts
            
        v_constructor = lambda v: TimeSeriesSegmentVectorGenerator(v, increment, starts, lengths)
        
        self.vectors = map(v_constructor, data)
        self.index = 0
               
    def __iter__(self):
        return self
    
    def next(self):
        try:
            return self.vectors[self.index].next()
        except StopIteration:
            self.index += 1
            if self.index >= len(self.vectors):
                raise StopIteration
            else:
                return self.next()

class TimeSeriesSegmentVectorGenerator(object):
    ''' Segment a vector of data (possibly a 2-dim data type vs time matrix) into data segments'''
    def __init__(self, vec, inc, starts, lengths):
        self.vec = np.array(np.array(vec).tolist())
        self.inc = inc
        self.segments = zip(starts, lengths)
        self.total = starts[-1] + lengths[-1]
        self.index = -self.inc
        
        if type(vec[0]) == np.ndarray or type(vec[0]) == ListType:
            self.nsamps = len(vec[0])
            self.selectData = lambda start: [[vv[start+s:start+s+l] for vv in self.vec] for s, l in self.segments]
        else:
            self.nsamps = len(vec)
            self.selectData = lambda start: [self.vec[start+s:start+s+l] for s, l in self.segments]
            
    def __iter__(self):
        return self
    
    def next(self):
        self.index += self.inc
        if self.index + self.total > self.nsamps:
            raise StopIteration
        else:
            return self.selectData(self.index)
            
def timeSeriesSegmenter(data, 
             segment_parts,
             allowable_overlap=0,
             validation_split=0.25,
             segment_overlap=[]):
    '''
    A class for handling the extraction of meaningful individual data segments
    from a set of time series data.
    
    Each segment is comprised of parts of static lengths. These are specified by a 
    tuple passed in at construction. Also specified is the allowable overlap between
    individual segments in the time series data.
        
    Arguments:
    data: a two dimensional array of observables by time series
    segment_parts: a tuple of the parts of the segment lengths 
    allowable_overlap: optionally allow overlap value between whole segments.
    segment_overlap: optionally allow overlap in between segment parts
    '''
    
    segmentGenerator = TimeSeriesSegmentMasterGenerator(data, segment_parts, allowable_overlap, segment_overlap)
    
    test_data_set = []
    validation_data_set = []
    
    for segment in segmentGenerator:
        if np.random.rand() < validation_split:
            validation_data_set.append(segment)
        else:
            test_data_set.append(segment)
        
    return np.array(test_data_set), np.array(validation_data_set)

