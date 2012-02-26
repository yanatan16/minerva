'''
Created on Jan 6, 2012

@author: Jon Eisen
'''
import numpy as np


def timeSeriesSegmenter(data, 
             segment_parts,
             allowable_overlap=0,
             validation_split=0.25):
    '''
    A class for handling the extraction of meaningful individual data segments
    from a set of time series data.
    
    Each segment is comprised of parts of static lengths. These are specified by a 
    tuple passed in at construction. Also specified is the allowable overlap between
    individual segments in the time series data.
    
    After construction, this data can be requested in multiple ways. It can be 
    randomized and split into groups for training/validation. It can also be split along
    the time dimension and have segments streamed in time order.
        
    Arguments:
    data: a two dimensional array of observables by time series
    segment_parts: a tuple of the parts of the segment lengths 
    allowable_overlap: optionally allow overlap value between segments.
    '''
    data = np.array(data)
    
    nVectors = data.shape[0]
    nSamples = np.array(map(len, data))
    segment_length = np.sum(segment_parts)
    assert allowable_overlap < segment_length, 'Allowable overlap must not be as large as segment length.'
    
    nSegments = np.array(np.ceil((nSamples - segment_length + 1.0) / (segment_length - allowable_overlap)), dtype=np.int64)
    segment_starts = np.array(map(lambda x: np.arange(x) * (segment_length - allowable_overlap),nSegments))
    nTotal = np.sum(nSegments)
    
    nValidationSegments = int(np.floor(nTotal * validation_split))
    nTrainingSegments = nTotal - nValidationSegments
    
    unflattened_available_segments = [[[vec,start] for start in segment_starts[vec]] for vec in range(nVectors)]
    available_segments = np.empty((nTotal,2), dtype=int)
    counter = 0
    for avail_segs_vec in unflattened_available_segments:
        available_segments[counter:len(avail_segs_vec)+counter,:] = avail_segs_vec
        counter += len(avail_segs_vec);
    available_segments = available_segments.tolist()
    
    test_data_set = []
    validation_data_set = []
    
    def selectData(vec,start):
        out = []
        for seglen in segment_parts:
            out.append(data[vec][start:(start+seglen)])
            start += seglen
        return out
    
    for unused_i in np.arange(nTrainingSegments):
        selection = np.random.randint(len(available_segments))
        segment = available_segments.pop(selection)
        test_data_set.append(selectData(segment[0], segment[1]))
        
    for unused_i in np.arange(nValidationSegments):
        selection = np.random.randint(len(available_segments))
        segment = available_segments.pop(selection)
        validation_data_set.append(selectData(segment[0], segment[1]))
    
    return np.array(test_data_set), np.array(validation_data_set)

