'''
Created on Feb 19, 2012

@author: jon
'''
import numpy as np
from minerva.experiment import RegressionExperiment, deserializeConfiguration
import minerva.features.generators as gens
from minerva.data import QuoteReaderCsv, QuoteReaderPickle, DataNormalizer
from minerva.regression import *
import sys

from optparse import OptionParser, OptionGroup

class RegressionExperimentRunner(object):
    
    readers = dict({ 'csv': QuoteReaderCsv,
                     'pkl': QuoteReaderPickle
                     })
    
    def __init__(self, arguments=sys.argv):
        usage = 'usage: %prog [options] DATA_FILE'
        parser = OptionParser(usage=usage)
        
        # Input configuration
        parser.add_option('-i','--input-file',dest='input_files',action='append',
                          type='string',metavar='FILE',
                          help='Configuration input file for a single run')
        
        parser.add_option('-f','--format',dest='format',action='store',choices=self.readers.keys(),
                          metavar='FMT',help='Data File Format',default='csv')
        
        parser.add_option('-o','--output-file',dest='output_file',action='store',type='string',
                          default='regresion_experiment.csv',help='Output File',metavar='FILE')
        
        parser.add_option('-q','--quiet',dest='verbose',action='store_false',type='bool',
                          default=True,help='Quiet the run')
                
        (options, datafile) = parser.parse_args(arguments)
        
        data = self.readers[options['format']](datafile)
        
        exp = RegressionExperiment()
        
        if options['input_files'] == []:
            print 'Running regression experiment using no input file'
            print exp.run(data.quotes(), graph=options['verbose'], disp=options['verbose'])
        else:
            for ifn in options['input_files']:
                print 'Running regression experiment using input file:', ifn
                (static_config, variable_config) = deserializeConfiguration(ifn) 
                print exp.run(
                        data.quotes(), 
                        graph=options['verbose'],
                        disp=options['verbose'],
                        static_variables=static_config,
                        variables_under_test=variable_config)
    
