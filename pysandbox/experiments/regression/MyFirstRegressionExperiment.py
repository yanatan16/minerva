'''
Created on Feb 19, 2012

@author: jon
'''
from minerva.experiment import RegressionExperiment
from minerva.data import QuoteReaderCsv, DataNormalizer
import numpy as np
import minerva.features.generators as gens
from minerva.regression import SupportVectorRegressor

datadir = '/home/jon/Code/Minerva/test_data/'
datafile = 'test_data_nasdaq_a'
# datafile = 'nasdaq_full_1990_to_2010'
testvars = dict({'seg:predictor_length': range(10,50,2)})
nontestvars = dict({
                    'fg:generators': [[gens.identity],
                                      [gens.identity,gens.mean,gens.stdev],
                                      [gens.identity],
                                      [gens.identity],
                                      [gens.identity,gens.maximum,gens.minimum]],
                    
                    'data_mapping': DataNormalizer(ror_divisor_row=1, volume_row=4),
                    'output_fncs': [lambda v: np.mean(v[1]), lambda v: np.std(v[1])],
                    'reg:constructor': SupportVectorRegressor,
                    'reg:training_params': dict({'use_shrinking': '0'})
                   })
repeats = 10

if __name__ == '__main__':
    data_file = QuoteReaderCsv(datadir + datafile)
    exp = RegressionExperiment()
    outdata = exp.run(data_file.quotes(), 
                      graph=True, 
                      disp=True, 
                      variables_under_test=testvars,
                      static_variables=nontestvars, 
                      repeats=repeats)
    
    