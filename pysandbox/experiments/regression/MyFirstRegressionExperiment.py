'''
Created on Feb 19, 2012

@author: jon
'''
from minerva.experiment import RegressionExperiment
import csv
import pickle
import numpy as np

datadir = '/home/jon/Code/Minerva/pysandbox/data/'
datafile = 'nasdaq_test_90_to_10.pkl'
testvars = dict({'seg:predictor_length': range(5,100,1)})
nontestvars = dict()
repeats = 10

if __name__ == '__main__':
    data_set = pickle.load(open(datadir + datafile,'rb'))
    exp = RegressionExperiment()
    outdata = exp.run(data_set, 
                      graph=True, 
                      disp=True, 
                      variables_under_test=testvars,
                      static_variables=nontestvars, 
                      repeats=repeats)
    
    