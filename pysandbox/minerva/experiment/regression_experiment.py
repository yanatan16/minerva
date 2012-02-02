'''
Created on Jan 6, 2012

@author: Jon Eisen
'''
from base_experiment import BaseExperiment
from minerva.data import timeSeriesSegmenter
from minerva.regression import LinearRegressor
from minerva.features import FeatureGenerator
from minerva.features import generators as gens
import numpy as np

class RegressionExperiment(BaseExperiment):
    '''
    classdocs
     
    This is a base experiment class for testing out regressors.
    It defines a run function for testing regression algorithms.
    '''
    
    # The default values for all available test variables
    default_test_variables = dict({
        # Functions on an array to predict (outputs of regression)
        'output_fncs':             [np.mean, np.std],
                                   
        'seg:predictor_length':    30,
        'seg:predictee_length':    5,
        'seg:allowable_overlap':   0,
        'seg:validation_split':    .25,
        
        'fg:generators':            [gens.identity],
        'fg:generator_params':      dict(),
        'fg:threshold':             .1,
        
        'reg:constructor':          LinearRegressor,
        'reg:training_params':      dict(),
        
        })
      
    def run(self, 
            data_set, 
            graph=True,            
            variables_under_test = dict(), # These values can be tested
            static_variables = dict()):
        '''
        Test a regression algorithm. Tests an arbitrary input to a
        regressor or feature generator. Graphs and returns output.
        
        regressor: A Regressor object to test
        data_set: A data set for time series regression
        graph: An optional (default true) boolean to graph the output
        
        Values to test:
        The possible keys that should be in the dictionaries
            variables_under_test and static_variables
        are defined in the variable self.default_test_variables
        '''
        nTest = len(variables_under_test)
        
        variables = self._merge_default_variables(static_variables)
        runner = self._make_runner(data_set,
                                   variables, 
                                   variables_under_test.keys())
            
        if nTest == 0:
            # Not testing any variable, just running
            graph = False
        
        return super(RegressionExperiment, self).run(
                 runner, variables_under_test.values(), graph)

    
    def _base_runner(self, runvars):
        ''' A            ctually run a test will all variables having single values 
        
        To run a case:
        - Segment the test data into training and validation
        - Extract the input data
        - Calculate the output data
        - Generate the training features and testing features
        - Train the regressor
        - Evaluate the regressor
        '''
        
        # Segment the test data
        train_data, test_data = timeSeriesSegmenter(
            runvars['data'],
            (runvars['seg:predictor_length'], runvars['seg:predictee_length']),
            allowable_overlap=runvars['seg:allowable_overlap'],
            validation_split=runvars['seg:validation_split'])
        
        # Extract the input data
        train_data_input = np.array([[d[0]] for d in train_data])
        test_data_input = np.array([[d[0]] for d in test_data])
        
        # Calculate the expected output data of this input data
        train_data_output = np.array([[fn([d[1]]) for fn in runvars['output_fncs']] for d in train_data])
        test_data_output = np.array([[fn([d[1]]) for fn in runvars['output_fncs']] for d in test_data])
        
        # Generate features
        fg = FeatureGenerator(train_data_input,
                              generators=runvars['fg:generators'],
                              gen_params=runvars['fg:generator_params'])
        train_features = fg.process(threshold=runvars['fg:threshold'])
        test_features = fg.apply_weights(test_data_input)
        
        # Train the regressor on the data
        regressor = runvars['reg:constructor'](len(runvars['output_fncs']))
        regressor.train(train_features, train_data_output, runvars['reg:training_params'])
        
        # Now evaluate the regressor
#        unused_train_mean_error, unused_train_mse, unused_train_stdev = regressor.evaluate(train_features, train_data_output)
        unused_test_mean_error, test_mse, unused_test_stdev = regressor.evaluate(test_features, test_data_output)
        
        return test_mse
    
    def _make_runner(self, data_set,
                     variables, test_var_names=[]):
        ''' Create a runner function for BaseExperiment '''
        nvars = len(test_var_names)
        variables['data'] = data_set
        if nvars == 0:
            def runner():
                return self._base_runner(variables)
        elif nvars == 1:
            def runner(var1):
                variables[test_var_names[0]] = var1
                return self._base_runner(variables)
        elif nvars == 2:
            def runner(var1, var2):
                variables[test_var_names[0]] = var1
                variables[test_var_names[1]] = var2
                return self._base_runner(variables)
        return runner
    
    def _merge_default_variables(self, new_variables):
        ''' Merge default variables with another dictionary '''
        defaults = dict()
        defaults.update(self.default_test_variables)
        defaults.update(new_variables)
        return defaults
  
