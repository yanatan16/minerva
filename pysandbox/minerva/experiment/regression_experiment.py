'''
Created on Jan 6, 2012

@author: jon
'''
from base_experiment import BaseExperiment

class RegressionExperiment(BaseExperiment):
    '''
    classdocs
    
    This is a base experiment class for testing out regressors.
    It defines a run function for testing regression algorithms.
    '''
    
    # The default values for all available test variables
    default_test_variables = dict({
        'segmentation_predictor_length':    30,
        'segmentation_predictee_length':    5,
        'segmentation_allowable_overlap':   0,
        'segmentation_validation_split':    .25
        #TODO Fill
        })
    
    def _base_runner(self, vars):
        ''' Actually run a test will all variables having single values '''
        #TODO implement
        pass
    
    def _make_runner(self, regressor, data_set,
                     variables, test_var_names=[]):
        ''' Create a runner function for BaseExperiment '''
        nvars = len(test_var_names)
        if nvars == 0:
            def runner():
                self._base_runner(variables)
        elif nvars == 1:
            def runner(var1):
                variables[test_var_names[0]] = var1
                self._base_runner(variables)
        elif nvars == 2:
            def runner(var1, var2):
                variables[test_var_names[0]] = var1
                variables[test_var_names[1]] = var2
                self._base_runner(variables)
        return runner
    
    def _merge_default_variables(self, new_variables):
        ''' Merge default variables with another dictionary '''
        defaults = dict()
        defaults.update(self.default_test_variables)
        defaults.update(new_variables)
        return defaults
    
    def run(self, 
            regressor, 
            data_set, 
            graph=True,
            
            # These values can be tested
            variables_under_test = dict(),
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
        assert nTest <= 2, "Can only test up to two variables at a time"
        
        variables = self._merge_default_variables(static_variables)
        runner = self._make_runner(regressor, data_set,
                                   variables, variables_under_test.keys())
            
        if nTest == 0:
            # Not testing any variable, just running
            graph = False
        
        return super(RegressionExperiment, self).run(
                 variables, runner, graph)

