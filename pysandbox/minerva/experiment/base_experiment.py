'''
Created on Jan 17, 2012

@author: Jon Eisen
'''
import numpy as np
from minerva.graphing import plot_scatter, plot_image

def none_runner(a=None,b=None):
    return None

class BaseExperiment(object):
    '''
    classdocs
    '''
    
    def run(self, runner=None, experimental_values=None, graph=True):
        '''
        Run the experiment with a 1, 2, or 3 independent variables and graph it at the end
        
        experimental_values: an array or tuple of up to two arrays (or set self.experimental_values)
        runner: A function that takes one or two inputs 
            (depending on the dimension of the experimental_values)
            and returns a number for analysis (or set self.run_single)
        graph: A boolean (default true) to graph the output
            
        Returns an array of output values corresponding to the experimental inputs
        '''
        if experimental_values == None or experimental_values == []:
            ndim = 0
        else:
            ndim = len(experimental_values)
            
        assert ndim <= 2, 'Can only experiment with two independent variables at this time.'
            
        if ndim == 0:
            output = runner()
            # Don't graph single output
        elif ndim == 1:
            output = map(runner, experimental_values[0])
            if graph:
                self._graph(experimental_values[0], output)
        elif ndim == 2:
            output = [[runner(e1,e2) 
                       for e1 in experimental_values[0]]
                      for e2 in experimental_values[1]]
            if graph:
                self._graph(experimental_values, output)
        
        return output
        
            
    def _graph(self, inputs, outputs):
        ''' Graph the experimental values and the output they created '''
        ndim = len(np.shape(inputs))
        if ndim == 1:
            plot_scatter(inputs, outputs)
        elif ndim == 2:
            plot_image(inputs[0], inputs[1], outputs)
    