'''
Created on Jan 17, 2012

@author: Jon Eisen
'''
import numpy as np
from minerva.graphing import plot_scatter, plot_image

class BaseExperiment(object):
    '''
    classdocs
    '''
    
    def run(self, experimental_values=None, graph=True):
        '''
        Run the experiment with a 1, 2, or 3 independent variables and graph it at the end
        
        experimental_values: an array or tuple of up to three arrays
        '''
        if experimental_values == None:
            self.run(self.experimental_values)
        else:
            self.run(experimental_values)
            
        if type(experimental_values) == tuple:
            ndim = len(experimental_values)
            lens = map(len, experimental_values)
            if ndim == 1:
                output = map(self.run_single, experimental_values[0])
                if graph:
                    self.graph(experimental_values[0], output)
            elif ndim == 2:
                output = [[self.run_single(ev1, ev2) 
                           for ev1 in experimental_values[0]] 
                          for ev2 in experimental_values[1]]
                if graph:
                    self.graph(experimental_values[0], experimental_values[1], output)
        else:
            output = map(self.run_single, experimental_values)
            if graph:
                self.graph(experimental_values, output)
        
            
    def graph(self, inputs, outputs):
        ''' Graph the experimental values and the output they created '''
        ndim = len(np.shape(inputs))
        if ndim == 1:
            plot_scatter(inputs, outputs)
        elif ndim == 2:
            plot_image(inputs[0], inputs[1], outputs)
    
    def run_single(self, values):
        '''
        Implement this function to run a single experiment given the values that is to be run
        
        Return a value to graph
        '''
        pass