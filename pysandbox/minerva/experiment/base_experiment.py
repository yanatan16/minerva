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
    xlabel = 'X Value'
    ylabel = 'Outputs'
    
    def run(self, runner=None, experimental_values=None, graph=True, disp=True, repeats=0):
        '''
        Run the experiment with a 1, 2, or 3 independent variables and graph it at the end
        1
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
        
        if disp:
            print 'Running', ndim, 'experiment with', repeats, 'repeats.'
            
        if ndim == 0:
            try:
                output = np.mean([runner() for unused in range(repeats+1)])
                if disp:
                    print output
            except Exception as e:
                print 'ERROR: Caught exception! ' + type(e) + ': ' + e
        elif ndim == 1:
            repeated_runner = lambda x: np.mean([runner(x) for unused in range(repeats+1)])
            
            output = np.empty(len(experimental_values[0]))
            for i, v in enumerate(experimental_values[0]):
                if disp:
                    print 'Now running experiment ', i, ' with value ', v
                try:
                    output[i] = repeated_runner(v)
                except Exception as e:
                    print 'ERROR: Caught exception! ' + type(e) + ': ' + e 
            if graph:
                self._graph(experimental_values[0], output)
            if disp:
                self._disp(experimental_values, output)
        elif ndim == 2:
            repeated_runner = lambda x, y: np.mean([runner(x, y) for unused in range(repeats+1)])
            output = np.empty((len(experimental_values[1]),len(experimental_values[0])))
            i = 0
            for j, v1 in enumerate(experimental_values[0]):
                for k, v2 in enumerate(experimental_values[1]):
                    if disp:
                        print 'Now running experiment ', i, ' with values ', v1, ' ', v2
                        i += 1
                    try:
                        output[k,j] = repeated_runner(v1,v2)
                    except Exception as e:
                        print 'ERROR: Caught exception! ' + type(e) + ': ' + e
            if graph:
                self._graph(experimental_values, output)
            if disp:
                self._disp(experimental_values, output)
        
        return output
        
    def _disp(self, inputs, outputs):
        ''' Graph the experimental values and the output they created '''
        ndim = len(np.shape(inputs))
        if ndim == 1:
            for i, o in zip(inputs, outputs):
                print i, o
        elif ndim == 2:
            if len(inputs) == 1:
                for i, o in zip(inputs[0], outputs):
                    print i, o
            else:
                for i1, i2 in [(i1,i2) for i1 in range(len(inputs[0])) for i2 in range(len(inputs[1]))]:
                    print inputs[0][i1], inputs[0][i2], outputs[i1][i2]
            
    def _graph(self, inputs, outputs):
        ''' Graph the experimental values and the output they created '''
        ndim = len(np.shape(inputs))
        if ndim == 1:
            plot_scatter(inputs, outputs, ylabel=self.ylabel, xlabel=self.xlabel)
        elif ndim == 2:
            plot_image(inputs[0], inputs[1], outputs)
    