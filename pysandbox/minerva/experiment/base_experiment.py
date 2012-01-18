'''
Created on Jan 17, 2012

@author: jon
'''

class BaseExperiment(object):
    '''
    classdocs
    '''

    def main(self, experimental_values=None):
        '''
        This will run the experiments with input values stored in self.experimental_values
        or the values passed in.
        ''' 
        if experimental_values == None:
            self.run(self.experimental_values)
        else:
            self.run(experimental_values)
    
    def run(self, experimental_values, graph=True):
        '''
        Run the experiment with a 1, 2, or 3 independent variables and graph it at the end
        
        experimental_values: an array or tuple of up to three arrays
        '''
        if type(experimental_values) == tuple:
            ndim = len(experimental_values)
            lens = map(len, experimental_values)
            if ndim == 1:
                output = map(self.run_single, experimental_values[0])
            elif ndim == 2:
                output = [[self.run_single(ev1, ev2) 
                           for ev1 in experimental_values[0]] 
                          for ev2 in experimental_values[1]]
            elif ndim == 3:
                output = [[[self.run_single(ev1, ev2, ev3) 
                            for ev1 in experimental_values[0]] 
                           for ev2 in experimental_values[1]]
                          for ev3 in experimental_values[2]]
        else:
            output = map(self.run_single, experimental_values)
        
        if graph:
            self.graph(output)
            
    def graph(self):
                pass #TODO
    
    def run_single(self, values):
        '''
        Implement this function to run a single experiment given the values that is to be run
        
        Return a value to graph
        '''
        pass