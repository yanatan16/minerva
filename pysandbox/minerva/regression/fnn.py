'''
Created on Jan 10, 2012

@author: jon

TODO: Add more parameter options
'''
from regressor import Regressor
from pybrain.structure import FeedForwardNetwork, LinearLayer, SoftmaxLayer, FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import numpy as np

def get_parameter(params, name, default):
    if params.has_key(name):
        return params[name]
    else:
        return default

class FeedforwardNeuralNetworkRegressor(Regressor):
    '''
    classdocs
    '''
    ann = None
    
    def train(self, input_vectors, expected_output_vectors, params = dict()): 
        '''
        Trains the network by solving for its weights
        
        input_vectors: 2-d array of reals
        expected_output_vectors: 2-d array of reals
        ''' 
        nIn = input_vectors.shape[1]
        nOut = expected_output_vectors.shape[1]
        ann = self._constructNetwork(nIn, nOut, params)
        
        ds = SupervisedDataSet(nIn, nOut)
        ds.setField('input', input_vectors)
        ds.setField('target', expected_output_vectors)
        
        trainer = BackpropTrainer(ann, ds)
        trainer.trainUntilConvergence()
        
        self.ann = ann
                
    def regress(self, input_vectors):
        '''
        Regress activates the network with the input to predict the vectors
        
        input_vectors: 2-d array of reals
        returns: 2-d array of reals
        '''
        if self.ann == None:
            raise "Train must be called before Regress"
        return np.array(map(self.ann.activate, input_vectors))
    
    def _constructNetwork(self, nIn, nOut, params):
        ''' Construct the network '''
        nHidden = get_parameter(params, 'nHidden', 2)
        hiddenSize = np.empty(nHidden)
        for i in range(nHidden):
            pstr = 'hiddenSize[' + str(i) + ']'
            hiddenSize[i] = get_parameter(params, pstr, nIn + nOut)
        # Construct network
        ann = FeedForwardNetwork()
        
        # Add layers
        layers = []
        layers.append(LinearLayer(nIn))
        for nHid in hiddenSize:
            layers.append(SoftmaxLayer(nHid))
        layers.append(LinearLayer(nOut))
        ann.addOutputModule(layers[-1])
        ann.addInputModule(layers[0])
        for mod in layers[1:-1]:
            ann.addModule(mod)
        
        # Connections
        for i, mod in enumerate(layers):
            if i < len(layers) - 1:
                conn = FullConnection(mod, layers[i+1])
                ann.addConnection(conn)
        
        # Sort the modules
        ann.sortModules()
        return ann
