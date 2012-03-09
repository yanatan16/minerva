'''
Created on Dec 4, 2011

@author: Eisen
'''
import numpy as np
from matplotlib.mlab import PCA
from types import FunctionType, ListType

class FeatureGenerator(object):
    '''
    classdocs
    
    This class provides feature generation capability
    Simply pass in the data in this format:
        observations x data_types x  time 
    Generators (defined in minerva.generators) will arbitrarily generate possible features
    Selection selects the best generated features using PCA
    Once generation and selection have occurred, the process can be repeated using apply()
    Apply will apply the generate the best features using the weights computed in selection
    '''
    generated_features = [];
    selected_fracs = [];
    selected_weights = [];
    scaling = np.empty((2,0));

    def __init__(self, datas, generators = [], gen_params = dict()):
        '''
        Constructor
        '''
        self.datas = np.array(datas, dtype='float')
        self.generators = generators
        self.generator_params = gen_params
        
    def generate(self):
        '''
        Generate scaling and features with current generators
        '''
        self._generate_scaling()
        self.generated_features = self._generate_helper(self.datas)
        
    def select(self, threshold = 0.1):
        '''
        Select final features from generated ones using PCA.
        '''       
        assert len(self.generated_features) > 0, 'Must call generate() before select()'
        pca = PCA(self.generated_features)
        count = sum(pca.fracs.cumsum() < (1-threshold)) + 1
        self.selected_fracs = pca.fracs[0:count]
        self.selected_weights = pca.Wt[0:count, :]
        return self.selected_fracs
    
    def process(self, threshold = 0.1):
        '''Perform generate, select, and application all in one call.'''
        self.generate()
        self.select(threshold)
        return self.apply_weights()
    
    def apply_weights(self, data = []):
        '''
        Apply the computed weights process to arbitrary data
        If data is not included, this will apply to data used to generate weights
        '''     
        assert len(self.selected_weights) > 0, "Must call select() before applicator()"
        
        if (data == []):
            gen_feat = self.generated_features
        else:
            gen_feat = self._generate_helper(data)
        weights = np.matrix(self.selected_weights)
        
        sel_feat = np.array(map(lambda obs: np.array(weights * np.matrix(obs).T)[:,0], gen_feat))
        return sel_feat
        
    def _make_generator_executor(self, data, params):
        return lambda fn: fn(data, params)
    
    def _generate_helper_features(self, obs):
        '''Generate features for a given observation'''
        assert len(self.generators) > 0, "No generators have been selected!"
        if len(np.shape(self.generators)) == 1 and type(self.generators[0]) == FunctionType:
            gexec = self._make_generator_executor(obs, self.generator_params)
            return np.concatenate(map(gexec, self.generators))
        else:
            assert len(obs) == len(self.generators), 'Generators must have first dimension length equal to number of data types'
            assert type(self.generators[0]) == ListType, 'Generators must be a 1 or 2 dimension list of functions'
            features = []
            for o, gs in zip(obs,self.generators):
                if len(gs) > 0:
                    gexec = self._make_generator_executor([o], self.generator_params)
                    features += np.concatenate(map(gexec, gs)).tolist()
            return np.array(features)
        
    def _generate_helper_apply_scaling(self,obs):
        '''Apply scaling to a given observation'''
        for i in range(len(obs)):
            obs[i] = map(lambda x: self.scaling[0][i] * (x + self.scaling[1][i]), obs[i])
        return obs
        
    def _generate_helper(self, datas):
        '''Perform feature scaling and generation step on arbitrary data'''
        return np.array(map(self._generate_helper_features, 
                            map(self._generate_helper_apply_scaling, datas)))
        
    def _generate_scaling(self):
        '''Generate scaling parameters for _generate_helper'''
        slen = self.datas.shape[1]
        self.scaling = np.empty((2,slen))
        if (self.generator_params.has_key('scaling') and \
                self.generator_params['scaling']):  
            for i in range(slen):
                data = np.array(self.datas[:,i,:])
                
                minimum = np.min(data)
                isneg = minimum < 0
                # If the data is negative, use a -1 to 1 instead of 0 to 1 interval
                if isneg:
                    median = np.median(data)
                    self.scaling[1][i] = -median
                else:
                    self.scaling[1][i] = -minimum
                    
                absmax = np.max(np.abs(data + self.scaling[1][i]))
                self.scaling[0][i] = 1.0 / absmax
        else:
            # unscaled
            self.scaling[0] = np.ones((slen))
            self.scaling[1] = np.zeros((slen))
            