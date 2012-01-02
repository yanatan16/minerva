'''
Created on Dec 4, 2011

@author: Eisen
'''
import numpy as np
from matplotlib.mlab import PCA
from minerva.generators import make_generator_executor

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
        
    def generate_helper(self, datas):
        '''Perform feature scaling and generation step on arbitrary data'''
        def generate_features(obs):
            gexec = make_generator_executor(obs, self.generator_params)
            return np.concatenate(map(gexec, self.generators))
        def apply_scaling(obs):
            for i in range(len(obs)):
                obs[i] = map(lambda x: self.scaling[0][i] * (x + self.scaling[1][i]), obs[i])
            return obs
            
        return np.array(map(generate_features, 
                            map(apply_scaling, datas)))
        
    def generate_scaling(self):
        '''Generate scaling parameters for generate_helper'''
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
            
        
    def generate(self):
        '''
        Generate scaling and features with current generators
        '''
        self.generate_scaling()
        self.generated_features = self.generate_helper(self.datas)
        
    def select(self, threshold = 0.1):
        '''
        Select final features from generated ones using PCA.
        '''       
        if len(self.generated_features) == 0:
            raise Exception('Must call generate() before select()')
        pca = PCA(self.generated_features)
        count = sum(pca.fracs.cumsum() < (1-threshold))
        self.selected_fracs = pca.fracs[0:count]
        self.selected_weights = pca.Wt[0:count, :]
        return self.selected_fracs
    
    def process(self, threshold = 0.1):
        self.generate()
        self.select(threshold)
        return self.apply()
    
    def apply_weights(self, data = []):
        '''
        Apply the computed weights process to arbitrary data
        If data is not included, this will apply to data used to generate weights
        '''     
        if len(self.selected_weights) == 0:
            raise Exception("Must call select() before applicator()")
        
        if (data == []):
            gen_feat = self.generated_features
        else:
            gen_feat = self.generate_helper(data)
        weights = np.matrix(self.selected_weights)
        
        sel_feat = np.array(map(lambda obs: np.array(weights * np.matrix(obs).T)[:,0], gen_feat))
        return sel_feat
