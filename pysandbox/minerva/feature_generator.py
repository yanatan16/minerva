'''
Created on Dec 4, 2011

@author: Eisen
'''
from numpy import concatenate, matrix
from matplotlib.mlab import PCA

class FeatureGenerator(object):
    '''
    classdocs
    
    This class provides feature generation capability
    Simply pass in the data in this format:
        data_types x observations x time 
    Generators (defined in minerva.generators) will arbitrarily generate possible features
    Once generation and selection have occured, the process can be repeated using apply()
    '''
    generated_features = [];
    selected_fracs = [];
    selected_weights = [];

    def __init__(self, datas, generators = [], gen_params = []):
        '''
        Constructor
        '''
        self.datas = datas
        self.generators = generators
        self.generator_params = gen_params
        
    def generate_helper(self, datas):
        features = []
        for data in datas:
            for gen in self.generators:
                features += gen(data, self.generator_params)
        return concatenate(features)
        
    def generate(self):
        '''
        Generate features with current generators
        '''
        self.generated_features = self.generate_helper(self.datas)
        
    def select(self, threshold = 0.1):
        '''
        Select final features from generated ones using PCA.
        '''       
        if (self.generated_features == []):
            raise Exception("Must call generate()") 
        pca = PCA(self.generated_features)
        count = sum(pca.fracs > threshold)
        self.selected_fracs = pca.fracs[0:count]
        self.selected_weights = pca.Wt[0:count, :]
        return self.selected_fracs
    
    def apply(self, data = []):
        '''
        Apply the generation/selection process to arbitrary input data
        '''     
        if (self.selected_weights == []):
            raise Exception("Must call select()")
        
        if (data == []):
            gen_feat = self.generated_features
        else:
            gen_feat = self.generate_helper(data)
        
        weights = matrix(self.selected_weights)
        
        sel_feat = []
        for obs in gen_feat:
            obsmat = matrix(obs).transpose()
            obs_feat = weights * obsmat
            sel_feat += obs_feat
        
        return sel_feat 
        
   
        
        
        
        