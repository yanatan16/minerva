'''
Created on Jan 6, 2012

@author: jon
'''
from svm import svm_train, svm_parameter, svm_problem, svm_predict
import numpy as np


##################################
# Basic Regressor Implementation #
##################################

def RegressorImpl(object):
    def __init__(self, params):
        pass
    def train(self, in_vecs, out_vec):
        pass
    def regress(self, in_vecs):
        return [] # out_vecs


#############################
# Support Vector Regression #
#############################

base_parameters = dict({"svm_type": 4, "use_probability": 1})
parameter_translation = dict({
    "svm_type": "s", "kernel_type": "t", "degree": "d",
    "gamma": "g", "coef0": "r", "nu": "n", "epsilon": "p",
    "cache": "m", "termination": "e", "use_shrinking": "h",
    "use_probability": "b", "weight": "wi", "n-fold_validation": "v",
    "use_quiet": "q"})

def svm_translate_params(inparams):
    myparams = base_parameters
    myparams.update(inparams)
    tparams = ""
    for p in myparams:
        tparams += "-" + parameter_translation[p] + " " + myparams[p] + " "
    return svm_parameter(tparams)

class svr(object):
    '''
    classdocs
    '''
    params = [] # libsvm-formatted parameters
    model = [] # SVM model

    def __init__(self, params):
        '''
        Constructor
        '''
        # Save translated parameters
        self.params = svm_translate_params(params)
    
    def train_impl(self, in_vecs, exp_out_vec):
        if type(in_vecs) == type(np.array([])):
            in_vecs = in_vecs.tolist()
        if type(exp_out_vec) == type(np.array([])):
            exp_out_vec = exp_out_vec.tolist()
            
        problem = svm_problem(exp_out_vec, in_vecs)
        self.model = svm_train(problem, self.params)
    
    def regress_impl(self, in_vecs):
        y = [0.0] * len(in_vecs)
        if type(in_vecs) == type(np.array([])):
            in_vecs = in_vecs.tolist()
            
        out_vec = svm_predict(y, in_vecs, self.model)
        return out_vec
        