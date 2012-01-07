'''
Created on Jan 6, 2012

@author: jon
'''
from svmutil import svm_train, svm_parameter, svm_problem, svm_predict
import numpy as np


##################################
# Basic Regressor Implementation #
##################################

class RegressorImplBase(object):
    def __init__(self, params):
        pass
    def train_impl(self, in_vecs, out_vec):
        pass
    def regress_impl(self, in_vecs):
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
    "quiet": "q"})

def libsvm_translate_params(inparams):
    myparams = base_parameters
    myparams.update(inparams)
    tparams = ""
    for p in myparams:
        tparams += "-" + parameter_translation[p] + " " + str(myparams[p]) + " "
    return svm_parameter(tparams)

class svr(RegressorImplBase):
    '''
    classdocs
    
    This is an implementation of Support-Vector Regression using
    libsvm. This class provides a n-to-1 mapping to be used by
    Regressor as part of an n-to-m mapping.
    
    Parameter options:
    svm_type : set type of SVM (default 0)
        0 -- C-SVC
        1 -- nu-SVC
        2 -- one-class SVM
        3 -- epsilon-SVR
        4 -- nu-SVR
    kernel_type : set type of kernel function (default 2)
        0 -- linear: u'*v
        1 -- polynomial: (gamma*u'*v + coef0)^degree
        2 -- radial basis function: exp(-gamma*|u-v|^2)
        3 -- sigmoid: tanh(gamma*u'*v + coef0)
        4 -- precomputed kernel (kernel values in training_set_file)
    degree : set degree in kernel function (default 3)
    gamma : set gamma in kernel function (default 1/num_features)
    coef0 : set coef0 in kernel function (default 0)
    cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
    nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
    epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
    cache : set cache memory size in MB (default 100)
    termination : set tolerance of termination criterion (default 0.001)
    use_shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
    use_probability : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
    weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
    n-fold_validation : n-fold cross validation mode
    quiet : quiet mode (no outputs) (no value)
    '''
    params = [] # libsvm-formatted parameters
    model = [] # SVM model

    def __init__(self, params):
        '''
        Constructor
        '''
        # Save translated parameters
        self.params = libsvm_translate_params(params)
    
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
        