'''
Created on Jan 6, 2012

@author: jon
'''
from svmutil import *
import numpy as np
from regressor import SingleOutputExtensionRegressor, SingleOutputRegressor


#############################
# Support Vector Regression #
#############################

#### Parameter translation
base_parameters = dict({"svm_type": 4, "use_probability": 0})
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
    return tparams

class SupportVectorRegressor(SingleOutputExtensionRegressor):
    '''
    classdocs
    
    This is an implementation of Support-Vector Regression using
    libsvm. This class provides an n-to-n mapping.
    
    Parameter options:
    svm_type : (4) nu-SVR
    kernel_type : set type of kernel function (default 2)
        0 -- linear: u'*v
        1 -- polynomial: (gamma*u'*v + coef0)^degree
        2 -- radial basis function: exp(-gamma*|u-v|^2)
        3 -- sigmoid: tanh(gamma*u'*v + coef0)
    degree : set degree in kernel function (default 3)
    gamma : set gamma in kernel function (default 1/num_features)
    coef0 : set coef0 in kernel function (default 0)
    cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
    nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
    cache : set cache memory size in MB (default 100)
    termination : set tolerance of termination criterion (default 0.001)
    use_shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
    use_probability : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
    weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
    n-fold_validation : n-fold cross validation mode
    quiet : quiet mode (no outputs) (no value)
    '''
    
    def __init__(self, output_len, *unused_args):
        '''
        Construct a SVR-SingleOutputExtenstion Regressor
        '''
        constructor = lambda: svr_single()
        super(SupportVectorRegressor,self).__init__(output_len, constructor)

    

#### Single Output Class
class svr_single(SingleOutputRegressor):
    '''
    classdocs
    
    This is an implementation of Support-Vector Regression using
    libsvm. This class provides a n-to-1 mapping to be used by
    Regressor as part of an n-to-m mapping.
    
    Parameter options:
    svm_type : (4) nu-SVR
    kernel_type : set type of kernel function (default 2)
        0 -- linear: u'*v
        1 -- polynomial: (gamma*u'*v + coef0)^degree
        2 -- radial basis function: exp(-gamma*|u-v|^2)
        3 -- sigmoid: tanh(gamma*u'*v + coef0)
    degree : set degree in kernel function (default 3)
    gamma : set gamma in kernel function (default 1/num_features)
    coef0 : set coef0 in kernel function (default 0)
    cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
    nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
    cache : set cache memory size in MB (default 100)
    termination : set tolerance of termination criterion (default 0.001)
    use_shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
    use_probability : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
    weight : set the parameter C of class i to weight*C, for C-SVC (default 1)
    n-fold_validation : n-fold cross validation mode
    quiet : quiet mode (no outputs) (no value)
    '''
    model = [] # SVM model
    
    def train(self, in_vecs, exp_out_vals, params = dict()):
        if type(in_vecs) == type(np.array([])):
            in_vecs = in_vecs.tolist()
        if type(exp_out_vals) == type(np.array([])):
            exp_out_vals = exp_out_vals.tolist()
            
        problem = svm_problem(exp_out_vals, in_vecs)
        param = svm_parameter(libsvm_translate_params(params))
        self.model = svm_train(problem, param)
    
    def regress(self, in_vecs):
        if self.model == []:
            raise "Must call train() before regress()"
        
        y = [0.0] * len(in_vecs)
        if type(in_vecs) == type(np.array([])):
            in_vecs = in_vecs.tolist()
            
        prob_parameter = '-b 0'
            
        out_prediction = svm_predict(y, in_vecs, self.model, prob_parameter)[0]
        return out_prediction
    
    def prepareForSave(self, name):
        '''
        Save the model off if it's been trained
        '''
        if self.model != []:
            fn = name + ".svr.mod"
            svm_save_model(fn, self.model)
            self.model = []
            
    def recoverFromLoad(self, name):
        '''
        Reload the model if it exists
        '''
        fn = name + ".svr.mod"
        try:
            self.model = svm_load_model(fn)
        except IOError:
            pass # This means no file
        
        
        