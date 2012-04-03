'''
Created on Apr 2, 2012

@author: jon
'''

class OutputFunction(object):
    '''
    classdocs
    '''


    def __init__(self, func, row):
        '''
        Constructor
        '''
        self.func = func
        self.row = row
        
    def __call__(self, data):
        return self.func(data[self.row])
    
    def __eq__(self, other):
        return type(self) == type(other) and self.func == other.func and self.row == other.row