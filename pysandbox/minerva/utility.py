'''
Created on Jan 8, 2012

@author: jon
'''

class Callable(object):
    '''
    classdocs
    Make a method static
    '''
    def __init__(self, anycallable):
        self.__call__ = anycallable