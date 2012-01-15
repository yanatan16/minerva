'''
Created on Jan 8, 2012

@author: jon
'''


def get_parameter(params, name, default):
    if params.has_key(name):
        return params[name]
    else:
        return default