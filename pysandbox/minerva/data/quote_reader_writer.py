'''
Created on Feb 26, 2012

@author: jon
'''

class QuoteReader(object):
    '''
    QuoteReader Interface
    '''
    startdate = '' # String of date in form YYYYMMDD
    enddate = '' # String of date in form YYYYMMDD
    data = dict() # Dictionary of symbols to quotes
    
    def __init__(self, fn_base):
        '''
        Read in the file, populating the startdate and enddate strings
        Populate the data field with a dictionary of symbols to quotes data
        '''
        pass
    
    def symbols(self):
        return self.data.keys()
    def quotes(self):
        return self.data.values()
    def __getitem__(self, symbol):
        return self.data[symbol]
    
class QuoteWriter(object):
    '''
    QuoteWriter Interface
    '''
    
    def __init__(self, fn_base, startdate, enddate):
        pass
    def writeQuote(self, symbol, quotes):
        pass
    def close(self):
        pass
