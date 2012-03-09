'''
Created on Feb 26, 2012

@author: jon
'''

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
    
class QuoteWriterCsv(QuoteWriter):
    '''
    classdocs
    QuoteWriter using CSV. 
    '''
    def __init__(self, fn_base, startdate, enddate):
        import csv
        self.file = open(fn_base + '.csv','wb')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['Startdate:', startdate,'','Enddate:',enddate])
         
    def writeQuote(self, symbol, quotes):
        self.writer.writerow(['Symbol:',symbol,'Length:',len(quotes)])
        for q in quotes:
            self.writer.writerow(q)
    
    def close(self):
        self.file.close()

class QuoteWriterPickle(QuoteWriter):
    '''
    classdocs
    QuoteWriter using Pickle. 
    Note: This does not manage memory well and might fail for large data sets (~500meg)
    '''

    def __init__(self, fn_base, startdate, enddate):
        self.file = open(fn_base + '.pkl','wb')
        self.data = dict()
        self.data['startdate'] = startdate
        self.data['enddate'] = enddate
        self.data['symbols'] = []
        self.data['quotes'] = []

    def writeQuote(self, symbol, quotes):
        self.data['symbols'].append(symbol)
        self.data['quotes'].append(quotes)
    
    def close(self):
        import pickle
        pickle.dump(self.data, self.file)
        self.file.close()