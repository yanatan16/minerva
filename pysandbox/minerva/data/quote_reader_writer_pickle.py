'''
Created on Feb 26, 2012

@author: jon
'''
from quote_reader_writer import QuoteReader, QuoteWriter

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
        
class QuoteReaderPickle(QuoteReader):
    '''
    classdocs
    QuoteReader using Pickle. 
    '''

    def __init__(self, fn_base):
        import pickle
        fid = open(fn_base + '.pkl','rb')
        data = pickle.load(fid)
        self.startdate = data['startdate']
        self.enddate = data['enddate']
        for i in range(len(data['symbols'])):
            self.data[data['symbols'][i]] = data['quotes'][i]
