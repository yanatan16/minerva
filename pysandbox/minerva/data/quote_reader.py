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
    
class QuoteReaderCsv(QuoteReader):
    '''
    classdocs
    QuoteReader using CSV. 
    '''
    startdate = ''
    enddate = ''
    
    def __init__(self, fn_base):
        import csv
        file = open(fn_base + '.csv','rb')
        reader = csv.reader(file)
        firstRow = reader.next()
        self.startdate = firstRow[1]
        self.enddate = firstRow[4]
        
        for row in reader:
            symbol = row[1]
            data = []
            length = int(row[3])
            for unused in range(length):
                row = reader.next()
                data.append([float(row[0]),float(row[1]),float(row[2]),float(row[3]),long(float(row[4]))])
            self.data[symbol] = data
        
        file.close()

class QuoteReaderPickle(QuoteReader):
    '''
    classdocs
    QuoteReader using Pickle. 
    '''

    def __init__(self, fn_base):
        import pickle
        file = open(fn_base + '.pkl','rb')
        data = pickle.load(file)
        self.startdate = data['startdate']
        self.enddate = data['enddate']
        for i in range(len(data['symbols'])):
            self.data[data['symbols'][i]] = data['quotes'][i]
