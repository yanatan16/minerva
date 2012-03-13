'''
Created on Feb 26, 2012

@author: jon
'''
from quote_reader_writer import QuoteReader, QuoteWriter
import numpy as np
    
class QuoteReaderCsv(QuoteReader):
    '''
    classdocs
    QuoteReader using CSV. 
    '''
    startdate = ''
    enddate = ''
    
    def __init__(self, fn_base):
        import csv
        fid = open(fn_base + '.csv','rb')
        reader = csv.reader(fid)
        firstRow = reader.next()
        self.startdate = firstRow[1]
        self.enddate = firstRow[4]
        
        for row in reader:
            symbol = row[1]
            length = int(row[3])
            data = np.empty((5, length))
            for i in range(length):
                row = reader.next()
                data[:,i] = [float(row[0]),float(row[1]),float(row[2]),float(row[3]),long(float(row[4]))]
            self.data[symbol] = data
        
        fid.close()
            
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
        self.writer.writerow(['Symbol:',symbol,'Length:',len(quotes[0])])
        for q in np.transpose(quotes):
            self.writer.writerow(q)
    
    def close(self):
        self.file.close()
