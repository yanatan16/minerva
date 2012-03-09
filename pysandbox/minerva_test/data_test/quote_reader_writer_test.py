'''
Created on Feb 26, 2012

@author: jon
'''
import unittest
import numpy as np


class QuoteReaderWriterTest(unittest.TestCase):

    def testCsvQuoteReaderWriter(self):
        ''' Test the QuoteReaderCsv and QuoteWriterCsv '''
        from minerva.data import QuoteWriterCsv, QuoteReaderCsv
        self.runQuotesTest(QuoteWriterCsv, QuoteReaderCsv)
        
    def testPickleQuoteReaderWriter(self):
        ''' Test the QuoteReaderPickle and QuoteWriterPickle '''
        from minerva.data import QuoteWriterPickle, QuoteReaderPickle
        self.runQuotesTest(QuoteWriterPickle, QuoteReaderPickle)

    def runQuotesTest(self, writer_constructor, reader_constructor):
        startdate = '19861105'
        enddate = '19870929'
        symbols = ['ax1','ax2','ax3']
        data = np.random.rand(len(symbols), 200, 5)
        data[:,:,4] = np.int64(np.floor(data[:,:,4] * 100000))
        quotes = data.tolist()
        
        fn = 'quote_csv_test'
        
        writer = writer_constructor(fn, startdate, enddate)
        
        for s,q in zip(symbols,quotes):
            writer.writeQuote(s,q)
        
        writer.close()
        
        reader = reader_constructor(fn)
        
        assert reader.startdate == startdate, 'Start Date not read correctly'
        assert reader.enddate == enddate, 'End Date not read correctly'
        assert all([s in symbols for s in reader.symbols()]), 'Symbols not read correctly: [' + ','.join(reader.symbols()) + '] != [' + ','.join(symbols) + ']'
        assert all([(np.array(reader[s]) - data[i] < 1e-5).all() for i,s in enumerate(symbols)]), 'Data not read correctly'


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testCsvReaderWriter']
    unittest.main()