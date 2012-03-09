#!/usr/bin/env python
'''
Created on Feb 19, 2012

@author: jon
'''

import csv
from minerva.data import *

def getQuotes(symbol, startdate, enddate):
    quotes = ystockquote.get_historical_prices(symbol, startdate, enddate)
    if len(quotes[0]) != 7:
        return None;
    ## Parse out headers, dates, and closing, 
    # we just need (open, adjclose, high, low, vol) information 
    return [[q[1],q[6],q[2],q[3],q[5]] for q in quotes[1:]];
    
def getStockData(symbols, startdate, enddate, filebase, QuoteWriter):
    writer = QuoteWriter(filebase, startdate, enddate)
    for symbol in symbols:
        quotes = getQuotes(symbol, startdate, enddate)
        if quotes is None:
            print symbol + ' not found!'
            continue
        else:
            print symbol + ' found!'
        writer.writeQuote(symbol, quotes)

if __name__ == '__main__':
    from optparse import OptionParser, OptionGroup
    import datetime
    
    formats = dict({    'csv':      QuoteWriterCsv,
                        'pickle':   QuoteWriterPickle   })
    usage = 'usage: %prog [options] [symbol] [symbol] ...'
    
    parser = OptionParser(usage=usage)
    parser.add_option('-o','--outfile', dest='out_filebase', 
                      help='Base output filename to save data',
                      action='store', type='string', 
                      default='stock_data_' + str(datetime.date.today()),
                      metavar='FILEBASE')
    parser.add_option('-s','--storage-format', dest='format',
                      help='Storage Format; choices: (' + ','.join(formats.keys()) + ')', 
                      choices=formats.keys(),
                      action='store', default='csv', metavar='FORMAT')
    parser.add_option('-b','--begin-date', dest='startdate',
                      help='Starting date for stocks (YYYYMMDD)',
                      type='string', action='store',
                      default='19900101', metavar='START')
    parser.add_option('-e','--end-date', dest='enddate',
                      help='Ending date for stocks (YYYYMMDD)',
                      type='string', action='store',
                      default='20101231', metavar='END')
    
    group = OptionGroup(parser, 'Symbol Input File (optional)')
    group.add_option('-i','--infile', dest='in_file',
                      help='Input csv file with symbols',
                      action='store', default=None, metavar='FILE')
    group.add_option('-c','--column', dest='col',metavar='COL',
                      help='Column number of the input file with the symbol names',
                      type='int', action='store', default=0)
    group.add_option('-d','--delimiter', dest='delimiter',
                      default=',', help='Delimiter for the input csv file',
                      type='string', action='store', metavar='D')
    group.add_option('-q','--quote-char', dest='quotechar',
                      default='"', help='Quote Character for the input csv file',
                      type='string', action='store', metavar='Q')
    parser.add_option_group(group)
    
    (options, symbols) = parser.parse_args()
    
    if options.in_file is not None:
        print("Reading symbols from input file " + options.in_file)
        f_symbol = open(options.in_file)
        csv_symbol = csv.reader(f_symbol, delimiter=options.delimiter, quotechar=options.quotechar)
        symbols = []
        for row in csv_symbol:
            symbols.append(row[options.col])
        f_symbol.close()
    
    print("Now grabbing and outputting stock data.")
    getStockData(symbols, options.startdate, options.enddate,
                 options.out_filebase, formats[options.format]);
                 