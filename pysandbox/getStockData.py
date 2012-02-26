'''
Created on Feb 19, 2012

@author: jon
'''

import csv
import pickle
import ystockquote as ysq
import numpy as np

startdate = '19900101';
enddate = '20091231';

datadir = 'data/'
symbolFilename = 'nasdaq_companylist.csv'
outputFilename = 'nasdaq_packed_90_to_10.pkl';

def getStockData(symbols, startdate, enddate):
    data = dict()
    for symbol in symbols:
        quotes = ysq.get_historical_prices(symbol, startdate, enddate);
        if len(quotes[0]) != 7:
            print symbol + ' not found!'
            continue
        else:
            print symbol + ' found!'
        data[symbol] = quotes[1:]
    return np.array(data)

if __name__ == '__main__':
    f_symbol = open(datadir + symbolFilename, 'r');
    csv_symbol = csv.reader(f_symbol, delimiter=',', quotechar='"');
    symbols = [];
    for row in csv_symbol:
        symbols.append(row[0])
        
    data = getStockData(symbols, startdate, enddate);
    
    f_out = open(datadir + outputFilename, 'wb');
    pickle.dump(data, f_out);
    
    f_symbol.close();
    f_out.close();