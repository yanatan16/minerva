'''
Created on Nov 25, 2011

@author: Eisen
'''

from ystockquote import get_historical_prices
import numpy
from matplotlib.mlab import PCA
import 

def get_data(quote, startdate, enddate):
    
    # Get Data
    data = get_historical_prices(quote, startdate, enddate)
    ac = [0.0] * (len(data) - 1)
    vol = [0] * (len(data) - 1)
    for j, row in enumerate(data[1:]):
        ac[j] = float(row[6])
        vol[j] = int(row[5])
    ror = [0.0] * (len(ac) - 1)
    for j, val in enumerate(ac[1:]):
        ror[j] = (val - ac[j]) / ac[j]
    
    # Now calc stats
    stats = []
    stats.append(numpy.min(ror));
    stats.append(numpy.max(ror));
    stats.append(numpy.mean(ror));
    stats.append(numpy.std(ror));
    stats.append(numpy.mean(vol));
    stats.append(numpy.std(vol));
    
    return stats

def mfnc():
    quotes = ["xom", "cvx", "cop", "hal", "slb", "apa", "apc", "dvn", "dal", "luv", "amr", "jblu", "aapl", "abc"]
    startdate = "20110101"
    enddate = "20110731"
    data = []
    for q in quotes:
        data.append(get_data(q, startdate, enddate))
    
    pca = PCA(numpy.array(data))
    
    print 'PCA Analysis on the statistics: '
    print 'minror, maxror, meanror, stdror, meanvol, stdvol'
    print ''
    print 'Principal components : fractions of variance:'
    print pca.fracs
    
    threshold = 0.1
    print 'Principal Component Weights (for fracs >', threshold, ')'
    for i, f in enumerate(pca.fracs):
        if (f > threshold):
            print i, pca.Wt[i,:]
        else:
            break

if __name__ == '__main__':
    mfnc()
    
    
    
    
            
        
    
      
    