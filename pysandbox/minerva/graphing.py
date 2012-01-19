'''
Created on Jan 18, 2012

@author: Jon Eisen

'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from pylab import *

def plot_image(x, y, z):
    """
    Plot a 2-d index with 1-d values on an image plot
    """
    X,Y = meshgrid(x, y)

    pcolor(X, Y, z)
    colorbar()
    show()
    
def plot_scatter(x, y, 
                 title='Scatter Plot',
                 xlabel='Y value',
                 ylabel='X value'):
    """
    Plot a scatter plot
    """
    import matplotlib
    
    assert len(x) == len(y)
    
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'o')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def plot_time_series(dates, values, 
                     title='Time Series Plot',
                     xlabel='Time',
                     ylabel='Value'):
    '''
    When plotting time series, eg financial time series, one often wants
    to leave out days on which there is no data, eg weekends.  The example
    below shows how to use an 'index formatter' to achieve the desired plot
    '''
    assert len(dates) == len(values)
    
    # first we'll do it the default way, with gaps on weekends
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dates, values, 'o-')
    fig.autofmt_xdate()
    
    # next we'll write a custom formatter
    N = len(dates)
    ind = np.arange(N)  # the evenly spaced plot indices
    
    def format_date(x, pos=None):
        thisind = np.clip(int(x+0.5), 0, N-1)
        return dates[thisind].strftime('%Y-%m-%d')
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ind, values, 'o-')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.autofmt_xdate()
    
    plt.show()
