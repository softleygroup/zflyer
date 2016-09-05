###This version of the program is only used as a library for the readDirectory and plotProfile function to be used by the testRotation.py file to compare simulated and experimental traces.

import os
import os.path
import re
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.optimize
import FigureSetup
from functools import reduce

def readDirectory(d, nofield=False):
    """ List directory contents, load files that match the file name
    pattern and extract the position information from the name. The pattern
    is ::
    <number>_INTscan_<HA|noHA|4down|4up>_P<1|3>_yag<number>_<y|norm><number>.mat

    The data set it cached in a file named 'allData.npz' to speed up loading on
    subsequent calls. This file is regenerated if it is missing.

    The signal is fit to a Gaussian with zero baseline, and the areas are saved
    in array `sig`

    nofield (boolean) : load Field-free reference (sumresult_ref) if True.
    """

    datafile = os.path.join(d, 'allData.npz')
    if os.path.exists(datafile):
        alldata = np.load(datafile)
        index = alldata['index']
        HA = alldata['HA']
        P = alldata['P']
        meas = alldata['meas']
        ypos = alldata['ypos']
        sig = alldata['sig']
        std = alldata['std']
        alldata.close()
        return index, HA, P, meas, ypos, sig, std


    namePattern = '(\d+)_INTscan_(.+)_P(\d+)_yag(\d+)_([a-z]+)(\d+).mat'
    if nofield:
        section = 'sumresult_ref'
    else:
        section = 'sumresult'

    files = os.listdir(d)
    HA = np.array(['']*len(files), dtype='|S50')
    meas = np.array(['']*len(files), dtype='|S50')
    P = np.zeros(len(files))
    ypos = np.zeros(len(files))
    sig = np.zeros(len(files))
    std = np.zeros(len(files))
    index = np.zeros(len(files))
    i=0
    for f in files:
        match = re.match(namePattern, f)
        if match is None:
            continue

        index[i] = match.group(1)
        HA[i] = match.group(2)
        P[i] = match.group(3)
        meas[i] = match.group(5)
        ypos[i] = match.group(6)

        dat = sio.loadmat(os.path.join(d, f))
        sig[i] = -np.mean(dat[section])
        std[i] = -np.min(dat[section])
        i = i+1

    # Trim arrays to length
    index=index[:i]
    HA=HA[:i]
    P=P[:i]
    meas=meas[:i]
    ypos=ypos[:i]
    sig=sig[:i]
    std=std[:i]
    
    # Sort by measurement index number
    order = np.argsort(index)
    index=index[order]
    HA=HA[order]
    P=P[order]
    meas=meas[order]
    ypos=ypos[order]
    sig=sig[order]
    std=std[order]
        
    np.savez(datafile, index=index, HA=HA, P=P, meas=meas,
            ypos=ypos, sig=sig, std=std)

    return index, HA, P, meas, ypos, sig, std


def plotProfile(ax, d, ha_name, peak_number, normalise=True, nofield=False, p0 = [0.0, 1.0, 15.0, 2.0], **plot_args):
    """Load the data, optionally normalise to the centre point, then fit a
    Gaussian though.  Normalisation uses a centre reading has been made at the
    start and end of a set of measurements. Then - assuming a linear change in
    intensity with time, and that each measurement is spaced fairly equally in
    time - divide each measurement by a linear interpolation between the two
    normalisation values.
    
    Arguments:
        ax (matplotlib.axes) : Graph in which to add a new plot.
        d (string) : Path to all the data.
        ha_name (string) : Name of data set to plot, 'HA', 'noHA', ...
        peak number (int) : Peak to process, (1, 3).
        color (string) : Matplotlib color specification for the plot points and
            fitted curve.
        label (string) : Legend label for this plot.
        normalise (bool) : True to perform normalisation, False to plot raw
            data.
    """

    index, HA, P, meas, ypos, sig, std = readDirectory(d, nofield)

    ind_HA = np.where(HA==ha_name)[0]
    assert len(ind_HA)>0, 'No data named ' + ha_name
    ind_p = np.where(P==peak_number)[0]
    ind_y = np.where(meas=='y')[0]

    ind_norm = np.where(meas=='norm')[0]

    ind_sig = reduce(np.intersect1d, (ind_HA, ind_p, ind_y))
    ind_norm = reduce(np.intersect1d, (ind_HA, ind_p, ind_norm))
    assert len(ind_norm)==2, 'Too many norm points'

    data = sig[ind_sig]
    norm_line = np.polyfit([np.min(ypos[ind_sig]), np.max(ypos[ind_sig])], sig[ind_norm], 1)
    if normalise:
        data = data/np.polyval(norm_line, ypos[ind_sig])
        
        #Scale all traces to 1
        data = data/np.max(data)
            
    #gauss_p = scipy.optimize.leastsq(lambda p : gauss(ypos[ind_sig],np.concatenate(([p0[0]], p[1:])))-data, p0)[0]
    
    def gauss_fit(x, p1, p2, p3):
        return gauss(x, [p0[0],p1,p2,p3])
        
    gauss_p, gauss_p_cov = scipy.optimize.curve_fit(gauss_fit, ypos[ind_sig], data, p0=p0[1:])
    gauss_p = np.concatenate([[p0[0]], gauss_p])
    error_p = np.sqrt(np.diag(gauss_p_cov))
    error_p = np.concatenate([[0], error_p])
    ss = np.sum((gauss(ypos[ind_sig],gauss_p)-data)**2, axis=0)
    print gauss_p, ss
    ax.scatter(-(ypos[ind_sig]-15.2), data, color=plot_args['color'])
    #xx = np.linspace(np.min(ypos[ind_sig]), np.max(ypos[ind_sig]), 100)
    xx = np.linspace(0, 30, 500)
    ax.plot(-(xx-15.2), gauss(xx, gauss_p), linestyle='solid', alpha=0.4, **plot_args)
    if normalise==False:
        ax.plot(xx, np.polyval(norm_line, xx), linestyle='solid', alpha=0.4, **plot_args)

    return gauss_p, error_p, ss
    
    
# Fit each slice along x to a Gaussian and add a dotted line to guide the eye.
def gauss(x, p):
    """ Define a Gaussian function with parameters:
        p[0] : y offset
        p[1] : y scale
        p[2] : x centre
        p[3] : standard deviation
    """
    return p[0] + p[1] * np.exp(-(x-p[2])**2/(2.0 * p[3]**2))


if __name__=='__main__':
    #f, ((ax1, ax2),(ax3, ax4)) = FigureSetup.new_figure(nrows=2, ncols=2, sharex='all', sharey='all')
    f, ax = FigureSetup.new_figure()

    directory = '.'
    p0 = [0.06, 1.0, 15.0, 2.0]
    normalise = True
    peak = 3

    colours = ['#88CCEE','#332288','#117733','#AA4499','#999933','#CC6677',
            '#882255','#44AA99','#DDCC77']

    p1, err_p1, ss1 = plotProfile(ax, d=directory, 
            ha_name='noHA', peak_number=peak, normalise=normalise, 
            nofield=False, p0=p0, color=colours[0], label='no HA')

    p2, err_p2, ss2 = plotProfile(ax, d=directory, 
            ha_name='HA', peak_number=peak, normalise=normalise, p0=p0,
            color=colours[1], label='HA')
    
    p3, err_p3, ss3 = plotProfile(ax, d=directory, 
            ha_name='4up', peak_number=peak, normalise=normalise,  p0=p0,
            color=colours[2], label='4 up')
            
    p4, err_p4, ss4 = plotProfile(ax, d=directory, 
            ha_name='2up', peak_number=peak, normalise=normalise, p0=p0,
            color=colours[3], label='2 up')
            
    p5, err_p5, ss5 = plotProfile(ax, d=directory, 
            ha_name='2down', peak_number=peak, normalise=normalise, p0=p0,
            color=colours[4], label='2 down')
            
    p6, err_p6, ss6 = plotProfile(ax, d=directory, 
            ha_name='4down', peak_number=peak, normalise=normalise, p0=p0,
            color=colours[5], label='4 down')
    
    ax.set_xlim(5, 25)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel('Normalised signal')
    ax.set_xlabel('Height (mm)')

    ax.legend(loc=2)
    
    pAll = np.vstack([p2,p3,p4,p5,p6])
    err_pAll = np.vstack([err_p2,err_p3,err_p4,err_p5,err_p6])
    gap = [0, 4, 2, -2, -4]
    ##fit = np.polyfit(gap, pAll[:,2], 1) #fits a straight line through the data without taking error bars into account
    def line(x, m, c):

        return m*x + c  
    fit2, _ = scipy.optimize.curve_fit(line, gap, pAll[:,2], sigma = err_pAll[:,2]) #fits a straight line through the data using error bars as weights
        
    f2, ax = FigureSetup.new_figure()
    plt.scatter(gap, pAll[:, 2], color ='#332288')
    plt.errorbar(gap, pAll[:, 2], yerr=err_pAll[:,2], fmt=' ', color ='#332288')
    ##plt.plot(gap, np.polyval(fit, gap), color = '#AA4499')
    plt.plot(gap, np.polyval(fit2, gap), color = '#117733')
    ax.set_xlim(-5, 5)
    ax.set_ylim(13.5, 16.5)
    ax.set_ylabel('Peak position in y (mm)')
    ax.set_xlabel('HA tilt (turns)')
    ##ax.annotate('y = {:0.3f} x + {:0.3f}'.format(fit[0], fit[1]), (2,17), color = '#AA4499')
    ax.annotate('y = {:0.3f} x + {:0.3f}'.format(fit2[0], fit2[1]), (0,16), color = '#117733')
    
    
    
    f.show()


    plt.show()

