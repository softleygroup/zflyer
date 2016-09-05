from hexapole import HexArray
from Verlet import verletFlyer, loadFinal, rewind
from IntScanAnalysis import plotProfile

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import FigureSetup
import os
import os.path

# Set up logging and message detail level. Set the level to logging.INFO for a
# quieter output.
logger = logging.getLogger()
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(name)s - %(name)s - %(message)s'))
logging.getLogger().addHandler(ch)


def noHA (bins, state):
    # Load some atoms
    pos, vel, times = loadFinal(r'C:\Users\tpsgroup\Google Drive\Oxford\Zeeman\Simulations\HA\Sim HA\90deg seq', states=[state])

    # Speed things up by only flying the slower particles.
    # ind = np.where(vel[:,2]<0.300)[0]
    # pos = pos[ind, :]
    # vel = vel[ind, :]
    # times = times[ind]
    
    #Rewind back directly to detection position
    pos, vel, times = rewind(263.0, pos, vel, times)
    
    ind_pos = np.where(pos[:,2]>=263.0)[0]
    ind_tofmin = np.where(times>717.0)[0] # 717.0 for Peak 3, 422.0 for Peak 1
    ind_tofmax = np.where(times<777.0)[0] # 767.0 for Peak 3, 522.0 for Peak 1
    ind_tot = reduce(np.intersect1d, 
        (ind_pos, ind_tofmin, ind_tofmax))
    n, _ = np.histogram((pos[ind_tot, 1]), bins)
            
    return n

def HAtilt (position, angle, bins, state):

    pos_coil12 = 226.6
    pos_YAG = 263.0 - 0.0
    totalZ = 10.0 # where the Radia simulation ends and B=0 
        
    # Load some atoms
    pos, vel, times = loadFinal(r'C:\Users\tpsgroup\Google Drive\Oxford\Zeeman\Simulations\HA\Sim HA\90deg seq', states=[state])

    # Speed things up by only flying the slower particles.
    # ind = np.where(vel[:,2]<0.300)[0]
    # pos = pos[ind, :]
    # vel = vel[ind, :]
    # times = times[ind]

    # Initialise the hexapole.
    hh = HexArray('B.npz', position=position, angle=angle)
        
    # Move atoms back to starting position and transform coordinates into the
    # magnet frame of reference.
    pos, vel, times = rewind(pos_coil12, pos, vel, times)
    pos, vel = hh.toMagnet(pos, vel)

    # Fly the atoms
    pos, vel, times = verletFlyer(pos, vel, times, state=0, 
            hexapole=hh, totalZ=totalZ, dt=0.5, totalTime=500)
            #totalZ=21.85 for detection at 263 (263-21.85=241.15 HA position)
     
    #Pick out the particles that have not collided with the magnet
    ind_notcollided = hh.notCollided(pos)

    # Transform back into the lab frame and histogram detected positions.
    pos, vel = hh.toLab(pos, vel)
    
    #Fly them forward to the YAG
    pos[ind_notcollided,:], vel[ind_notcollided,:], times[ind_notcollided] = rewind(pos_YAG, pos[ind_notcollided,:], vel[ind_notcollided,:], times[ind_notcollided])

    #Select particles that reach the detection position (263) with a TOF between 717 and 767 mus (range of 50 mus like in the experiment, with 60mus (sim delay) subtracted from exptal values) - 422 to 522 mus (range 100mus like in exp) for Peak 1.
    ind_pos = np.where(pos[:,2]>=pos_YAG)[0]
    ind_tofmin = np.where(times>717.0)[0] #717.0 for Peak 3, 422.0 for Peak 1
    ind_tofmax = np.where(times<777.0)[0] #767.0 for Peak 3, 522.0 for Peak 1
    ind_tot = reduce(np.intersect1d, 
        (ind_pos, ind_tofmin, ind_tofmax))
    n, _ = np.histogram((pos[ind_tot, 1]), bins)
    
    return n
              
def ConvolvePlot (bins, data, laserwidth, ScalingFactor, ax, yoffset, colour):
    sumdata = np.sum(data[:,:], axis=1)
    gauss = np.exp(-(bins[:-1]**2/(2*laserwidth**2)))
    convdata = np.convolve(sumdata, gauss, mode='same')
    #convdata=sumdata
    if ScalingFactor==None:
        normdata = convdata/np.float(np.max(convdata))
        NormMax = np.float(np.max(convdata))
        ax.plot(bins[:-1], normdata+yoffset, color=colour, linestyle='solid')
        return NormMax
    else:
        normdata = convdata/ScalingFactor
        ax.plot(bins[:-1], normdata+yoffset, color=colour, linestyle=':')
        

    
    
#Testing the sensitivity to the alignment
dx = 0.0
dy = 0.0
dz = 0.0
da = 0.0
    
#HA Tilting     
bins = np.linspace(-15.0, 15.0, 100)
laserwidth = 0.4
yoffset = 0.06
nNoHA = np.zeros((len(bins)-1,4))
nHA = np.zeros((len(bins)-1,4))
n4up = np.zeros((len(bins)-1,4))
n2up = np.zeros((len(bins)-1,4))
n4down = np.zeros((len(bins)-1,4))
n2down = np.zeros((len(bins)-1,4))

datafile = os.path.join(r'C:\Users\tpsgroup\Google Drive\Oxford\Zeeman\Simulations\HA\Sim HA\90deg seq','TiltDataSaved.npz')
if os.path.exists(datafile):
   alldata = np.load(datafile)
   nNoHA = alldata['nNoHA']
   nHA = alldata['nHA']
   n4up = alldata['n4up']
   n2up = alldata['n2up']
   n4down = alldata['n4down']
   n2down = alldata['n2down']
   alldata.close()
else:
    for i in [0,1,2,3]:
        nNoHA[:,i] = noHA(bins=bins, state=i)
        nHA[:,i] = HAtilt(bins=bins, state=i, position=[0.0+dx, 0.0+dy, 241.15+dz], angle=[(0.0+da)/180*np.pi, 0.0, 0.0])
        n4up[:,i] = HAtilt(bins=bins, state=i, position=[0.0+dx, 0.0+dy, 241.15+dz], angle=[(25.0+da)/180*np.pi, 0.0, 0.0])
        n2up[:,i] = HAtilt(bins=bins, state=i, position=[0.0+dx, 0.0+dy, 241.15+dz], angle=[(12.5+da)/180*np.pi, 0.0, 0.0])
        n4down[:,i] = HAtilt(bins=bins, state=i, position=[0.0+dx, 0.0+dy, 241.15+dz], angle=[(-25.0+da)/180*np.pi, 0.0, 0.0])
        n2down[:,i] = HAtilt(bins=bins, state=i, position=[0.0+dx, 0.0+dy, 241.15+dz], angle=[(-12.5+da)/180*np.pi, 0.0, 0.0])
        
    np.savez(datafile, nNoHA=nNoHA, nHA=nHA, n4up=n4up, n2up=n2up, n4down=n4down, n2down=n2down)
    
#HA Vertical shift 
n100 = np.zeros((len(bins)-1,4))
n150 = np.zeros((len(bins)-1,4))
n180 = np.zeros((len(bins)-1,4))
n200 = np.zeros((len(bins)-1,4))
n220 = np.zeros((len(bins)-1,4))
n250 = np.zeros((len(bins)-1,4))
n300 = np.zeros((len(bins)-1,4))

datafile = os.path.join(r'C:\Users\tpsgroup\Google Drive\Oxford\Zeeman\Simulations\HA\Sim HA\90deg seq','ShiftDataSaved.npz')
if os.path.exists(datafile):
   alldata = np.load(datafile)
   n100 = alldata['n100']
   n150 = alldata['n150']
   n180 = alldata['n180']
   n200 = alldata['n200']
   n220 = alldata['n220']
   n250 = alldata['n250']
   n300 = alldata['n300']
   alldata.close()
else:
    for i in [0,1,2,3]:
        n100[:,i] = HAtilt(bins=bins, state=i, position=[0.0+dx, -1.0+dy, 241.15+dz], angle=[(0.0+da)/180*np.pi, 0.0, 0.0])
        n150[:,i] = HAtilt(bins=bins, state=i, position=[0.0+dx, -0.5+dy, 241.15+dz], angle=[(0.0+da)/180*np.pi, 0.0, 0.0])
        n180[:,i] = HAtilt(bins=bins, state=i, position=[0.0+dx, -0.2+dy, 241.15+dz], angle=[(0.0+da)/180*np.pi, 0.0, 0.0])
        n200[:,i] = HAtilt(bins=bins, state=i, position=[0.0+dx, 0.0+dy, 241.15+dz], angle=[(0.0+da)/180*np.pi, 0.0, 0.0])
        n220[:,i] = HAtilt(bins=bins, state=i, position=[0.0+dx, 0.2+dy, 241.15+dz], angle=[(0.0+da)/180*np.pi, 0.0, 0.0])
        n250[:,i] = HAtilt(bins=bins, state=i, position=[0.0+dx, 0.5+dy, 241.15+dz], angle=[(0.0+da)/180*np.pi, 0.0, 0.0])
        n300[:,i] = HAtilt(bins=bins, state=i, position=[0.0+dx, 1.0+dy, 241.15+dz], angle=[(0.0+da)/180*np.pi, 0.0, 0.0])
    np.savez(datafile, n100=n100, n150=n150, n180=n180, n200=n200, n220=n220, n250=n250, n300=n300)
        
        
#Figure 1: Ha tilting_ exp vs sim
f, ((ax1, ax2, ax3),(bx1, bx2, bx3)) = FigureSetup.new_figure(nrows=2, ncols=3, sharex='all', sharey='all')


colours = ['#88CCEE','#332288','#117733','#AA4499','#999933','#CC6677',
           '#882255','#44AA99','#AA7744', 'grey']
           
#Sim
NormMaxNoHA = ConvolvePlot (bins, nNoHA, laserwidth, None, ax1, yoffset, colour=colours[0])
NormMaxHA = ConvolvePlot (bins, nHA, laserwidth, None, bx1, yoffset, colour=colours[1])
NormMax4up = ConvolvePlot (bins, n4up, laserwidth, None, ax2, yoffset, colour=colours[2])
NormMax2up = ConvolvePlot (bins, n2up, laserwidth, None, ax3, yoffset, colour=colours[3])
NormMax4down = ConvolvePlot (bins, n4down, laserwidth, None, bx2, yoffset, colour=colours[4])
NormMax2down = ConvolvePlot (bins, n2down, laserwidth, None, bx3, yoffset, colour=colours[5])

#Show HFS contribution
ConvolvePlot (bins, nNoHA[:,2:4], laserwidth, NormMaxNoHA, ax1, yoffset, colour=colours[8])
ConvolvePlot (bins, nHA[:,2:4], laserwidth, NormMaxHA, bx1, yoffset, colour=colours[8])
ConvolvePlot (bins, n4up[:,2:4], laserwidth, NormMax4up, ax2, yoffset, colour=colours[8])
ConvolvePlot (bins, n2up[:,2:4], laserwidth, NormMax2up, ax3, yoffset, colour=colours[8])
ConvolvePlot (bins, n4down[:,2:4], laserwidth, NormMax4down, bx2, yoffset, colour=colours[8])
ConvolvePlot (bins, n2down[:,2:4], laserwidth, NormMax2down, bx3, yoffset, colour=colours[8])

#Show LFS contribution
ConvolvePlot (bins, nNoHA[:,0:2], laserwidth, NormMaxNoHA, ax1, yoffset, colour='grey')
ConvolvePlot (bins, nHA[:,0:2], laserwidth, NormMaxHA, bx1, yoffset, colour='grey')
ConvolvePlot (bins, n4up[:,0:2], laserwidth, NormMax4up, ax2, yoffset, colour=colours[9])
ConvolvePlot (bins, n2up[:,0:2], laserwidth, NormMax2up, ax3, yoffset, colour=colours[9])
ConvolvePlot (bins, n4down[:,0:2], laserwidth, NormMax4down, bx2, yoffset, colour=colours[9])
ConvolvePlot (bins, n2down[:,0:2], laserwidth, NormMax2down, bx3, yoffset, colour=colours[9])

#Load and plot experimental data (load from directory folder and plot using IntScanAnalysis.py in the zflyer directory as a library for the plotProfile function

directory = r'C:\Users\tpsgroup\Google Drive\Oxford\Zeeman\Analysis\All Data\2016\July2016\2016_07_28\INTscans tilting expts'
p0 = [0.06, 1.0, 15.0, 2.0]
normalise = True
peak = 3


p1, err_p1, ss1 = plotProfile(ax1, d=directory, 
            ha_name='noHA', peak_number=peak, normalise=normalise, 
            nofield=False, p0=p0, color=colours[0], label='no HA')

p2, err_p2, ss2 = plotProfile(bx1, d=directory, 
            ha_name='HA', peak_number=peak, normalise=normalise, p0=p0,
            color=colours[1], label='HA')
    
p3, err_p3, ss3 = plotProfile(ax2, d=directory, 
            ha_name='4up', peak_number=peak, normalise=normalise,  p0=p0,
            color=colours[2], label='4 up')
            
p4, err_p4, ss4 = plotProfile(ax3, d=directory, 
            ha_name='2up', peak_number=peak, normalise=normalise, p0=p0,
            color=colours[3], label='2 up')
            
p5, err_p5, ss5 = plotProfile(bx2, d=directory, 
            ha_name='4down', peak_number=peak, normalise=normalise, p0=p0,
            color=colours[4], label='4 down')
            
p6, err_p6, ss6 = plotProfile(bx3, d=directory, 
            ha_name='2down', peak_number=peak, normalise=normalise, p0=p0,
            color=colours[5], label='2 down')
    
ax1.set_xlim(-15, 15)
ax1.set_ylim(0, 1.3)
ax1.set_ylabel('Normalised signal')
bx1.set_xlabel('Height (mm)')
ax1.legend(loc=1)
ax2.legend(loc=1)
ax3.legend(loc=1)
bx1.legend(loc=1)
bx2.legend(loc=1)
bx3.legend(loc=1)
ax1.plot([0, 0], [0, 2], color='k', linestyle='--', linewidth=1.0)
ax2.plot([0, 0], [0, 2], color='k', linestyle='--', linewidth=1.0)
ax3.plot([0, 0], [0, 2], color='k', linestyle='--', linewidth=1.0)
bx1.plot([0, 0], [0, 2], color='k', linestyle='--', linewidth=1.0)
bx2.plot([0, 0], [0, 2], color='k', linestyle='--', linewidth=1.0)
bx3.plot([0, 0], [0, 2], color='k', linestyle='--', linewidth=1.0)


f.subplots_adjust(hspace=0.05, wspace=0.05)
#plt.show()

# figure 2: HA vertical shift_ exp vs sim
f, ((cx1, cx2, cx3, cx4),(dx1, dx2, dx3, dx4)) = FigureSetup.new_figure(nrows=2, ncols=4, sharex='all', sharey='all')

colours = ['#88CCEE','#332288','#117733','#AA4499','#999933','#CC6677',
           '#882255','#44AA99','#AA7744', 'grey']
           
#Sim
NormMaxNoHA = ConvolvePlot (bins, nNoHA, laserwidth, None, cx1, yoffset, colour=colours[0])
NormMaxn100 = ConvolvePlot (bins, n100, laserwidth, None, cx2, yoffset, colour=colours[1])
NormMaxn150 = ConvolvePlot (bins, n150, laserwidth, None, cx3, yoffset, colour=colours[2])
NormMaxn180 = ConvolvePlot (bins, n180, laserwidth, None, cx4, yoffset, colour=colours[3])
NormMaxn200 = ConvolvePlot (bins, n200, laserwidth, None, dx1, yoffset, colour=colours[4])
NormMaxn220 = ConvolvePlot (bins, n220, laserwidth, None, dx2, yoffset, colour=colours[5])
NormMaxn250 = ConvolvePlot (bins, n250, laserwidth, None, dx3, yoffset, colour=colours[6])
NormMaxn300 = ConvolvePlot (bins, n300, laserwidth, None, dx4, yoffset, colour=colours[7])

#Show HFS contribution
ConvolvePlot (bins, nNoHA[:,2:4], laserwidth, NormMaxNoHA, cx1, yoffset, colour=colours[8])
ConvolvePlot (bins, n100[:,2:4], laserwidth, NormMaxn100, cx2, yoffset, colour=colours[8])
ConvolvePlot (bins, n150[:,2:4], laserwidth, NormMaxn150, cx3, yoffset, colour=colours[8])
ConvolvePlot (bins, n180[:,2:4], laserwidth, NormMaxn180, cx4, yoffset, colour=colours[8])
ConvolvePlot (bins, n200[:,2:4], laserwidth, NormMaxn200, dx1, yoffset, colour=colours[8])
ConvolvePlot (bins, n220[:,2:4], laserwidth, NormMaxn220, dx2, yoffset, colour=colours[8])
ConvolvePlot (bins, n250[:,2:4], laserwidth, NormMaxn250, dx3, yoffset, colour=colours[8])
ConvolvePlot (bins, n300[:,2:4], laserwidth, NormMaxn300, dx4, yoffset, colour=colours[8])

#Show LFS contribution
ConvolvePlot (bins, nNoHA[:,0:2], laserwidth, NormMaxNoHA, cx1, yoffset, colour=colours[9])
ConvolvePlot (bins, n100[:,0:2], laserwidth, NormMaxn100, cx2, yoffset, colour=colours[9])
ConvolvePlot (bins, n150[:,0:2], laserwidth, NormMaxn150, cx3, yoffset, colour=colours[9])
ConvolvePlot (bins, n180[:,0:2], laserwidth, NormMaxn180, cx4, yoffset, colour=colours[9])
ConvolvePlot (bins, n200[:,0:2], laserwidth, NormMaxn200, dx1, yoffset, colour=colours[9])
ConvolvePlot (bins, n220[:,0:2], laserwidth, NormMaxn220, dx2, yoffset, colour=colours[9])
ConvolvePlot (bins, n250[:,0:2], laserwidth, NormMaxn250, dx3, yoffset, colour=colours[9])
ConvolvePlot (bins, n300[:,0:2], laserwidth, NormMaxn300, dx4, yoffset, colour=colours[9])

#Exp
directory = r'C:\Users\tpsgroup\Google Drive\Oxford\Zeeman\Analysis\All Data\2016\July2016\2016_07_26\INTscans'
p0 = [0.06, 1.0, 15.0, 2.0]
normalise = True
peak = 3

p1, err_p1, ss1 = plotProfile(cx1, d=directory, 
            ha_name='noHA', peak_number=peak, normalise=normalise, 
            nofield=False, p0=p0, color=colours[0], label='no HA')

p2, err_p2, ss2 = plotProfile(cx2, d=directory, 
            ha_name='1.00mm', peak_number=peak, normalise=normalise, p0=p0,
            color=colours[1], label='1.00 mm')
    
p3, err_p3, ss3 = plotProfile(cx3, d=directory, 
            ha_name='1.50mm', peak_number=peak, normalise=normalise,  p0=p0,
            color=colours[2], label='1.50 mm')
            
p4, err_p4, ss4 = plotProfile(cx4, d=directory, 
            ha_name='1.80mm', peak_number=peak, normalise=normalise, p0=p0,
            color=colours[3], label='1.80 mm')
            
p5, err_p5, ss5 = plotProfile(dx1, d=directory, 
            ha_name='2.00mm', peak_number=peak, normalise=normalise, p0=p0,
            color=colours[4], label='2.00 mm')
            
p6, err_p6, ss6 = plotProfile(dx2, d=directory, 
            ha_name='2.20mm', peak_number=peak, normalise=normalise, p0=p0,
            color=colours[5], label='2.20 mm')
            
p7, err_p7, ss7 = plotProfile(dx3, d=directory, 
            ha_name='2.50mm', peak_number=peak, normalise=normalise, p0=p0,
            color=colours[6], label='2.50 mm')
            
p8, err_p8, ss8 = plotProfile(dx4, d=directory, 
            ha_name='3.00mm', peak_number=peak, normalise=normalise, p0=p0,
            color=colours[7], label='3.00 mm')
    
cx1.set_xlim(-15, 15)
cx1.set_ylim(0, 1.3)
cx1.set_ylabel('Normalised signal')
dx1.set_xlabel('Height (mm)')
cx1.legend(loc=1)
cx2.legend(loc=1)
cx3.legend(loc=1)
cx4.legend(loc=1)
dx1.legend(loc=1)
dx2.legend(loc=1)
dx3.legend(loc=1)
dx4.legend(loc=1)
cx1.plot([0, 0], [0, 2], color='k', linestyle='--', linewidth=1.0)
cx2.plot([0, 0], [0, 2], color='k', linestyle='--', linewidth=1.0)
cx3.plot([0, 0], [0, 2], color='k', linestyle='--', linewidth=1.0)
cx4.plot([0, 0], [0, 2], color='k', linestyle='--', linewidth=1.0)
dx1.plot([0, 0], [0, 2], color='k', linestyle='--', linewidth=1.0)
dx2.plot([0, 0], [0, 2], color='k', linestyle='--', linewidth=1.0)
dx3.plot([0, 0], [0, 2], color='k', linestyle='--', linewidth=1.0)
dx4.plot([0, 0], [0, 2], color='k', linestyle='--', linewidth=1.0)


f.subplots_adjust(hspace=0.05, wspace=0.05)

# Quantitative analysis

# HFSPercentageNoHA = np.sum(nNoHA[:,2:4])/np.sum(nNoHA[:,:])*100
# HFSPercentageHA = np.sum(nHA[:,2:4])/np.sum(nHA[:,:])*100
# print('Percentage HFS particles without HA = %.1f' % HFSPercentageNoHA)
# print('Percentage HFS particles with HA = %.1f' % HFSPercentageHA)
# LFSPercentageNoHA = np.sum(nNoHA[:,0:2])/np.sum(nNoHA[:,:])*100
# LFSPercentageHA = np.sum(nHA[:,0:2])/np.sum(nHA[:,:])*100
# print('Percentage LFS particles without HA = %.1f' % LFSPercentageNoHA)
# print('Percentage LFS particles with HA = %.1f' % LFSPercentageHA)

HFSNumberNoHA = np.sum(nNoHA[:,2:4])
LFSNumberNoHA = np.sum(nNoHA[:,0:2])
print('Number HFS particles without HA = %.1f' % HFSNumberNoHA)
print('Number LFS particles without HA = %.1f' % LFSNumberNoHA)
HFSNumberHA = np.sum(nHA[:,2:4])
LFSNumberHA = np.sum(nHA[:,0:2])
print('Number HFS particles with HA = %.1f' % HFSNumberHA)
print('Number LFS particles with HA = %.1f' % LFSNumberHA)
HFSNumber200 = np.sum(n200[:,2:4])
LFSNumber200 = np.sum(n200[:,0:2])
print('Number HFS particles 200(with HA) = %.1f' % HFSNumber200)
print('Number LFS particles 200(with HA) = %.1f' % LFSNumber200)

HFSNumber4up = np.sum(n4up[:,2:4])
LFSNumber4up = np.sum(n4up[:,0:2])
print('Number HFS particles tilt 4 up = %.1f' % HFSNumber4up)
print('Number LFS particles tilt 4 up = %.1f' % LFSNumber4up)
HFSNumber4down = np.sum(n4down[:,2:4])
LFSNumber4down = np.sum(n4down[:,0:2])
print('Number HFS particles tilt 4 down = %.1f' % HFSNumber4down)
print('Number LFS particles tilt 4 down = %.1f' % LFSNumber4down)
HFSNumber2up = np.sum(n2up[:,2:4])
LFSNumber2up = np.sum(n2up[:,0:2])
print('Number HFS particles tilt 2 up = %.1f' % HFSNumber2up)
print('Number LFS particles tilt 2 up = %.1f' % LFSNumber2up)
HFSNumber2down = np.sum(n2down[:,2:4])
LFSNumber2down = np.sum(n2down[:,0:2])
print('Number HFS particles tilt 2 down = %.1f' % HFSNumber2down)
print('Number LFS particles tilt 2 down = %.1f' % LFSNumber2down)

HFSNumber300 = np.sum(n300[:,2:4])
LFSNumber300 = np.sum(n300[:,0:2])
print('Number HFS particles shift 300 (up) = %.1f' % HFSNumber300)
print('Number LFS particles shift 300 (up) = %.1f' % LFSNumber300)
HFSNumber100 = np.sum(n100[:,2:4])
LFSNumber100 = np.sum(n100[:,0:2])
print('Number HFS particles shift 100 (down) = %.1f' % HFSNumber100)
print('Number LFS particles shift 100 (down) = %.1f' % LFSNumber100)

plt.show()