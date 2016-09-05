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

def PosVelPlots (pos_LFS, pos_HFS, vel_LFS, vel_HFS):
    
    f, ((ax1, ax2), (bx1, bx2)) = FigureSetup.new_figure(nrows=2, ncols=2, sharex='col', sharey='all')

    ax1.hexbin(pos_LFS[:,0], pos_LFS[:,1], gridsize=(300,300), bins='log', cmap='viridis')
    ax2.hexbin(pos_LFS[:,2], pos_LFS[:,1], gridsize=(300,300), bins='log', cmap='viridis')
    bx1.hexbin(pos_HFS[:,0], pos_HFS[:,1], gridsize=(300,300), bins='log', cmap='viridis')
    bx2.hexbin(pos_HFS[:,2], pos_HFS[:,1], gridsize=(300,300), bins='log', cmap='viridis')

    ax1.set_ylabel('LFS - pos(y)')
    bx1.set_ylabel('HFS - pos(y)')
    bx1.set_xlabel('pos(x)')
    bx2.set_xlabel('pos(z)')
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax2.set_xlim(250, 280)
    f.subplots_adjust(hspace=0.05, wspace=0.05)
    
    f, ((ax3, ax4), (bx3, bx4)) = FigureSetup.new_figure(nrows=2, ncols=2, sharex='col', sharey='all')

    ax3.hexbin(vel_LFS[:,0], vel_LFS[:,1], gridsize=(300,300), bins='log', cmap='viridis')
    ax4.hexbin(vel_LFS[:,2], vel_LFS[:,1], gridsize=(300,300), bins='log', cmap='viridis')
    bx3.hexbin(vel_HFS[:,0], vel_HFS[:,1], gridsize=(300,300), bins='log', cmap='viridis')
    bx4.hexbin(vel_HFS[:,2], vel_HFS[:,1], gridsize=(300,300), bins='log', cmap='viridis')

    ax3.set_ylabel('LFS - vel(y)')
    bx3.set_ylabel('HFS - vel(y)')
    bx3.set_xlabel('vel(x)')
    bx4.set_xlabel('vel(z)')
    ax3.set_xlim(-0.05, 0.05)
    ax3.set_ylim(-0.05, 0.05)
    ax4.set_xlim(0.15, 1.0)
    f.subplots_adjust(hspace=0.05, wspace=0.05)

detectorPos = 263.0

### NoHA

pos_NoHA = np.empty(shape=(0, 3), dtype=np.float64)
vel_NoHA = np.empty(shape=(0, 3), dtype=np.float64)
times_NoHA = np.empty(shape=(0, ), dtype=np.float64)

for states in [0,1,2,3]:
    # Load atoms
    pos, vel, times = loadFinal(r'C:\Users\tpsgroup\Google Drive\Oxford\Zeeman\Simulations\HA\Sim HA\90deg seq', states=[states])
    # Rewind back directly to detection position
    pos, vel, times = rewind(detectorPos, pos, vel, times)
    pos_NoHA = np.concatenate((pos_NoHA, pos))
    vel_NoHA = np.concatenate((vel_NoHA, vel))
    times_NoHA = np.concatenate((times_NoHA, times))
ind_vel_NoHA = np.where(vel_NoHA[:,2]<0.25)[0]


# # Load some atoms (LFS and HFS)
# pos_LFS, vel_LFS, times_LFS = loadFinal(r'C:\Users\tpsgroup\Google Drive\Oxford\Zeeman\Simulations\HA\Sim HA\90deg seq', states=[0])
# pos_HFS, vel_HFS, times_HFS = loadFinal(r'C:\Users\tpsgroup\Google Drive\Oxford\Zeeman\Simulations\HA\Sim HA\90deg seq', states=[2])
# # Rewind back directly to detection position (NoHA)
# pos_LFS, vel_LFS, times_LFS = rewind(detectorPos, pos_LFS, vel_LFS, times_LFS)
# pos_HFS, vel_HFS, times_HFS = rewind(detectorPos, pos_HFS, vel_HFS, times_HFS)

# PosVelPlots(pos_LFS, pos_HFS, vel_LFS, vel_HFS)

### HA

def Fly (shift, tilt):
    pos_coil12 = 226.6
    totalZ = 10.0 # where the Radia simulation ends and B=0 
    HAposition = [0.0, shift, 241.15]
    HAangle = [tilt/180*np.pi, 0.0, 0.0]
    pos_HA = np.empty(shape=(0, 3), dtype=np.float64)
    vel_HA = np.empty(shape=(0, 3), dtype=np.float64)
    times_HA = np.empty(shape=(0, ), dtype=np.float64)
    
    for states in [0,1,2,3]:
        # Load atoms
        pos, vel, times = loadFinal(r'C:\Users\tpsgroup\Google Drive\Oxford\Zeeman\Simulations\HA\Sim HA\90deg seq', states=[states])
        # Initialise the hexapole.
        hh = HexArray('B.npz', position=HAposition, angle=HAangle)
        # Move atoms back to starting position and transform coordinates into the magnet frame of reference.
        pos, vel, times = rewind(pos_coil12, pos, vel, times)
        pos, vel = hh.toMagnet(pos, vel)
        # Fly the atoms
        pos, vel, times = verletFlyer(pos, vel, times, state=states, hexapole=hh, totalZ=totalZ, dt=0.5, totalTime=500)
        # Pick out the particles that have not collided with the magnet
        ind_notcollided = hh.notCollided(pos)
        # Transform back into the lab frame and histogram detected positions.
        pos, vel = hh.toLab(pos, vel)
        # Fly them forward to the YAG
        pos, vel, times = rewind(detectorPos, pos, vel, times)
        
        pos_HA = np.concatenate((pos_HA, pos[ind_notcollided,:]))
        vel_HA = np.concatenate((vel_HA, vel[ind_notcollided,:]))
        times_HA = np.concatenate((times_HA, times[ind_notcollided]))
                        
    ind_vel = np.where(vel_HA[:,2]<0.25)[0]
    return float(len(ind_vel))/float(len(vel_HA))*100, float(len(ind_vel)), pos_HA, vel_HA, ind_vel
    

    
#### Density of slow particles (<250 ms-1) as a function of shift and tilt
shifts = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.50, 3.00, 3.50, 4.00, 4.50, 5.0, 5.50, 6.00])
tilts = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0])
purity = np.zeros((len(shifts), len(tilts)))
density = np.zeros((len(shifts), len(tilts)))

datafile = os.path.join(r'C:\Users\tpsgroup\Google Drive\Oxford\Zeeman\Simulations\HA\Sim HA\90deg seq','PurityDensityDataSaved.npz')
if os.path.exists(datafile):
    alldata = np.load(datafile)
    purity = alldata['purity'] 
    density = alldata['density']
else:
    for i, s in enumerate(shifts):
        for j, t in enumerate(tilts):
            purity[i,j], density[i,j], _, _, _ = Fly(s, t) 
            print i,j
    np.savez(datafile, purity=purity, density=density)

# f, ((ax1, ax2)) = FigureSetup.new_figure(nrows=1, ncols=2, sharex='all', sharey='all')
f, ax1 = FigureSetup.new_figure()
    
# ax1.imshow(purity, origin='lower', extent= [0.0,60.0,0.0,6.0], aspect='auto', cmap='viridis')
# ax2.imshow(density, origin='lower', extent= [0.0,60.0,0.0,6.0], aspect='auto', cmap='viridis')
ax1.imshow(density/np.max(density), origin='lower', extent= [0.0,60.0,0.0,6.0], aspect='auto', cmap='viridis')

ax1.set_xlabel('Tilt (deg)')
ax1.set_ylabel('Shift (mm)')
# ax2.set_xlabel('Tilt (deg)')
ax1.set_xlim(0,60.0)
ax1.set_ylim(0,6.0)
f.subplots_adjust(hspace=0.05, wspace=0.05)
plt.show()


    
    
    
# # Purity and Density of slow particles (<250 ms-1) as a function of shift
# shifts = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.0, 5.25, 5.50, 5.75, 6.00])
# purity = np.zeros(len(shifts))
# density = np.zeros(len(shifts))

# for i, s in enumerate(shifts):
    # purity[i], density[i], _ = Fly(s, 0.0) 

# fig, ax1 = plt.subplots()
    
# ax1.plot(shifts, purity, color='b')
# ax2 = ax1.twinx()
# ax2.plot(shifts, density/np.max(density), color='g')

# ax1.set_xlabel('Shift (mm)')
# ax1.set_ylabel('Percentage of slow particles', color='b')
# for tl in ax1.get_yticklabels():
    # tl.set_color('b')
# ax2.set_ylabel('Density of slow particles', color='g')
# for tl in ax2.get_yticklabels():
    # tl.set_color('g')
# ax1.set_ylim(0,12)
# ax2.set_ylim(0,1)


    
# # Purity and Density of slow particles (<250 ms-1) as a function of tilt
# tilts = np.array([0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0, 52.5, 55.0, 57.5, 60.0])
# puritytilt = np.zeros(len(tilts))
# densitytilt = np.zeros(len(tilts))

# for i, t in enumerate(tilts):
    # puritytilt[i], densitytilt[i], _, _, _ = Fly(2.0, t) 

# fig, ax1 = plt.subplots()
    
# ax1.plot(tilts, puritytilt, color='b')
# ax2 = ax1.twinx()
# ax2.plot(tilts, densitytilt/np.max(densitytilt), color='g')

# ax1.set_xlabel('Tilt (deg)')
# ax1.set_ylabel('Percentage of slow particles', color='b')
# for tl in ax1.get_yticklabels():
    # tl.set_color('b')
# ax2.set_ylabel('Density of slow particles', color='g')
# for tl in ax2.get_yticklabels():
    # tl.set_color('g')
# ax1.set_ylim(0,20)
# ax2.set_ylim(0,1)

# plt.show()



# ###xy pos plots for All, Slow and Slow/All for NoHA, HA and modified HA
##Straight HA
# _, _, pos_HA, vel_HA, ind_vel_HA = Fly(0.0, 0.0)
##Modified HA
# _, _, pos_mHA, vel_mHA, ind_vel_mHA = Fly(2.5, 0.0)

# f, ((ax1, ax2, ax3), (bx1, bx2, bx3)) = FigureSetup.new_figure(nrows=2, ncols=3, sharex='all', sharey='all')

# bins = np.linspace(-50,50,300)

# ###NoHA
# All_NoHA, _, _ = np.histogram2d(pos_NoHA[:,0], pos_NoHA[:,1], bins=bins)
# Slow_NoHA, _, _ = np.histogram2d(pos_NoHA[ind_vel_NoHA,0], pos_NoHA[ind_vel_NoHA,1], bins=bins)
# Ratio_NoHA = np.nan_to_num(Slow_NoHA/All_NoHA)

# maxInt_All_NoHA = np.max(All_NoHA)
# maxInt_Slow_NoHA = np.max(Slow_NoHA)

# # ax1.imshow(All_NoHA.T, vmax=maxInt_All_NoHA, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# # ax2.imshow(Slow_NoHA.T, vmax=maxInt_Slow_NoHA, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# # ax3.imshow(Ratio_NoHA.T, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# # ax1.set_ylabel('NoHA - $pos_y (mm)$')

# ###HA
# All_HA, _, _ = np.histogram2d(pos_HA[:,0], pos_HA[:,1], bins=bins)
# Slow_HA, _, _ = np.histogram2d(pos_HA[ind_vel_HA,0], pos_HA[ind_vel_HA,1], bins=bins)
# Ratio_HA = np.nan_to_num(Slow_HA/All_HA)

# bx1.imshow(All_HA.T, vmax=maxInt_All_NoHA, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# bx2.imshow(Slow_HA.T, vmax=maxInt_Slow_NoHA, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# bx3.imshow(Ratio_HA.T, origin='lower', extent= [-50,50,-50,50], cmap='viridis')

# ###modifiedHA
# All_mHA, _, _ = np.histogram2d(pos_mHA[:,0], pos_mHA[:,1], bins=bins)
# Slow_mHA, _, _ = np.histogram2d(pos_mHA[ind_vel_mHA,0], pos_mHA[ind_vel_mHA,1], bins=bins)
# Ratio_mHA = np.nan_to_num(Slow_mHA/All_mHA)

# ax1.imshow(All_mHA.T, vmax=maxInt_All_NoHA, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# ax2.imshow(Slow_mHA.T, vmax=maxInt_Slow_NoHA, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# ax3.imshow(Ratio_mHA.T, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# ax1.set_ylabel('mHA - $pos_y (mm)$')


# bx1.set_ylabel('HA - $pos_y (mm)$')
# bx1.set_xlabel('$pos_x (mm)$')
# bx2.set_xlabel('$pos_x (mm)$')
# bx3.set_xlabel('$pos_x (mm)$')
# ax1.set_xlim(-15, 15)
# ax1.set_ylim(-15, 15)
# f.subplots_adjust(hspace=0.05, wspace=0.05)
# plt.show()



# ####Summary images
# ##Straight HA
# _, _, pos_HA, vel_HA, ind_vel_HA = Fly(0.0, 0.0)
# ##Modified HA
# _, _, pos_s2HA, vel_s2HA, ind_vel_s2HA = Fly(2.0, 0.0)
# _, _, pos_s3HA, vel_s3HA, ind_vel_s3HA = Fly(3.0, 0.0)
# _, _, pos_t15HA, vel_t15HA, ind_vel_t15HA = Fly(0.0, 15.0)
# _, _, pos_t25HA, vel_t25HA, ind_vel_t25HA = Fly(0.0, 25.0)

# bins = 100
# ##NoHA
# y_NoHA = np.linspace(np.min(pos_NoHA[:,1]), np.max(pos_NoHA[:,1]),bins)
# h_All_NoHA, _ = np.histogram(pos_NoHA[:,1], bins=bins)
# h_Slow_NoHA, _ = np.histogram(pos_NoHA[ind_vel_NoHA,1], bins=bins)
# ##straight HA
# y_HA = np.linspace(np.min(pos_HA[:,1]), np.max(pos_HA[:,1]),bins)
# h_All_HA, _ = np.histogram(pos_HA[:,1], bins=bins)
# h_Slow_HA, _ = np.histogram(pos_HA[ind_vel_HA,1], bins=bins)
# ##Shift 2.0 mm
# y_s2HA = np.linspace(np.min(pos_s2HA[:,1]), np.max(pos_s2HA[:,1]),bins)
# h_All_s2HA, _ = np.histogram(pos_s2HA[:,1], bins=bins)
# h_Slow_s2HA, _ = np.histogram(pos_s2HA[ind_vel_s2HA,1], bins=bins)
# ##Shift 3.0 mm
# y_s3HA = np.linspace(np.min(pos_s3HA[:,1]), np.max(pos_s3HA[:,1]),bins)
# h_All_s3HA, _ = np.histogram(pos_s3HA[:,1], bins=bins)
# h_Slow_s3HA, _ = np.histogram(pos_s3HA[ind_vel_s3HA,1], bins=bins)
# ##Tilt 15deg
# y_t15HA = np.linspace(np.min(pos_t15HA[:,1]), np.max(pos_t15HA[:,1]),bins)
# h_All_t15HA, _ = np.histogram(pos_t15HA[:,1], bins=bins)
# h_Slow_t15HA, _ = np.histogram(pos_t15HA[ind_vel_t15HA,1], bins=bins)
# ##Tilt 25deg
# y_t25HA = np.linspace(np.min(pos_t25HA[:,1]), np.max(pos_t25HA[:,1]),bins)
# h_All_t25HA, _ = np.histogram(pos_t25HA[:,1], bins=bins)
# h_Slow_t25HA, _ = np.histogram(pos_t25HA[ind_vel_t25HA,1], bins=bins)


# ###Intensity profiles in y
# f, ((ax1, bx1), (ax2, bx2), (ax3, bx3), (ax4, bx4), (ax5, bx5), (ax6, bx6)) = FigureSetup.new_figure(nrows=6, ncols=2, sharex='col', sharey='all')

# colours = ['#ADD8E6', '#000080']# '#8B0000', '#114477', '#771122'

# ax1.plot(h_All_NoHA, y_NoHA, color=colours[0], label='All') #/float(np.max(h_All_NoHA))
# ax1.plot(h_Slow_NoHA, y_NoHA, color=colours[1], label='Slow')
# ax1.set_ylabel('NoHA - $pos_y (mm)$')

# ax2.plot(h_All_HA, y_HA, color=colours[0], label='All')
# ax2.plot(h_Slow_HA, y_HA, color=colours[1], label='Slow')
# ax2.set_ylabel('HA - $pos_y (mm)$')

# ax3.plot(h_All_s2HA, y_s2HA, color=colours[0], label='All')
# ax3.plot(h_Slow_s2HA, y_s2HA, color=colours[1], label='Slow')
# ax3.set_ylabel('Shift 2.0 mm')

# ax4.plot(h_All_s3HA, y_s3HA, color=colours[0], label='All')
# ax4.plot(h_Slow_s3HA, y_s3HA, color=colours[1], label='Slow')
# ax4.set_ylabel('Shift 3.0 mm')

# ax5.plot(h_All_t15HA, y_t15HA, color=colours[0], label='All')
# ax5.plot(h_Slow_t15HA, y_t15HA, color=colours[1], label='Slow')
# ax5.set_ylabel('Tilt 15 deg')

# ax6.plot(h_All_t25HA, y_t25HA, color=colours[0], label='All')
# ax6.plot(h_Slow_t25HA, y_t25HA, color=colours[1], label='Slow')
# ax6.set_ylabel('Tilt 25 deg')

# bx1.plot(h_All_NoHA/float(np.max(h_All_NoHA)), y_NoHA, color=colours[0], label='All') 
# bx1.plot(h_Slow_NoHA/float(np.max(h_Slow_NoHA)), y_NoHA, color=colours[1], label='Slow')

# bx2.plot(h_All_HA/float(np.max(h_All_HA)), y_HA, color=colours[0], label='All')
# bx2.plot(h_Slow_HA/float(np.max(h_Slow_HA)), y_HA, color=colours[1], label='Slow')

# bx3.plot(h_All_s2HA/float(np.max(h_All_s2HA)), y_s2HA, color=colours[0], label='All')
# bx3.plot(h_Slow_s2HA/float(np.max(h_Slow_s2HA)), y_s2HA, color=colours[1], label='Slow')

# bx4.plot(h_All_s3HA/float(np.max(h_All_s3HA)), y_s3HA, color=colours[0], label='All')
# bx4.plot(h_Slow_s3HA/float(np.max(h_Slow_s3HA)), y_s3HA, color=colours[1], label='Slow')

# bx5.plot(h_All_t15HA/float(np.max(h_All_t15HA)), y_t15HA, color=colours[0], label='All')
# bx5.plot(h_Slow_t15HA/float(np.max(h_Slow_t15HA)), y_t15HA, color=colours[1], label='Slow')

# bx6.plot(h_All_t25HA/float(np.max(h_All_t25HA)), y_t25HA, color=colours[0], label='All')
# bx6.plot(h_Slow_t25HA/float(np.max(h_Slow_t25HA)), y_t25HA, color=colours[1], label='Slow')


# ax6.set_xlabel('Num particles')
# bx6.set_xlabel('Normalized')
# # ax1.set_xlim(-18, 18)
# bx1.set_xlim(0, 1.1)
# ax1.set_ylim(-18, 18)
# f.subplots_adjust(hspace=0.05, wspace=0.05)
# plt.show()




###Imshow images xy pos
# f, ((ax1, ax2, ax3), (bx1, bx2, bx3), (cx1, cx2, cx3), (dx1, dx2, dx3), (ex1, ex2, ex3), (fx1, fx2, fx3)) = FigureSetup.new_figure(nrows=6, ncols=3, sharex='all', sharey='all')

# bins = np.linspace(-50,50,300)

# ###NoHA
# All_NoHA, _, _ = np.histogram2d(pos_NoHA[:,0], pos_NoHA[:,1], bins=bins)
# Slow_NoHA, _, _ = np.histogram2d(pos_NoHA[ind_vel_NoHA,0], pos_NoHA[ind_vel_NoHA,1], bins=bins)
# Ratio_NoHA = np.nan_to_num(Slow_NoHA/All_NoHA)

# maxInt_All_NoHA = np.max(All_NoHA)
# maxInt_Slow_NoHA = np.max(Slow_NoHA)

# ax1.imshow(All_NoHA.T, vmax=maxInt_All_NoHA, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# ax2.imshow(Slow_NoHA.T, vmax=maxInt_Slow_NoHA, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# ax3.imshow(Ratio_NoHA.T, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# ax1.set_ylabel('NoHA - $pos_y (mm)$')

# ###HA
# All_HA, _, _ = np.histogram2d(pos_HA[:,0], pos_HA[:,1], bins=bins)
# Slow_HA, _, _ = np.histogram2d(pos_HA[ind_vel_HA,0], pos_HA[ind_vel_HA,1], bins=bins)
# Ratio_HA = np.nan_to_num(Slow_HA/All_HA)

# bx1.imshow(All_HA.T, vmax=maxInt_All_NoHA, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# bx2.imshow(Slow_HA.T, vmax=maxInt_Slow_NoHA, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# bx3.imshow(Ratio_HA.T, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# bx1.set_ylabel('HA - $pos_y (mm)$')

# ###modifiedHA
# ##Shift 2mm
# All_s2HA, _, _ = np.histogram2d(pos_s2HA[:,0], pos_s2HA[:,1], bins=bins)
# Slow_s2HA, _, _ = np.histogram2d(pos_s2HA[ind_vel_s2HA,0], pos_s2HA[ind_vel_s2HA,1], bins=bins)
# Ratio_s2HA = np.nan_to_num(Slow_s2HA/All_s2HA)

# cx1.imshow(All_s2HA.T, vmax=maxInt_All_NoHA, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# cx2.imshow(Slow_s2HA.T, vmax=maxInt_Slow_NoHA, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# cx3.imshow(Ratio_s2HA.T, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# cx1.set_ylabel('HA shift 2.0mm')

# ##Shift 3mm
# All_s3HA, _, _ = np.histogram2d(pos_s3HA[:,0], pos_s3HA[:,1], bins=bins)
# Slow_s3HA, _, _ = np.histogram2d(pos_s3HA[ind_vel_s3HA,0], pos_s3HA[ind_vel_s3HA,1], bins=bins)
# Ratio_s3HA = np.nan_to_num(Slow_s3HA/All_s3HA)

# dx1.imshow(All_s3HA.T, vmax=maxInt_All_NoHA, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# dx2.imshow(Slow_s3HA.T, vmax=maxInt_Slow_NoHA, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# dx3.imshow(Ratio_s3HA.T, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# dx1.set_ylabel('HA shift 3.0mm')

# ##Tilt 15deg
# All_t15HA, _, _ = np.histogram2d(pos_t15HA[:,0], pos_t15HA[:,1], bins=bins)
# Slow_t15HA, _, _ = np.histogram2d(pos_t15HA[ind_vel_t15HA,0], pos_t15HA[ind_vel_t15HA,1], bins=bins)
# Ratio_t15HA = np.nan_to_num(Slow_t15HA/All_t15HA)

# ex1.imshow(All_t15HA.T, vmax=maxInt_All_NoHA, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# ex2.imshow(Slow_t15HA.T, vmax=maxInt_Slow_NoHA, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# ex3.imshow(Ratio_t15HA.T, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# ex1.set_ylabel('HA tilt 15deg')

# ##Tilt 25deg
# All_t25HA, _, _ = np.histogram2d(pos_t25HA[:,0], pos_t25HA[:,1], bins=bins)
# Slow_t25HA, _, _ = np.histogram2d(pos_t25HA[ind_vel_t25HA,0], pos_t25HA[ind_vel_t25HA,1], bins=bins)
# Ratio_t25HA = np.nan_to_num(Slow_t25HA/All_t25HA)

# fx1.imshow(All_t25HA.T, vmax=maxInt_All_NoHA, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# fx2.imshow(Slow_t25HA.T, vmax=maxInt_Slow_NoHA, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# fx3.imshow(Ratio_t25HA.T, origin='lower', extent= [-50,50,-50,50], cmap='viridis')
# fx1.set_ylabel('HA tilt 25deg')


# fx1.set_xlabel('$pos_x (mm)$')
# fx2.set_xlabel('$pos_x (mm)$')
# fx3.set_xlabel('$pos_x (mm)$')
# ax1.set_xlim(-18, 18)
# ax1.set_ylim(-18, 18)
# f.subplots_adjust(hspace=0.05, wspace=0.05)
# plt.show()