from hexapole import HexArray
from Verlet import verletFlyer, loadFinal, rewind
from IntScanAnalysis import plotProfile

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import FigureSetup

# Set up logging and message detail level. Set the level to logging.INFO for a
# quieter output.
logger = logging.getLogger()
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(name)s - %(name)s - %(message)s'))
logging.getLogger().addHandler(ch)

# Initialise the hexapole.
hh = HexArray('B.npz', position=[0.0,0.0,241.15], angle=[0.0,0.0,0.0])

# Load some atoms
tofbins = np.linspace(0,1000, 500)
tof = np.zeros((len(tofbins)-1, 4))
tofNoHA = np.zeros((len(tofbins)-1, 4))

#Loop for atoms going though the HA
for i in [0,1,2,3]:
    pos, vel, times = loadFinal(r'C:\Users\tpsgroup\Google Drive\Oxford\Zeeman\Simulations\HA\Sim HA\90deg seq', states=[i])

    ##Speed things up by only flying the slower particles.
    # ind = np.where(vel[:,2]<0.300)[0]
    # pos = pos[ind, :]
    # vel = vel[ind, :]
    # times = times[ind]

    # Move atoms back to starting position and transform coordinates into the
    # magnet frame of reference.
    pos, vel, times = rewind(226.6, pos, vel, times)
    pos, vel = hh.toMagnet(pos, vel)

    # Fly the atoms
    pos, vel, times = verletFlyer(pos, vel, times, state=0, 
            hexapole=hh, totalZ=21.85, dt=0.5, totalTime=500)

    # Transform back into the lab frame and histogram detected positions.
    pos, vel = hh.toLab(pos, vel)

    ind = np.where(pos[:,2]>263.0)[0]
    bins = np.linspace(-15, 15, 200)
    n, _ = np.histogram((pos[ind, 1]), bins)
    tof[:,i], _ = np.histogram(times[ind],tofbins)
    
#Loop for atoms NOT going through the HA
for i in [0,1,2,3]:
    pos, vel, times = loadFinal(r'C:\Users\tpsgroup\Google Drive\Oxford\Zeeman\Simulations\HA\Sim HA\90deg seq', states=[i])

    #Rewind back directly to detection position
    pos, vel, times = rewind(263.0, pos, vel, times)

    bins = np.linspace(-15, 15, 200)
    n, _ = np.histogram((pos[:, 1]), bins)
    tofNoHA[:,i], _ = np.histogram(times[ind],tofbins)
    
# TOF trace passing through HA
f = FigureSetup.new_figure()
    
#Simulations actual TOF without shift applied
# plt.plot(tofbins[:-1], tof[:,0], 'b', label='Unshifted - LFS') # LFS states
# plt.plot(tofbins[:-1], tof[:,2], 'c', label='Unshifted - HFS') # HFS states
plt.plot(tofbins[:-1], np.sum(tof[:,:]/np.max(np.sum(tof[:,:], axis=1)), axis=1), 'r', label='Unshifted - All') #all states

#Simulations TOF with 60mus shift as in SimExpCf to match with experimental traces
plt.plot(tofbins[:-1]+60, tof[:,0], 'b', label='HA LFS')#/np.max(np.sum(tof[:,:], axis=1)), 'b', label='HA LFS') # LFS states
plt.plot(tofbins[:-1]+60, tof[:,2], 'g', label='HA HFS') #/np.max(np.sum(tof[:,:], axis=1)), 'g', label='HA HFS') # HFS states
plt.plot(tofbins[:-1]+60, np.sum(tof[:,:], axis=1),'k', label='HA All')#/np.max(np.sum(tof[:,:], axis=1)), axis=1), 'k', label='HA All') #all states

plt.xlim(300, 1000)
#plt.ylim(0, 1.1)
plt.ylabel('Normalised signal')
plt.xlabel('TOF (mus)')
plt.legend(loc=1)

# TOF trace without passing through HA
#f = FigureSetup.new_figure()
    
#Simulations actual TOF without shift applied
# plt.plot(tofbins[:-1], tof[:,0], 'b', label='Unshifted - LFS') # LFS states
# plt.plot(tofbins[:-1], tof[:,2], 'c', label='Unshifted - HFS') # HFS states
#plt.plot(tofbins[:-1], np.sum(tofNoHA[:,:]/np.max(np.sum(tofNoHA[:,:], axis=1)), axis=1), 'r', label='Unshifted - All') #all states

#Simulations TOF with 60mus shift as in SimExpCf to match with experimental traces
plt.plot(tofbins[:-1]+60, tofNoHA[:,0],'c', label='NoHA LFS')#/np.max(np.sum(tofNoHA[:,:], axis=1)), 'c', label='NoHA LFS') # LFS states
plt.plot(tofbins[:-1]+60, tofNoHA[:,2], 'y', label='NoHA HFS')#/np.max(np.sum(tofNoHA[:,:], axis=1)), 'y', label='NoHA HFS') # HFS states
plt.plot(tofbins[:-1]+60, np.sum(tofNoHA[:,:], axis=1), 'grey', label='NoHA All')#/np.max(np.sum(tofNoHA[:,:], axis=1)), axis=1), 'grey', label='NoHA All') #all states

plt.xlim(300, 1000)
#plt.ylim(0, 1.1)
plt.ylabel('Normalised signal')
plt.xlabel('TOF (mus)')
plt.legend(loc=1)
plt.show()

