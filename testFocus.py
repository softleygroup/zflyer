from hexapole import HexArray
from Verlet import verletFlyer, loadFinal, rewind

import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl

# Test the hexapole flying code by generating a streak image of the decelerated
# H atom density through the tilted magnet.

# Set up logging and message detail level. Set the level to logging.INFO for a
# quieter output.
logger = logging.getLogger()
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(name)s - %(name)s - %(message)s'))
logging.getLogger().addHandler(ch)

# Select the state to fly, and the start and end z-plane relative to the centre
# of the magnet.

start = -8.4
end = 15.0

# Set up the number of bins in the flight direction and image height.
nsteps = 200
nheight = 200
hmax = 4.0

hh = HexArray('B.h5', position=[0.0, 0.0, 241.15], 
        angle=[0.0/180 * np.pi, 0.0, 0.0])

flightImage = np.zeros((4, nsteps, nheight))
        
for i in [0,1,2,3]:

    # Load some atoms.
    pos, vel, times = loadFinal(r'C:\Users\tpsgroup\Google Drive\Oxford\Zeeman\Simulations\HA\Sim HA\90deg seq', states=[i])
    pos, vel, times = rewind(226.6, pos, vel, times)

    # Speed things up by only flying the slower particles.
    # ind = np.where(vel[:,2]<0.300)[0]
    # pos = pos[ind, :]
    # vel = vel[ind, :]
    # times = times[ind]


    flightBins = np.linspace(-hmax, hmax, nheight+1)
    steps = np.linspace(start, end, nsteps)

    # Step along flight direction, flying particles to each column of pixels.
    for j, step in enumerate(steps):
        pos, vel = hh.toMagnet(pos, vel)
        pos, vel, times = verletFlyer(pos, vel, times, state=i, 
                hexapole=hh, totalZ=step, dt=0.5, totalTime=100)

        # Pick out particles that have not collided, transform back into the lab
        # frame and make the column of pixels.
        ind = hh.notCollided(pos)
        pos, vel = hh.toLab(pos, vel)
        flightImage[i,j,:] = np.histogram(pos[ind,1], flightBins)[0]

# flightImage[:,:] for sum of all quantum states, flightImage[0:2,:] for LFS only and flightImage[2:4,:] for HFS only
plt.imshow(np.sum(flightImage[:,:], axis=0).T, origin='lower', cmap='viridis',
        extent=(hh.position[2]+start, hh.position[2]+end,-hmax, hmax))

# Draw the outline of the magnets as rectangle, then transform this into the
# lab frame for plotting.
m1 = np.array([[0.0, -hh.ri, -3.5], [0.0, -10.0, -3.5], [0, -10.0, 3.5], [0, -hh.ri, 3.5]])
m2 = np.array([[0.0,  hh.ri, -3.5], [0.0,  10.0, -3.5], [0,  10.0, 3.5], [0,  hh.ri, 3.5]])
vv = np.zeros_like(m1)

m1, vv = hh.toLab(m1, vv)
m2, vv = hh.toLab(m2, vv)

# Draw the magnets.
plt.gca().add_patch(mpl.patches.Polygon(np.vstack((m1[:,2], m1[:,1])).T,
        fill=True, facecolor=u'#09B190'))
plt.gca().add_patch(mpl.patches.Polygon(np.vstack((m2[:,2], m2[:,1])).T,
        fill=True, facecolor=u'#09B190'))

plt.axis('image')
plt.axis((hh.position[2]+start, hh.position[2]+end, -5, 5))
plt.show()
