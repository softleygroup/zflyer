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
logger.setLevel(logging.DEBUG)

# Select the state to fly, and the start and end z-plane relative to the centre
# of the magnet.
state = 3
start = -8.4
end = 15.0
hh = HexArray('B.npz', position=[0.0, 0.0, 235.0], 
        angle=[20.0/180 * np.pi, 0.0, 0.0])

# Load some atoms.
pos, vel, times = loadFinal('/Users/chris/Documents/Work/Experiments/Zeeman/Animate/GA', states=[state])
pos, vel, times = rewind(226.6, pos, vel, times)

# Speed things up by only flying the slower particles.
ind = np.where(vel[:,2]<0.300)[0]
pos = pos[ind, :]
vel = vel[ind, :]
times = times[ind]

# Set up the number of bins in the flight direction and image height.
nsteps = 200
nheight = 200
hmax = 4.0

flightImage = np.zeros((nsteps, nheight))
flightBins = np.linspace(-hmax, hmax, nheight+1)
steps = np.linspace(start, end, nsteps)

# Step along flight direction, flying particles to each column of pixels.
for i, step in enumerate(steps):
    pos, vel = hh.toMagnet(pos, vel)
    pos, vel, times = verletFlyer(pos, vel, times, state=state, 
            hexapole=hh, totalZ=step, dt=0.5, totalTime=100)

    # Pick out particles that have not collided, transform back into the lab
    # frame and make the column of pixels.
    ind = hh.notCollided(pos)
    pos, vel = hh.toLab(pos, vel)
    flightImage[i,:] = np.histogram(pos[ind,1], flightBins)[0]

plt.imshow(flightImage.T, origin='lower', cmap='viridis',
        extent=(hh.position[2]+start, hh.position[2]+end,-hmax, hmax))

# Draw the outline of the magnets as rectangle, then transform this into the
# lab frame for plotting.
m1 = np.array([[0.0, -3.0, -3.5], [0.0, -10.0, -3.5], [0, -10.0, 3.5], [0, -3, 3.5]])
m2 = np.array([[0.0,  3.0, -3.5], [0.0,  10.0, -3.5], [0,  10.0, 3.5], [0,  3, 3.5]])
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
