from hexapole import HexVector, Assembly
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
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(name)s - %(name)s - %(message)s'))
logging.getLogger().addHandler(ch)

# Select the state to fly, and the start and end z-plane relative to the centre
# of the magnet.

start = 236.0
end = 268.0

# Set up the number of bins in the flight direction and image height.
nsteps = 100
nheight = 200
hmax = 4.0

h1 = HexVector('Bvec.h5', position=[0.0, 1.0, 241.0]) 
h2 = HexVector('Bvec.h5', position=[0.0, 0.5, 256.0]) 
#h3 = HexVector('Bvec.h5', position=[0.0, 0.0, 261.15]) 
hh = Assembly([h1, h2])


flightImage = np.zeros((nsteps, nheight))
plt.ion()

# Load some atoms.
pos, vel, times = loadFinal(r'../GenAlgSim/90deg', states=[0, 1])
pos, vel, times = rewind(235.0, pos, vel, times)

#Speed things up by only flying the slower particles.
ind = np.where(vel[:,2]<0.210)[0]
pos = pos[ind, :]
vel = vel[ind, :]
times = times[ind]


flightBins = np.linspace(-hmax, hmax, nheight+1)
steps = np.linspace(start, end, nsteps)

plt.clf()
# Step along flight direction, flying particles to each column of pixels.
for j, step in enumerate(steps):
    print '{}/{}'.format(j, len(steps))
    pos, vel, times = verletFlyer(pos, vel, times, state=0, 
            hexapole=hh, totalZ=step, dt=0.5, totalTime=100)

    # Pick out particles that have not collided.
    ind = hh.notCollided(pos)
    flightImage[j,:] = np.histogram(pos[ind,1], flightBins)[0]

plt.imshow(flightImage.T, origin='lower', cmap='viridis',
        extent=(start, end, -hmax, hmax))

for m in hh.magnetList:
    # Draw the outline of magnetlist:the magnets as rectangle, then transform
    # this into the lab frame for plotting.
    m1 = np.array([[0.0, -m.ri, -3.5], [0.0, -10.0, -3.5],
        [0, -10.0, 3.5], [0, -m.ri, 3.5]])
    m2 = np.array([[0.0,  m.ri, -3.5], [0.0,  10.0, -3.5],
        [0,  10.0, 3.5], [0,  m.ri, 3.5]])

    m1 = m.toLab(m1)
    m2 = m.toLab(m2)

    # Draw the magnets.
    plt.gca().add_patch(mpl.patches.Polygon(np.vstack((m1[:,2], m1[:,1])).T,
            fill=True, facecolor=u'#09B190'))
    plt.gca().add_patch(mpl.patches.Polygon(np.vstack((m2[:,2], m2[:,1])).T,
            fill=True, facecolor=u'#09B190'))

plt.axis('image')
plt.tight_layout()
plt.show()
