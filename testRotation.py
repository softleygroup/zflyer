from hexapole import HexArray
from Verlet import verletFlyer, loadFinal, rewind

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set up logging and message detail level. Set the level to logging.INFO for a
# quieter output.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Load some atoms
pos, vel, times = loadFinal('../Animate/GA', states=[0])

# Speed things up by only flying the slower particles.
ind = np.where(vel[:,2]<0.300)[0]
pos = pos[ind, :]
vel = vel[ind, :]
times = times[ind]

# Initialise the hexapole.
hh = HexArray('B.h5', position=[0.0, 0.0, 236.6], angle=[00.0/180*np.pi, 0.0, 0.0])

# Move atoms back to starting position and transform coordinates into the
# magnet frame of reference.
pos, vel, times = rewind(226.6, pos, vel, times)
pos, vel = hh.toMagnet(pos, vel)


# Fly the atoms

pos, vel, times = verletFlyer(pos, vel, times, state=0, 
        hexapole=hh, totalZ=13.0, dt=0.5, totalTime=500)

# Transform back into the lab frame and continue to the detection plane.
pos, vel = hh.toLab(pos, vel)
pos, vel, times = verletFlyer(pos, vel, times, state=0,
        hexapole=hh, totalZ=248.0, dt=0.5, totalTime=500)

# Histogram final positions.
ind = np.where(pos[:,2]>249.0)[0]
bins = np.linspace(-15, 15, 200)
n, _ = np.histogram(pos[ind, 1], bins)

plt.plot(bins[:-1], n)
