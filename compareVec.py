from hexapole import HexVector
from separator import Magnet

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

offset = 5.0

h1 = HexVector('Bvec.h5', position=[0.0, 0.0, -offset])
h2 = HexVector('Bvec.h5', position=[0.0, 1.75, offset])

zrange = 10.0
yrange = 5.0
gp = 10
z, y = np.meshgrid(np.linspace(-zrange, zrange, 2.0*zrange*gp), 
        np.linspace(-yrange, yrange, 2.0*yrange*gp))
gridshape = z.shape
z = z.flatten()
y = y.flatten()
x = np.zeros_like(z)
pos = np.vstack((x, y, z)).T
vel = np.zeros_like(pos)

pos, vel = h1.toMagnet(pos, vel)
b1 = h1.B(pos)
bv1 = h1.Bvec(pos)
pos, vel = h1.toLab(pos, vel)

pos, vel = h2.toMagnet(pos, vel)
b2 = h2.B(pos)
bv2 = h2.Bvec(pos)
pos, vel = h2.toLab(pos, vel)

bpot = b1 + b2
bpot = bpot.reshape(gridshape)
bv = bv1 + bv2
bvy = bv[:,1].reshape(gridshape)
bvz = bv[:,2].reshape(gridshape)

bvpot = np.sqrt(np.sum(bv**2, axis=1))
bvpot = bvpot.reshape(gridshape)

fig, ax = plt.subplots(2, 1, sharex=True)
im0 = ax[0].imshow(bpot, origin='lower', extent=[-zrange, zrange, -yrange, yrange],
        cmap='viridis', vmax=1.2)
divider0 = make_axes_locatable(ax[0])
cax0 = divider0.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im0, cax=cax0).set_label('B (T)')

#ax[0].quiver(z.reshape(gridshape)[::5,::5].flatten(),
        #y.reshape(gridshape)[::5,::5].flatten(), 
        #bvz[::5,::5].flatten(), bvy[::5,::5].flatten(),
        #scale=10)
ax[0].streamplot(z.reshape(gridshape), y.reshape(gridshape), bvz, bvy,
        color='grey', density=2)

im1 = ax[1].imshow((bpot-bvpot), origin='lower', extent=[-zrange, zrange, -yrange, yrange],
        cmap='viridis', vmax=0.05)
divider1 = make_axes_locatable(ax[1])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax1).set_label('Error in B (T)')

plt.tight_layout()
plt.show()

