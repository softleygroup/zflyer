from sim_zeeman import ZeemanFlyer
import numpy as np
from matplotlib import pyplot as plt

import ctypes
from ctypes import c_double, c_uint, c_int 		# shorthand form for data types
c_double_p = ctypes.POINTER(c_double)			# pointer type


# optimised radius
ot1 = np.array([163.60, 181.33, 185.68, 269.10, 291.50, 281.92, 301.30, 367.72, 394.36, 391.61, 446.75, 476.21])
d1 = np.array([ 76.80, 70.15, 84.41, 84.99, 84.93, 81.15, 77.37, 84.38, 77.73, 84.99, 80.56, 67.96])
oft1 = ot1 + d1
# optimised atom number
ot2 = np.array([150.30, 180.20, 215.07, 265.56, 296.53, 277.93, 310.70, 375.66, 402.51, 394.10, 449.47, 476.20])
d2 = np.array([74.59, 69.06, 84.23, 84.97, 84.97, 80.76, 78.69, 84.51, 74.30, 84.96, 79.66, 69.41])
oft2 = ot2 + d2
currents = 12*[243.]


folder = 'opt_radius/'
flyer = ZeemanFlyer()
flyer.loadParameters(folder)
flyer.addParticles(checkSkimmer=True)
flyer.calculateCoilSwitching()
flyer.loadBFields()



flyer.prop.overwriteCoils(ot1.ctypes.data_as(c_double_p), oft1.ctypes.data_as(c_double_p))
flyer.preparePropagation(currents)

pos1, vel1 = flyer.getTimeDependence(100)

flyer.calculateCoilSwitching()
flyer.preparePropagation()
pos1f, vel1f, _ = flyer.propagate(0)


flyer.prop.overwriteCoils(ot2.ctypes.data_as(c_double_p), oft2.ctypes.data_as(c_double_p))
flyer.preparePropagation(currents)

pos2, vel2 = flyer.getTimeDependence(100)

flyer.calculateCoilSwitching()
flyer.preparePropagation()
pos2f, vel2f, _ = flyer.propagate(0)


flyer.propagationProps['phase'] = 60
flyer.calculateCoilSwitching()

pos3, vel3 = flyer.getTimeDependence(100)

flyer.calculateCoilSwitching()
flyer.preparePropagation()
pos3f, vel3f, _ = flyer.propagate(0)


z1 = pos1[:, :, 2]
r1 = np.sqrt(pos1[:, :, 0]**2 + pos1[:, :, 1]**2)
z2 = pos2[:, :, 2]
r2 = np.sqrt(pos2[:, :, 0]**2 + pos2[:, :, 1]**2)
z3 = pos3[:, :, 2]
r3 = np.sqrt(pos3[:, :, 0]**2 + pos3[:, :, 1]**2)


ind1 = np.where(pos1f[:, 2] > 268.)
ind2 = np.where(pos2f[:, 2] > 268.)
ind3 = np.where(pos3f[:, 2] > 268.)
ind1g = np.where((pos1f[:, 2] > 268.) & (vel1f[:, 2] > 0.9*.288) & (vel1f[:, 2] < 1.1*.288))
ind2g = np.where((pos2f[:, 2] > 268.) & (vel2f[:, 2] > 0.9*.288) & (vel2f[:, 2] < 1.1*.288))
ind3g = np.where((pos3f[:, 2] > 268.) & (vel3f[:, 2] > 0.9*.288) & (vel3f[:, 2] < 1.1*.288))

# plt.figure()
# plt.hexbin(z1[:, ind1].flat, r1[:, ind1].flat, bins='log')
# plt.xlim([0, 267.])
# plt.ylim([0, 5.])
# plt.title('optimised')
# plt.figure()
# plt.hexbin(z2[:, ind2].flat, r2[:, ind2].flat, bins='log')
# plt.xlim([0, 267.])
# plt.ylim([0, 5.])
# plt.title('default')


plt.figure()
plt.hist(pos1f[ind1, 0].flat, histtype='step', bins=100, label='optimised radius', normed=True)
plt.hist(pos2f[ind2, 0].flat, histtype='step', bins=100, label='optimised atom number', normed=True)
plt.hist(pos3f[ind3, 0].flat, histtype='step', bins=100, label='default', normed=True)
plt.legend()

vars1 = []
vars2 = []
vars3 = []
vars1g = []
vars2g = []
vars3g = []
zs1 = z1[:, ind1]
rs1 = r1[:, ind1]
zs2 = z2[:, ind2]
rs2 = r2[:, ind2]
zs3 = z3[:, ind3]
rs3 = r3[:, ind3]
zs1g = z1[:, ind1g]
rs1g = r1[:, ind1g]
zs2g = z2[:, ind2g]
rs2g = r2[:, ind2g]
zs3g = z2[:, ind3g]
rs3g = r2[:, ind3g]

for x in np.arange(267.):
	ind = np.where((zs1 >= x) & (zs1 < x+1))
	vars1.append(rs1[ind].flatten().mean())
	ind = np.where((zs2 >= x) & (zs2 < x+1))
	vars2.append(rs2[ind].flatten().mean())
	ind = np.where((zs3 >= x) & (zs3 < x+1))
	vars3.append(rs3[ind].flatten().mean())
	ind = np.where((zs1g >= x) & (zs1g < x+1))
	vars1g.append(rs1g[ind].flatten().mean())
	ind = np.where((zs2g >= x) & (zs2g < x+1))
	vars2g.append(rs2g[ind].flatten().mean())
	ind = np.where((zs3g >= x) & (zs3g < x+1))
	vars3g.append(rs3g[ind].flatten().mean())

plt.figure()
plt.plot(vars1, 'b', label='optimised radius, all particles')
plt.plot(vars1g, 'g', label='optimised radius, only particles w/ target velocity')
plt.plot(vars2, 'y', label='optimised atom #, all particles')
plt.plot(vars2g, 'k', label='optimised atom #, only particles w/ target velocity')
plt.plot(vars3, 'r', label='default, all particles')
# plt.plot(vars2g, 'r', label='default, target v')

plt.xlabel('z-position')
plt.ylabel('mean radial position')
plt.legend(loc=2)
plt.show()
