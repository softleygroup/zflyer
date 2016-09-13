import h5py
import logging
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from hexapole import HexVector

class Magnet(HexVector):
    """ Store and interpolate magnetic vector field. Adds functionality to
    compute the components of the vector field gradient at each location.
    """

    def __init__(self, gridFile):
        super(Magnet, self).__init__(gridFile, 
                position=[0, 0, 0], angle=[0, 0, 0])

        gridData = h5py.File(gridFile, 'r')
        x = gridData['x'][:]
        y = gridData['y'][:]
        z = gridData['z'][:]
        field = gridData['field'][:]
        gridData.close()

        # Field in negative z direction. Reverses the order in this axis, and
        # reverses the sign of the z component.
        fieldNeg = field[...,-1::-1,:]
        fidldNeg[...,2] = -fieldNeg[...,2]

        # Concatenate positive and negative z direction arrays.
        z = np.hstack((-z[-1:0:-1], z))
        field = np.dstack((fieldNeg, field))

        # Compute the partial derivative of the field vector.
        dBdx = np.diff(field, axis=0)/self.dx
        dBdy = np.diff(field, axis=1)/self.dy
        dBdz = np.diff(field, axis=2)/self.dz
        x_dbdx = x[:-1]+self.dx/2
        y_dbdy = y[:-1]+self.dy/2
        z_dbdz = _z[:-1]+self.dz/2

        # Build an interpolator for each axis
        self.dBInterp = []

        self.dBInterp.append(RegularGridInterpolator((x_dbdx, y, _z), dBdx))
        self.dBInterp.append(RegularGridInterpolator((x, y_dbdy, _z), dBdy))
        self.dBInterp.append(RegularGridInterpolator((x, y, z_dbdz),  dBdz))


    def dBvec(self, pos, axis):
        """ Interpolate the gradient of the vector field at each location in
        pos.
        """

        pos = np.atleast_2d(pos)
        dBdaxis = np.zeros(len(pos))
        # Pick only valid points within the interpolation array
        ind = np.where((pos[:,0] > self.xmin) & (pos[:,0] < self.xmax) &
                (pos[:,1] > self.ymin) & (pos[:,1] < self.ymax) &
                (pos[:,2] > self.zmin) & (pos[:,2] < self.zmax))

        dBdAxis[ind] = self.dBinterp[axis](pos[ind])

        return dBdAxis

