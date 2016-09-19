""" Definition of a static-field magnetic hexapole. There are two classes
defined here: The first uses an analytical multipole expansion of the magnetic
field components, and the second interpolates from a pre-computed 3D grid of
magnetic field.
"""

import h5py
import logging
import numpy as np
#from scipy.interpolate import RegularGridInterpolator as Interpolator
from fastInterpolate import FastInterpolator as Interpolator

class Hexapole(object):
    """ Defines a single hexapole. The field is calculated analytically for
    positions relative to the centre of the magnet array. The position and
    angle are used in coordinate transform functions that should be applied to
    the position and velocity data first.

    The analytical expressions for the field components are a multipole
    expansion given by Ackermann and Weiland, with parameters fit to a
    numerical model of a 1.3 T magnet array with a 3 mm inner radius.
    """
    def __init__(self, position=[0.0, 0.0, 0.0], angle=None):
        """ Initialise the array. Stores the position and angle for later use,
        and defines parameters that have been fit to a numerical model.
        """
        self.LOG = logging.getLogger(type(self).__name__)
        self.LOG.debug('Initialised a hexapole')

        self.position = position
        self.angle = angle

        # Magnet parameters from Radia calculations.
        self.b3 = 0.86 # multipole expansion coefficient 3.
        self.b15 = -0.11 # Multipole expansion coefficient 15.
        self.d = 2.88 # mm (specific length)
        self.ri = 3.0 # mm (Inner radius)
        self.B0 = 1.3 # T (Remnance)
        self.t = 7.0  # mm (Thickness)
        self.a = self.d/np.sqrt(np.sqrt(2)-1)


    def Bvec(self, pos):
        """ Compute the magnetic field at position `pos` in the frame where
        the centre of the coil is at the origin. Use the `toMagnet` function to
        convert coordinates into the frame centred on this magnet.

        Parameters:
            x, y, z (1D np.Array)
                Vector of coordinates in the frame of reference of this magnet.
        Returns:
            B (1D np.Array)
                Vector of B at each set of coordinates.
        """

        r = np.sqrt(np.sum(pos[:,:2]**2, axis=1))
        phi = np.arctan2(pos[:,1], pos[:,0])

        A = 1.0/(1.0 + (pos[:,2]/self.a)**4)**2
        dAdz = -8.0 * pos[:,2]**3/(self.a**4 + (1.0 + (pos[:,2]/self.a)**4)**3)

        B_phi = A * self.B0 * (
                self.b3  * np.cos(3.0*phi)  * (r/self.ri)**2 +
                self.b15 * np.cos(15.0*phi) * (r/self.ri)**14)
        B_r   = A * self.B0 * (
                self.b3  * np.sin(3.0*phi)  * (r/self.ri)**2 +
                self.b15 * np.sin(15.0*phi) * (r/self.ri)**14)
        B_z   = dAdz * self.B0 * (
        #B_z   = A * self.B0 * (
                self.b3  * np.sin(3.0*phi)  * (r**3/(3.0 * r**2)) +
                self.b15 * np.sin(15.0*phi) * (r**15/(15.0 * r**14)))

        #ind = np.where(
            #(z<(self.position[2]+self.t/2)) &
            #(z>(self.position[2]-self.t/2)))[0]
        #ind = np.setdiff1d(np.arange(len(x)), ind)
        #B_phi[ind] = 0.0
        #B_r[ind] = 0.0
        #B_z[ind] = 0.0
        return B_phi, B_r, B_z


    def B(self, pos):
        """ Compute the magnetic potential at position `pos` in the frame where
        the centre of the coil is at the origin. Use the `toMagnet` function to
        convert coordinates into the frame centred on this magnet.

        Parameters:
            pos ((n,3) np.ndarray) (mm):
                Array of particle positions.
        Returns:
            B (1D np.Array)
                Vector of B at each set of coordinates.
        """
        x = pos[:,0]
        y = pos[:,1]
        z = pos[:,2]

        B_phi, B_r, B_z = self.Bvec(x, y, z)
        return np.sqrt(B_phi**2 + B_r**2 + B_z**2)


    def dB(self, pos, d=None):
        """ Compute the field gradient at the position listed in `x`, `y`, `z`
        in the frame where the centre of the coil is at the origin. The
        gradient is calculated using central differences. The default step size
        for the gradient is 1/1000 of the central radius of the magnets, or
        can be specified using the paramter `d`.

        Use the `toMagnet`
        function to convert coordinates into the frame centred on this magnet.

        Parameters:
            pos ((n,3) np.ndarray) (mm):
                Array of particle positions.
            d (optional, float) (mm)
                Step size to use in central differences approximation of the
                gradient.

        Returns:
            dBx, dBy, dBz (1D np.Array) (T/mm)
                Vector of derivative of B at each set of coordinates.
        """
        if d is None:
            d = 0.001 * self.ri

        x = pos[:,0]
        y = pos[:,1]
        z = pos[:,2]

        h = d/2.0
        dBx = (self.B(np.vstack((x+h, y, z)).T) 
                - self.B(np.vstack((x-h, y, z)).T))/d
        dBy = (self.B(np.vstack((x, y+h, z)).T) 
                - self.B(np.vstack((x, y-h, z)).T))/d
        dBz = (self.B(np.vstack((x, y, z+d)).T) 
                - self.B(np.vstack((x, y, z-d)).T))/d

        return np.vstack((dBx, dBy, dBz)).T


    @staticmethod
    def _xMatrix(theta):
        """ Generate the transformation matrix for rotation about x axis. """
        return np.array([
            [1,     0,              0],
            [0,     np.cos(theta),  -np.sin(theta)],
            [0,     np.sin(theta),  np.cos(theta)]
            ])


    @staticmethod
    def _yMatrix(theta):
        """ Generate the transformation matrix for rotation about y axis. """
        return np.array([
            [np.cos(theta),     0,  np.sin(theta)],
            [0,                 1,  0],
            [-np.sin(theta),    0,  np.cos(theta)]
            ])


    def toMagnet(self, pos, vel=None):
        """ Transform the set of lab-frame coordinates into the coordinate
        frame of this array. Positions are shifted then rotated, then velocity
        vectors are rotated.

        Parameters:
            pos ((n,3) np.ndarray) (mm):
                Array of particle positions.
            vel ((n,3) np.ndarray) (mm/us):
                Array of particle velocities.
        """

        #pos = np.atleast_2d(pos)
        # Move to magnet origin
        pos[:,0] -= self.position[0]
        pos[:,1] -= self.position[1]
        pos[:,2] -= self.position[2]

        if self.angle != None:
            # Generate rotation matrix
            rot = np.dot(self._yMatrix(self.angle[1]), self._xMatrix(self.angle[0]))
            # Take the dot product of each position and velocity with this matrix.
            pos[:] = np.einsum('jk,ik->ij', rot, pos)
            if vel != None:
                vel[:] = np.einsum('jk,ik->ij', rot, vel)

        if vel==None:
            return pos
        else:
            return pos, vel


    def toLab(self, pos, vel=None):
        """ Transform the set of coordinates from the magnet frame into the lab
        frame.

        Parameters:
            pos ((n,3) np.ndarray) (mm):
                Array of particle positions.
            vel ((n,3) np.ndarray) (mm/us):
                Array of particle velocities.
        """

        #pos = np.atleast_2d(pos)
        # Inplace modification
        pos[:,0] += self.position[0]
        pos[:,1] += self.position[1]
        pos[:,2] += self.position[2]

        if self.angle != None:
            # Generate rotation matrix
            rot = np.dot(self._yMatrix(-self.angle[1]), 
                    self._xMatrix(-self.angle[0]))
            # Take the dot product of each position and velocity with this matrix.
            pos[:] = np.einsum('jk,ik->ij', rot, pos)
            if vel != None:
                vel[:] = np.einsum('jk,ik->ij', rot, vel)


        if vel==None:
            return pos
        else:
            return pos, vel


    def collided(self, pos):
        """ Return a list of indicies for particles that have collided with the
        magnet, use to eliminate particles that have collided with the walls.

        Parameters:
            pos ((n,3) np.ndarray) (mm):
                Array of particle positions in the magnet frame.
        """

        #pos = np.atleast_2d(pos)
        r2 = np.sum(pos[:,0:2]**2, axis=1)
        return np.where((r2 >= self.ri**2) & 
            (pos[:,2]<+self.t/2) &
            (pos[:,2]>-self.t/2)
            )[0]


    def notCollided(self, pos):
        """ Return a list of indicies for particles that have not collided with
        the magnet, use to eliminate particles that have collided with the
        walls.

        Parameters:
            pos ((n,3) np.ndarray) (mm):
                Array of particle positions.
        """

        #pos = np.atleast_2d(pos)
        collided = self.collided(pos)
        return np.setdiff1d(np.arange(len(pos)), collided)


class HexArray(Hexapole):
    """ Holds a 3D grid of potentials calculated in Radia. Calculates B at a
    point by linear interpolation.

    The arrays are reflected in the x,y plane so that coordinates at (x, y, z)
    give the same field as (x, y, -z).
    """
    def __init__(self, gridFile, position=[0.0, 0.0, 0.0], angle=None):
        """ Initialise the interpolation arrays from the potential array stored
        in `greidFile`. Uses the `scipy` `Interpolator` for linear
        interpolation. The numerical derivative (`np.diff`) is also computed
        along each axis for the gradient function.
        """
        super(HexArray, self).__init__(position=position, angle=angle)
        self.LOG = logging.getLogger(type(self).__name__)
        # gridData = np.load(gridFile)
        gridData = h5py.File(gridFile, 'r')
        x = gridData['x'][:]
        y = gridData['y'][:]
        z = gridData['z'][:]
        self.ri = gridData['ri'][0] # mm (Inner radius)
        self.t = gridData['t'][0]  # mm (Thickness)
        pot = gridData['pot'][:]
        gridData.close()

        self._buildInterp(x, y, z, pot)


    def _buildInterp(self, x, y, z, pot):
        """ Private function to build interpolation arrays using potential
        array `pot`. Assumes that only the positive part of z is in the array,
        so reflects the array in the (x, y) plane.
        """
        self.xmin = x[0]
        self.xmax = x[-1]
        self.ymin = y[0]
        self.ymax = y[-1]
        self.zmin = -z[-1]
        self.zmax = z[-1]

        # Field in negative z direction. Reverse the order in this axis.
        potNeg = pot[...,-1:0:-1]
        # Concatenate positive and negative z direction arrays.
        _z = np.hstack((-z[-1:0:-1], z))
        _pot = np.dstack((potNeg, pot))

        self.bInterpolator = Interpolator((x, y, _z), _pot)

        # Build difference derivative arrays
        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        self.dz = z[1]-z[0]
        dbdx = np.diff(_pot, axis=0)/self.dx
        dbdy = np.diff(_pot, axis=1)/self.dy
        dbdz = np.diff(_pot, axis=2)/self.dz
        x_dbdx = x[:-1]+self.dx/2
        y_dbdy = y[:-1]+self.dy/2
        z_dbdz = _z[:-1]+self.dz/2

        self.dBdxInterp = Interpolator((x_dbdx, y, _z), dbdx)
        self.dBdyInterp = Interpolator((x, y_dbdy, _z), dbdy)
        self.dBdzInterp = Interpolator((x, y, z_dbdz), dbdz)


    def B(self, pos):
        """ Interpolate the magnetic potential at position `pos` in the frame
        where the centre of the coil is at the origin. Use the `toMagnet`
        function to convert coordinates into the frame centred on this magnet.

        Points outside the interpolation grid are ignored and give B=0.

        Parameters:
            pos ((n,3) np.ndarray) (mm):
                Array of particle positions.
        Returns:
            B (1D np.Array)
                Vector of B at each set of coordinates.
        """
        #pos = np.atleast_2d(pos)
        B = np.zeros(len(pos))
        # Pick only valid points within the interpolation array
        ind = np.where((pos[:,0] > self.xmin) & (pos[:,0] < self.xmax) &
                (pos[:,1] > self.ymin) & (pos[:,1] < self.ymax) &
                (pos[:,2] > self.zmin) & (pos[:,2] < self.zmax))

        B[ind] = self.bInterpolator(pos[ind])

        return B


    def Bvec(self, pos):
        """ Not implemented as we don't currently have arrays of B vectors.
        """
        raise NotImplementedError


    def dB(self, pos):
        """ Compute the field gradient at the position listed in the frame
        where the centre of the coil is at the origin. The gradient is
        interpolated from the three differences arrays, computed in the
        constructor.

        Use the `toMagnet`
        function to convert coordinates into the frame centred on this magnet.

        Parameters:
            pos ((n,3) np.ndarray) (mm):
                Array of particle positions.

        Returns:
            dBx, dBy, dBz (1D np.Array) (T/mm)
                Vector of derivative of B at each set of coordinates.
        """
        #pos = np.atleast_2d(pos)
        # Pick only valid points within the interpolation array
        ind = np.where(
            (pos[:,0] > self.xmin+self.dx) & (pos[:,0] < self.xmax-self.dx) &
            (pos[:,1] > self.ymin+self.dy) & (pos[:,1] < self.ymax-self.dy) &
            (pos[:,2] > self.zmin+self.dz) & (pos[:,2] < self.zmax-self.dz))

        dBdx = np.zeros(len(pos))
        dBdy = np.zeros(len(pos))
        dBdz = np.zeros(len(pos))
        dBdx[ind] = self.dBdxInterp(pos[ind])
        dBdy[ind] = self.dBdyInterp(pos[ind])
        dBdz[ind] = self.dBdzInterp(pos[ind])

        return np.vstack((dBdx, dBdy, dBdz)).T


class HexVector(HexArray):
    """ Holds a 3d grid of field vectors calculated in Radia. Calculates B and
    vector field at a point by linear interpolation.
    """
    def __init__(self, gridFile, position=[0.0, 0.0, 0.0], angle=None):
        """ Initialise the interpolation arrays from the potential array stored
        in `greidFile`. Uses the `scipy` `Interpolator` for linear
        interpolation. The numerical derivative (`np.diff`) is also computed
        along each axis for the gradient function.
        """
        #super(HexVector, self).__init__(position=position, angle=angle)
        self.position = position
        self.angle = angle
        self.LOG = logging.getLogger(type(self).__name__)
        gridData = h5py.File(gridFile, 'r')
        x = gridData['x'][:]
        y = gridData['y'][:]
        z = gridData['z'][:]
        self.ri = gridData['ri'][0] # mm (Inner radius)
        self.t = gridData['t'][0]  # mm (Thickness)
        field = gridData['field'][:]
        gridData.close()

        # Calculate field from potential
        pot = np.sqrt(np.sum(field**2, axis=3))
        self._buildInterp(x, y, z, pot)

        # Field in negative z direction. Reverses the order in this axis, and
        # reverses the sign of the z component.
        z = np.hstack((-z[-1:0:-1], z))
        fieldNeg = field[...,-1:0:-1,:]
        fieldNeg[...,2] = -fieldNeg[...,2]
        # Concatenate positive and negative z direction arrays.
        field = np.dstack((fieldNeg, field))

        self.bvecInterp = Interpolator((x, y, z), field)

        # Compute the partial derivative of the field vector.
        dBdx = np.diff(field, axis=0)/self.dx
        dBdy = np.diff(field, axis=1)/self.dy
        dBdz = np.diff(field, axis=2)/self.dz
        x_dbdx = x[:-1]+self.dx/2.0
        y_dbdy = y[:-1]+self.dy/2.0
        z_dbdz = z[:-1]+self.dz/2.0

        # Build an interpolator for each axis
        self.dBInterp = []

        self.dBInterp.append(Interpolator((x_dbdx, y, z), dBdx))
        self.dBInterp.append(Interpolator((x, y_dbdy, z), dBdy))
        self.dBInterp.append(Interpolator((x, y, z_dbdz),  dBdz))


    def Bvec(self, pos):
        """ Interpolate the magnetic field vector at each point in pos. Returns
        a list of 3D vectors.
        """
        #pos = np.atleast_2d(pos)
        B = np.zeros((len(pos), 3))

        # Pick only valid points within the interpolation array
        ind = np.where((pos[:,0] > self.xmin) & (pos[:,0] < self.xmax) &
                (pos[:,1] > self.ymin) & (pos[:,1] < self.ymax) &
                (pos[:,2] > self.zmin) & (pos[:,2] < self.zmax))

        B[ind, :] = self.bvecInterp(pos[ind])

        return B


    def dBvec(self, pos, axis):
        """ Interpolate the gradient of the vector field at each location in
        pos.
        """

        #pos = np.atleast_2d(pos)
        dBdAxis = np.zeros((len(pos), 3))
        # Pick only valid points within the interpolation array
        ind = np.where((pos[:,0] > (self.xmin+self.dx/2))
                & (pos[:,0] < self.xmax-self.dx/2)
                & (pos[:,1] > self.ymin+self.dy/2)
                & (pos[:,1] < self.ymax-self.dy/2)
                & (pos[:,2] > self.zmin+self.dz/2)
                & (pos[:,2] < self.zmax-self.dz/2))[0]

        dBdAxis[ind] = self.dBInterp[axis](pos[ind])

        return dBdAxis


class Assembly(Hexapole):
    """ Hold a list of hexapole elements specified by `HexVector` and return
    the total field correcly summed from all elements.

    Initialise with a list of `HexVector`, each at the correct location::

        h1 = HexVector('Bvec.h5', position=[0.0, 1.2, 241.15]) 
        h2 = HexVector('Bvec.h5', position=[0.0, -0.2,251.15]) 
        h3 = HexVector('Bvec.h5', position=[0.0, 0.0, 261.15]) 
        hh = Assembly([h1, h2, h3])

    """
    def __init__(self, magnetList):
        self.magnetList = magnetList


    def B(self, pos):
        """ Compute the magnetic field at positions in  `pos` from the norm of
        the vector field. Calls self.Bvec to sum the field of each magnet.

        Parameters:
            pos ((n,3) np.ndarray) (mm):
                Array of particle positions.
        Returns:
            B (1D np.Array)
                Vector of B at each set of coordinates.
        """
        bv = self.Bvec(pos)
        return np.sqrt(np.sum(bv**2, axis=1)) 


    def dB(self, pos):
        """ Compute the field gradient from the components of the vector field
        of each magnet (Bx, By, Bz).

               d|B|    1   /    d Bx      d By      d Bx \
               ---- = --- |  Bx ---- + By ---- + Bz ----  |
                dx     B   \     dx        dx        dx  /

        Parameters:
            pos ((n,3) np.ndarray) (mm):
                Array of particle positions.
        Returns:
            dB (1D np.Array)
                Vector of (dB/dx, dB/dy, dB/dz) at each set of coordinates.
        """
        #pos = np.atleast_2d(pos)

        # Get reciporcal of B, replacing infinities by zero.
        with np.errstate(divide='ignore'):
            B = self.B(pos)
            invb = 1.0/B
            invb[invb==np.inf] = 0

        Bv = self.Bvec(pos)

        dBvecdx = self.dBvec(pos, 0)
        dBvecdy = self.dBvec(pos, 1)
        dBvecdz = self.dBvec(pos, 2)

        dBdx = invb * (np.sum(Bv * dBvecdx, axis=1))
        dBdy = invb * (np.sum(Bv * dBvecdy, axis=1))
        dBdz = invb * (np.sum(Bv * dBvecdz, axis=1))

        return np.vstack((dBdx, dBdy, dBdz)).T


    def Bvec(self, pos):
        """ Sum the B vector of each magnet at a position.

        Parameters:
            pos ((n,3) np.ndarray) (mm):
                Array of particle positions.
        Returns:
            B (1D np.Array)
                Vector of B at each set of coordinates.
        """
        #pos = np.atleast_2d(pos)
        Bvec = np.zeros((len(pos), 3))

        for mag in self.magnetList:
            pos = mag.toMagnet(pos)
            Bvec += mag.Bvec(pos)
            pos = mag.toLab(pos)

        return Bvec


    def dBvec(self, pos, axis):
        """ Compute the change in the magnetic field vector with position along
        the given axis.

        Returns a vector (dBx/da, dBy/da, dBz,da) where a=x, y or z depending
        on choice given in `axis` parameter.

        Parameters:
            pos ((n,3) np.ndarray) (mm):
                Array of particle positions.
            axis (int):
                Axis to differentiate 0=x, 1=y, 2=z.
        Returns:
            B (1D np.Array)
                Vector (dBx/da, dBy/da, dBz,da) at each set of coordinates.
        """
        #pos = np.atleast_2d(pos)
        dBdaxis = np.zeros((len(pos), 3))

        for mag in self.magnetList:
            pos = mag.toMagnet(pos)
            dBdaxis += mag.dBvec(pos, axis)
            pos = mag.toLab(pos)

        return dBdaxis


    def collided(self, pos):
        """ Return a list of indicies for particles that have collided with the
        magnet, use to eliminate particles that have collided with the walls.

        Note: Calling the `notCollided` method uses the superclass definition,
        which calls this `collided` method.

        Parameters:
            pos ((n,3) np.ndarray) (mm):
                Array of particle positions in the magnet frame.
        """
        ind = []
        for mag in self.magnetList:
            pos = mag.toMagnet(pos)
            ind.append(mag.collided(pos))
            pos = mag.toLab(pos)

        return np.concatenate(ind)

