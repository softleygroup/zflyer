""" Implement a faster 3D interpolator that uses the properties of our data to
take some short cuts. The x, y, z coordinates are evenly spaced, so array
indicies are calculated directly.
"""

import numpy as np

class FastInterpolator(object):
    """ Rapidly interpolate on a regularly-spaced, 3D, cartesian grid.
    """
    def __init__(self, (x, y, z), data):
        """ Set up the interpolation.
            Assumptions:
            ------------
                x, y, z are equally spaced and in incresing order.
        """ 
        self.x0 = x[0]
        self.y0 = y[0]
        self.z0 = z[0]
        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        self.dz = z[1]-z[0]
        self.nx = len(x)
        self.ny = len(y)
        self.nz = len(z)

        self.data = np.copy(data)

        # Choose the interpolation method based on whether the data to
        # interpolate is a set of vectors or a set of scalar values.
        if len(self.data.shape) == 3:
            self._interpolate = self._scalar
        else:
            self._interpolate = self._vector


    def __call__(self, pos):
        return self._interpolate(pos)


    def _scalar(self, pos):
        """ interpolate a scalar from the data array at the set of positions in
        pos.

        assumptions
        -----------
            1) all entries in pos are in-bound (we'll get the standard numpy
               array out of bounds otherwise.

        """

        # fractional potential array index.
        ixf = (pos[:,0]-self.x0)/self.dx #- 1
        iyf = (pos[:,1]-self.y0)/self.dy #- 1
        izf = (pos[:,2]-self.z0)/self.dz #- 1

        # integer part of potential array index.
        ix = np.floor(ixf).astype(np.int)
        iy = np.floor(iyf).astype(np.int)
        iz = np.floor(izf).astype(np.int)

        # calculate distance of point from gridlines.
        xd = (ixf - ix)
        yd = (iyf - iy)
        zd = (izf - iz)

        # clamp out of range indicies to edges of array
        ix[ix<0] = 0
        iy[iy<0] = 0
        iz[iz<0] = 0
        ix[ix>self.nx] = self.nx
        iy[iy>self.ny] = self.ny
        iz[iz>self.nz] = self.nz

        q111 = self.data[ix    , iy    , iz  ]
        q112 = self.data[ix    , iy    , iz+1]
        q121 = self.data[ix    , iy+1, iz    ]
        q122 = self.data[ix    , iy+1, iz+1  ]
        q211 = self.data[ix+1, iy      , iz  ]
        q212 = self.data[ix+1, iy      , iz+1]
        q221 = self.data[ix+1, iy+1, iz      ]
        q222 = self.data[ix+1, iy+1, iz+1    ]

        i1 = (xd*q211 + (1-xd)*q111)
        i2 = (xd*q221 + (1-xd)*q121)
        j1 = (xd*q212 + (1-xd)*q112)
        j2 = (xd*q222 + (1-xd)*q122)

        k1 = (yd*i2 + (1-yd)*i1)
        k2 = (yd*j2 + (1-yd)*j1)

        return (zd*k2 + (1-zd)*k1)


    def _vector(self, pos):
        """ interpolate a scalar from the data array at the set of positions in
        pos.

        assumptions
        -----------
            1) all entries in pos are in-bound (we'll get the standard numpy
               array out of bounds otherwise.

        """

        # fractional potential array index.
        ixf = (pos[:,0]-self.x0)/self.dx #- 1
        iyf = (pos[:,1]-self.y0)/self.dy #- 1
        izf = (pos[:,2]-self.z0)/self.dz #- 1

        # integer part of potential array index.
        ix = np.floor(ixf).astype(np.int)
        iy = np.floor(iyf).astype(np.int)
        iz = np.floor(izf).astype(np.int)

        # calculate distance of point from gridlines.
        xd = np.tile((ixf - ix), (3, 1)).T
        yd = np.tile((iyf - iy), (3, 1)).T
        zd = np.tile((izf - iz), (3, 1)).T

        # clamp out of range indicies to edges of array
        ix[ix<0] = 0
        iy[iy<0] = 0
        iz[iz<0] = 0
        ix[ix>self.nx] = self.nx
        iy[iy>self.ny] = self.ny
        iz[iz>self.nz] = self.nz

        q111 = self.data[ix    , iy    , iz  ]
        q112 = self.data[ix    , iy    , iz+1]
        q121 = self.data[ix    , iy+1, iz    ]
        q122 = self.data[ix    , iy+1, iz+1  ]
        q211 = self.data[ix+1, iy      , iz  ]
        q212 = self.data[ix+1, iy      , iz+1]
        q221 = self.data[ix+1, iy+1, iz      ]
        q222 = self.data[ix+1, iy+1, iz+1    ]

        i1 = (xd*q211 + (1-xd)*q111)
        i2 = (xd*q221 + (1-xd)*q121)
        j1 = (xd*q212 + (1-xd)*q112)
        j2 = (xd*q222 + (1-xd)*q122)

        k1 = (yd*i2 + (1-yd)*i1)
        k2 = (yd*j2 + (1-yd)*j1)

        return (zd*k2 + (1-zd)*k1)


    def _slow(self, pos):
        """ Interpolate a vector from the data array at the set of positions in
        pos.

        ************************************************************

        Don't use this one, the einsum funtion is surprisingly slow to perform
        element-wise multiplication.

        ************************************************************

        Assumptions
        -----------
            1) All entries in pos are in-bound (we'll get the standard numpy
               array out of bounds otherwise.

        """

        # Fractional potential array index.
        ixf = (pos[:,0]-self.x0)/self.dx #- 1
        iyf = (pos[:,1]-self.y0)/self.dy #- 1
        izf = (pos[:,2]-self.z0)/self.dz #- 1

        # Integer part of potential array index.
        ix = np.floor(ixf).astype(np.int)
        iy = np.floor(iyf).astype(np.int)
        iz = np.floor(izf).astype(np.int)

        # Calculate distance of point from gridlines.
        xd = (ixf - ix)
        yd = (iyf - iy)
        zd = (izf - iz)

        # Clamp out of range indicies to edges of array
        ix[ix<0] = 0
        iy[iy<0] = 0
        iz[iz<0] = 0
        ix[ix>self.nx] = self.nx
        iy[iy>self.ny] = self.ny
        iz[iz>self.nz] = self.nz

        Q111 = self.data[ix    , iy    , iz  ]
        Q112 = self.data[ix    , iy    , iz+1]
        Q121 = self.data[ix    , iy+1, iz    ]
        Q122 = self.data[ix    , iy+1, iz+1  ]
        Q211 = self.data[ix+1, iy      , iz  ]
        Q212 = self.data[ix+1, iy      , iz+1]
        Q221 = self.data[ix+1, iy+1, iz      ]
        Q222 = self.data[ix+1, iy+1, iz+1    ]

        i1 = np.einsum('i,ij->ij', xd, Q211) + np.einsum('i,ij->ij', (1-xd), Q111)
        i2 = np.einsum('i,ij->ij', xd, Q221) + np.einsum('i,ij->ij', (1-xd), Q121)
        j1 = np.einsum('i,ij->ij', xd, Q212) + np.einsum('i,ij->ij', (1-xd), Q112)
        j2 = np.einsum('i,ij->ij', xd, Q222) + np.einsum('i,ij->ij', (1-xd), Q122)

        k1 = np.einsum('i,ij->ij', yd, i2) + np.einsum('i,ij->ij', (1-yd), i1)
        k2 = np.einsum('i,ij->ij', yd, j2) + np.einsum('i,ij->ij', (1-yd), j1)

        return np.einsum('i,ij->ij', zd, k2) + np.einsum('i,ij->ij', (1-zd), k1)
