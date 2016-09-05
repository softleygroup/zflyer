""" Utility functions for flying atoms in magnetic fields.
"""
from IPython.core.debugger import Tracer

import ConfigParser
from functools import reduce
import logging
import numpy as np
import os
import scipy.constants as c

#from hexapole import Hexapole

LOG = logging.getLogger('Verlet')

def loadFinal(path, states=[-1, 0, 1, 2, 3], onlyLive=True, excludeSynchronous=True):
    """ Load final positions and velocities from an output file path.

    This loads specified state, or concatenates all positions and velocities
    into an array of states. The default argument loads all. States are
    specified using the labels from sim_zeeman.

    Arguments:
        path (String):
            Directory in which sim_zeeman was run.
        states (list):
            States to load, e.g.::
                states=[-1]        # Load a single state.
                states=[0,1,2,3]   # Load and concatenate a set of states.
        onlyLive (Bool):
                Only return particles that made it through to the detection
                region. Optional, default `True`.
        excludeSynchronous (Bool):
                Exclude the first particle.

    Returns:
        pos, vel ((n,3) np.ndarray):
            Array of particle positions and velocities.
        times ((n, ) np.ndarray):
            Array of final time-of-flight.
    """
    pos = np.empty(shape=(0, 3), dtype=np.float64)
    vel = np.empty(shape=(0, 3), dtype=np.float64)
    times = np.empty(shape=(0, ), dtype=np.float64)

    for s in states:
        posfile = 'finalpos{}.npy'.format(s)
        velfile = 'finalvel{}.npy'.format(s)
        timesfile = 'finaltimes{}.npy'.format(s)
        p = np.load(os.path.join(path, posfile))
        v = np.load(os.path.join(path, velfile))
        t = np.load(os.path.join(path, timesfile))

        pos = np.concatenate((pos, p))
        vel = np.concatenate((vel, v))
        times = np.concatenate((times, t))


    if onlyLive:
        LOG.debug('Removing collided atoms from initial list.')
        config = ConfigParser.SafeConfigParser()
        config.read(os.path.join(path, 'config.info'))
        detectorPos = eval(config.get('DETECTION', 'position'))
        ind = np.where(pos[:,2]>=detectorPos)[0]
        pos = pos[ind, :]
        vel = vel[ind, :]
        times = times[ind]

    if excludeSynchronous:
        pos = pos[1:]
        vel = vel[1:]
        times = times[1:]

    return pos, vel, times


def rewind(position, pos, vel, times):
    """ Reverse particles along their free flight trajectories until their z
    coordinates are at the given position, rewind the time of flight by the
    correct amount. 

    The arrays are modified in place, so original values are overwritten.

    Arguments:
        position (float):
            Target location to reverse particles.
        pos, vel ((n,3) np.ndarray) (mm, mm/usec):
            Array of particle positions and velocities.
        times ((n, ) np.ndarray) (us):
            Array of final time-of-flight.

    Returns:
        pos, vel ((n,3) np.ndarray) (mm, mm/usec):
            Array of particle positions and velocities.
        times ((n, ) np.ndarray) (us):
            Array of final time-of-flight.
    """

    # Distance along z to move each particle, and the corresponding time.
    delta_d = position - pos[:,2]
    # Time for free flight over this distance.
    delta_t = delta_d/vel[:,2]

    # Rewind time of flight values
    times += delta_t

    # Repeat the time vector in three columns to simplify the following
    # calculation for x, y, z
    delta_t = np.repeat(np.atleast_2d(delta_d/vel[:,2]).T, 3, axis=1)

    # Rewind particles to new positions
    pos += delta_t * vel

    return pos, vel, times


def verletFlyer(pos, vel, times, state, hexapole, dt=1e-3,
        totalTime=1e3, totalZ=1000):
    """ Move hydrogen atoms in a static magnetic field stopping after either a
    total flight time, or when all atoms have arrived at the given z position.
    All input coordinates are assumed to be in the hexapole frame of reference.

    `hexapole` is a `Hexapole` class that can calculate the magnetic field when
    given a list of x, y, z coordinates, and identify atoms that have collided
    with the surface.

    Arguments:
        pos, vel ((n,3) np.ndarray) (mm, mm/usec):
            Array of particle positions and velocities.
        times ((n, ) np.ndarray) (us):
            Array of final time-of-flight.
        state (int):
            State label from set [0, 1, 2, 3]
        hexapole (Hexapole):
            Hexapole class.
        dt (float) (us):
            Time step size in microseconds.
        totalTime (float) (us):
            Total time to propagate in microseconds, stop when exeeded.
        totalZ (float) (mm):
            Total distance in z to propagate, stop when exceeded.
    """

    uB = c.physical_constants['Bohr magneton'][0]
    A = 1420405751.768 * 2.0 * c.pi/c.hbar # hf splitting in 1/((s^2)*J)
    mass = 1.00782503 * c.u # kg

    pos = np.atleast_2d(pos)
    vel = np.atleast_2d(vel)
    times = np.atleast_1d(times)

    # Definition of four Zeeman energy gradent functions in J/T. Choose using
    # standard state-ordering labels from sim_zeeman.
    dEdB_func = [
            lambda B : uB,
            lambda B : 1.0/np.sqrt((A**2 * c.hbar**4)/(4 * B**2 * uB**4) + 1/uB**2),
            lambda B : -uB,
            lambda B : -1.0/np.sqrt((A**2 * c.hbar**4)/(4 * B**2 * uB**4) + 1/uB**2)
    ]

    # Force = dBdx * dEdB
    B = hexapole.B(pos)
    dEdB = np.repeat(np.atleast_2d(dEdB_func[state](B)).T, 3, axis=1)
    dBdx = hexapole.dB(pos)
    acc = (-dEdB * dBdx)/mass * 1e-6 # mm us^-2
    acc_new = acc

    elapsedTime = 0
    while elapsedTime < totalTime:
        # Identify particles that will move
        ind_notcollided = hexapole.notCollided(pos)
        ind_notFinished = np.where(pos[:,2]<totalZ)[0]
        ind_moving = np.where(vel[:,2]>0)[0]
        ind_move = reduce(np.intersect1d, 
                (ind_notcollided, ind_notFinished, ind_moving))

        LOG.debug('Moving {} particles'.format(len(ind_move)))
        if len(ind_move)==0:
            break

        # 1. Position full-step x(t+dt) = x(t) + V(t)dt + 1/2 a(t) dt^2
        pos[ind_move, :] += vel[ind_move, :]*dt + 0.5 * acc[ind_move, :] * dt**2

        # 2. New acceleration at x(t+dt) a(t+dt) = ...
        B[ind_move] = hexapole.B(pos[ind_move, :])
        dEdB = np.repeat(np.atleast_2d(dEdB_func[state](B[ind_move])).T, 3, axis=1)
        dBdx[ind_move, :] = hexapole.dB(pos[ind_move, :])
        acc_new[ind_move, :] = (-dEdB * dBdx[ind_move, :])/mass * 1e-6

        # 3. New Velocity V(t+dt) = v(t) + 1/2(a(t) + a(t+dt))dt
        vel[ind_move, :] += 0.5 * (acc[ind_move, :] + acc_new[ind_move, :]) * dt

        acc = acc_new

        elapsedTime += dt
        times[ind_move] += dt

    if elapsedTime >= totalTime:
        LOG.info('Stopping flight, total time exceeded.')
    if len(ind_notcollided) == 0:
        LOG.info('Stopping flight, all particles lost.')
    if len(ind_move) == 0:
        LOG.info('Stopping flight, particles reached final z.')

    return pos, vel, times


