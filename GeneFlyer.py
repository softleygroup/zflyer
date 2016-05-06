""" Set of classes for converting genetic codes to ontimes and durations, then
flying the Zeeman simulation.

These classes derive from the :class:`GeneFlyer` base class and must implement
the abstract methods:

* :func:`GeneFlyer.createGene`
* :func:`GeneFlyer.geneBounds`
* :func:`GeneFlyer.flyGene`
* :func:`GeneFlyer.saveGene`

Implementation
==============

The role of a :class:`GeneFlyer` is to store a reference to
:class:`ZeemanFlyer` object, and act as a conduit between this and
:class:`GeneFitness`. The class essentially converts a gene to a set of `pos`
and `vel` arrays: the positions and velocities of the particles at the end of
flight. These arrays are returned to the :class:`GeneFitness` object that
called this. The :func:`GeneFlyer.flyGene` function performs the conversion,
then calls :func:`GeneFlyer.fly` to handle interacting with the
:class:`ZeeemanFlyer`.

Utility functions
=================

Subclasses also provide three utility functions that help set-up the CMA-ES
computation, by providing an initial gene through :func:GeneFlyer.createGene`,
and and array of upper and lower bounds through a call to
:func:`GeneFlyer.geneBounds`, that may be used by the CMA-ES algorithm. A single
gene can be saved to an output file by calling :func:`GeneFlyer.saveGene`.

Subclasses
==========

Currently implemented genes are:

* :class:`DurationGene` Stores the duration of each pulse, relative to a
  starting point and using a fixed overlap.
* :class:`OffTimeGene` Stores the off-time for each coil, uses a fixed overlap.

:author: Chris Rennick
:copyright: Copyright 2015 University of Oxford.
"""

import abc
import ctypes
import ConfigParser
import logging
import os
import numpy as np

class GeneFlyer(object):
    """ Abstract base class for converting a CMA-ES gene to a set of final
    particle positions and velocities.

    This class is used as a bridge between the :class:`GeneFitness` and the
    underlying :class:`ZeemanFlyer` dynamics simulation by calling
    :func:`flyGene`.

    Attributes:
        flyer (ZeemanFlyer) : Reference to an initialised simulation.
        optprops (dict) : Dictionary of parameters from `OPTIMISER` section of
            config file.
        states (list) : List of indicies of states that are run in the `flyer`
        ontimes (numpy:array) : Array of ontimes derived from the gene.
        offtimes (numpy:array) : Array of offtimes derived from the gene.
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, flyer, optprops):
        """ Initialise the abstract base class and store the states that will
        be flown.

        Args:
            flyer (ZeemanFlyer) : Reference to an initialised simulation.
            optprops (dict) : Dictionary of optimiser parameters.
        """
        self.flyer = flyer
        self.optprops = optprops
        self.states = optprops['optstates']
        self.offtimes = None
        self.ontimes = None


    @abc.abstractmethod
    def createGene(self):
        """ Generates a suitable initial guess for a gene, which is used as the
        starting point for a CMA-ES optimisation. Usually best to load
        `ontimes` and `durations` from the `flyer` passed on creation, and
        convert these to a gene.

        Returns:
            gene (list): A gene sequence of parameters to optimise.
        """


    @abc.abstractmethod
    def geneBounds(self):
        """ Return a list of the upper and lower bounds following the CMA
        options format. These are usually stored in the `optprops` dictionary.

        Returns:
            bound (list): List of [[upper, ...], [lower, ...]] bounds for the
            gene parameter set.
        """


    @abc.abstractmethod
    def flyGene(self, gene):
        """ Convert gene and store as `self.ontimes` and `self.offtimes`, then
        call `self.fly`. Return the tuple `(pos, vel)`.

        Args:
            gene (list): Gene to convert and simulate.

        Returns:
            (tuple): tuple containing:

                * pos (numpy.array): 3D position array.
                * vel (numpy.array): 3D velocity array.
        """

    @abc.abstractmethod
    def saveGene(self, path):
        """ Convert the gene to the arrays of ontimes and durations suitable
        for use in config.info and save to a file.

        Args:
            path (string): Path to save file (file name is up to implementation).
        """

    def fly(self):
        """ Method implemented in the base class that performs the necessary
        conversion and C-alignment of the `ontimes` and `durations` arrays and
        passes them to the :class:`ZeemanFlyer`. This should be called from the
        implementation of `flyGene`.

        Returns:
            (tuple): tuple containing:

                * pos (numpy.array): 3D position array.
                * vel (numpy.array): 3D velocity array.
        """
        if self.offtimes is None or self.ontimes is None:
            raise RuntimeError('offtimes and ontimes not set')

        c_double_p = ctypes.POINTER(ctypes.c_double)           # pointer type
        self.flyer.prop.overwriteCoils(
                self.ontimes.ctypes.data_as(c_double_p),
                self.offtimes.ctypes.data_as(c_double_p))
        self.flyer.offimes = self.offtimes
        self.flyer.ontimes = self.ontimes

        self.flyer.preparePropagation()
        pos = np.zeros((0, 3))
        vel = np.zeros((0, 3))
        for i in self.states:
            new_pos, new_vel, _ = self.flyer.propagate(i)
            pos = np.concatenate((pos, new_pos))
            vel = np.concatenate((vel, new_vel))
        return (pos, vel)


    def saveGene(self, path):
        """ Save the ontimes and offtimes of the last-run gene to a config file
        named optimised.info. Only the `[PROPAGATION]` section is created,
        containing `ontimes` and `durations`.

        Args:
            path (string): Path in which to save.
        """

        # Format a numpy array as a string, stripping extra spaces, and neatly
        # comma delimiting the numbers followed by a space:
        # np.array([1.0, 2.0, 3.0])
        def fm(a):
            return 'np.' + ''.join(repr(a).split()).replace(',', ', ')

        config = ConfigParser.ConfigParser(allow_no_value = True)
        config.optionxform=str
        config.add_section('PROPAGATION')
        config.set('PROPAGATION', '; Optimised using ' + self.__class__.__name__)
        config.set('PROPAGATION', 'ontimes', fm(self.ontimes))
        config.set('PROPAGATION', 'durations', fm(self.offtimes-self.ontimes))

        with open(path, 'wb') as f:
            config.write(f)


class OffTimeGene(GeneFlyer):
    """ Gene that stores the off times for each coil. On times are calculated
    assuming a 6 us overlap.

    Other Parameters:
        Options taken from the `config.info` file:

        * maxofftime -- The upper bound for each offtime.
        * minofftime -- The lower bound for each offtime.
    """
    def __init__(self, flyer, optprops):
        super(OffTimeGene, self).__init__(flyer, optprops)
        self.log = logging.getLogger(__name__)
        self.log.info('Using an OffTimeGene')


    def createGene(self):
        """ Just return the offtimes loaded from the config file by the `flyer`
        object.
        """
        return self.flyer.offtimes[:]


    def geneBounds(self):
        """ Upper and lower bounds taken from config.info.
        """
        try:
            maxOffTime = self.optprops['maxofftime']
            minOffTime = self.optprops['minofftime']
        except KeyError:
            log.critical('maxofftime or minofftime missing from config.info.')
            raise RuntimeError('Missing config values')

        return [12*[minOffTime], 12*[maxOffTime]]


    def flyGene(self, gene):
        """ Offtimes are given by the gene, ontimes are taken as 6 us before
        each subsequent `ontime`. A fixed 30 us duration sets the ontime of the
        first coil.
        """
        self.offtimes = np.require(gene[:12].copy(),
                requirements=['c', 'a', 'o', 'w'])
        self.ontimes = np.zeros((12,))
        self.ontimes[1:] = self.offtimes[:11] - 6
        self.ontimes[0] = self.offtimes[0] - 30

        return self.fly()


    def saveGene(self, gene, path):
        self.offtimes = np.require(gene[:12].copy(),
                requirements=['c', 'a', 'o', 'w'])
        self.ontimes = np.zeros((12,))
        self.ontimes[1:] = self.offtimes[:11] - 6
        self.ontimes[0] = self.offtimes[0] - 30

        print np.transpose((ontimes, offtimes, offtimes-ontimes))
        super(OffTimeGene, self).saveGene(path)


class DurationGene(GeneFlyer):
    """ This gene stores the duration of each coil pulse. Each coil pulse is
    taken to overlap the previous one by 6 us. The ontime for the first coil is
    stored from the input file and used to calculate all other switching times
    using the duration in the gene and a standard 6 us overlap.

    The actual ontime of the first coil is then set to 30 us before its offtime
    to ensure this first pulse is long enough.

    Note:

        The arbitrary ontime of coil 1 used for the duration gene may result in
        a negative duration for coil 1. This is not necessarily a problem, as
        the duration is fixed at 30 us and a correct sequence will be produced.

    Other Parameters:
        Options taken from `config.info`:

        * maxduration : The maximum pulse duration in us.
        * minduration : The minimum pulse duration in us.

    """
    def __init__(self, flyer, optprops):
        super(DurationGene, self).__init__(flyer, optprops)
        self.log = logging.getLogger(__name__)
        self.log.info('Using a DurationGene')

    def createGene(self):
        """ Store a zero-time as the first coil ontime, as given in the config
        file. The first coil duration is then relative to this. The duration of
        other coils is taken from the config file.
        """
        ontimes = self.flyer.ontimes
        offtimes = self.flyer.offtimes
        self.t0 = ontimes[0]

        return offtimes-ontimes


    def geneBounds(self):
        """ Upper and lower bounds for the gene are taken from parameters
        `maxduration' and `minduration` in config.info.
        """
        try:
            maxDuration = self.optprops['maxduration']
            minDuration = self.optprops['minduration']
        except KeyError:
            log.critical('maxduration or minduration missing from config.info.')
            raise RuntimeError('Missing config values')

        return [12*[minDuration], 12*[maxDuration]]


    def flyGene(self, gene):
        """ Convert the gene of durations to a set of `ontimes` and `offtimes`
        and fly the simulation.  The switching times are calculated by working
        forward from the initial time `t0` stored from initial sequence in the
        input file. This time is arbitrary, however, and only used to as a
        starting point for the relative times. The actual ontime for coil 1 is
        set to 30 us.
        """
        self._convertGene(gene)
        return self.fly()


    def saveGene(self, gene, path):
        self._convertGene(gene)
        durations = self.offtimes - self.ontimes
        print np.transpose((self.ontimes, self.offtimes, durations))
        super(DurationGene, self).saveGene(path)


    def _convertGene(self, gene):
        """ Convert the gene to ontimes and offtimes.
        """
        durations = np.require(gene[:12].copy(),
                requirements=['c', 'a', 'o', 'w'])

        self.ontimes = np.zeros((12,))
        self.offtimes = np.zeros((12,))
        # set the first coil on time to the stored value from config.
        self.ontimes[0] = self.t0 + durations[0] - 30
        self.offtimes[0] = self.t0 + durations[0]
        # The next coil ontime is 6 us before the previous coil is turned off.
        for i in range(1, len(durations)):
            self.ontimes[i] = self.offtimes[i-1] - 6
            self.offtimes[i] = self.ontimes[i] + durations[i]
