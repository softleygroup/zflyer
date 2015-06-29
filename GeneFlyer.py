""" Set of classes for converting genetic codes to ontimes and durations, then
flying the Zeeman simulation.

A subclass must subclass the functions `flyGene`

The GeneFlyer base class passes the coil times to a flyer object, prepares, and
propagates the atoms. It returns the final positions and velocities. This base
class defines an abstract flyGene method that subclasses must implement to
convert a gene to ontimes and offtimes.
"""

import abc
import ctypes
import logging
import numpy as np

class GeneFlyer(object):
    """ Abstract base class for simulating a CMA gene. Subclasses must
    implement the `flyGene` method which calls `self.fly` and returns the tuple
    `(pos, vel)`; and the method `createGene` to create a gene from a set of
    ontimes and offtimes.

    Args:
        flyer (ZeemanFlyer) : Used to fly atoms.
        states (list) : List of Zeeman states to fly.
        optprops (dict) : Dictionary of optimiser parameters.
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, flyer, optprops):
        """ Initialise the abstract base class and store the states that will
        be flown.
        """
        self._flyer = flyer
        self._optprops = optprops
        self._states = optprops['optstates']
        self._offtimes = None
        self._ontimes = None


    @staticmethod
    @abc.abstractmethod
    def createGene(ontimes, offtimes):
        """ Convert `ontimes` and `offtimes` to a gene. Best implemented as a
        static method as it does not use the parameters of an object.
        """


    @abc.abstractmethod
    def geneBounds(self):
        """ Return a list of the upper and lower bounds following the CMA
        options format.
        """


    @abc.abstractmethod
    def flyGene(self, gene):
        """ Convert gene and store as `self.ontimes` and `self.offtimes`, then
        call `self.fly`. Return the tuple `(pos, vel)`.

        Returns:
            (pos, vel) (tuple of np.array): Position and velocity arrays.
        """

    @abc.abstractmethod
    def saveGene(self, filename):
        """ Convert the gene to the arrays of ontimes and durations suitable
        for use in config.info and save to a file.
        """

    def fly(self):
        """ Pass the ontimes and offtimes to the flyer object, and fly the
        atoms. Accumulates all positions and velocities into a pair of arrays
        and returns these.

        Returns:
            (pos, vel) (tuple of np.array): Position and velocity arrays.
        """
        if self._offtimes is None or self._ontimes is None:
            raise RuntimeError('offtimes and ontimes not set')

        c_double_p = ctypes.POINTER(ctypes.c_double)           # pointer type
        self._flyer.prop.overwriteCoils(
                self._ontimes.ctypes.data_as(c_double_p),
                self._offtimes.ctypes.data_as(c_double_p))
        self._flyer.offimes = self._offtimes
        self._flyer.ontimes = self._ontimes

        self._flyer.preparePropagation()
        pos = np.zeros((0, 3))
        vel = np.zeros((0, 3))
        for i in self._states:
            new_pos, new_vel, _ = self._flyer.propagate(i)
            pos = np.concatenate((pos, new_pos))
            vel = np.concatenate((vel, new_vel))
        return (pos, vel)


class OffTimeGene(GeneFlyer):
    """ Gene that stores the off times for each coil. On times are calculated
    assuming a 6 us overlap.
    """
    def __init__(self, flyer, optprops):
        super(OffTimeGene, self).__init__(flyer, optprops)
        self.log = logging.getLogger(__name__)
        self.log.info('Using an OffTimeGene')


    def createGene(self):
        """ Just return the offtimes loaded from the config file by the `flyer`
        object.
        """
        return self._flyer.offtimes[:]


    def geneBounds(self):
        """ Bounds from config.info.
        """
        try:
            maxOffTime = self._optprops['maxofftime']
            minOffTime = self._optprops['minofftime']
        except KeyError:
            log.critical('maxofftime or minofftime missing from config.info.')
            raise RuntimeError('Missing config values')

        return [12*[minOffTime], 12*[maxOffTime]]


    def flyGene(self, gene):
        self._offtimes = np.require(gene[:12].copy(),
                requirements=['c', 'a', 'o', 'w'])
        self._ontimes = np.zeros((12,))
        self._ontimes[1:] = self._offtimes[:11] - 6
        self._ontimes[0] = self._offtimes[0] - 30

        return self.fly()


    def saveGene(self, gene, filename):
        offtimes = np.require(gene[:12].copy(),
                requirements=['c', 'a', 'o', 'w'])
        ontimes = np.zeros((12,))
        ontimes[1:] = self._offtimes[:11] - 6
        ontimes[0] = self._offtimes[0] - 30

        print np.transpose((ontimes, offtimes, offtimes-ontimes))


class DurationGene(GeneFlyer):
    """ This gene stores the duration of each coil pulse. Each coil pulse is
    taken to overlap the previous one by 6 us. The ontime for the first coil is
    stored from the input file and used to calculate all other switching times
    using the duration in the gene and a standard 6 us overlap.

    The actual ontime of the first coil is then set to 30 us before its offtime
    to ensure this first pulse is long enough.

    Note
    ----
    
    The arbitrary ontime of coil 1 used for the duration gene may result
    in a negative duration for coil 1. This is not necessarily a problem, as
    the duration is fixed at 30 us and a correct sequence will be produced.
    """
    def __init__(self, flyer, optprops):
        super(DurationGene, self).__init__(flyer, optprops)
        self.log = logging.getLogger(__name__)
        self.log.info('Using a DurationGene')

    def createGene(self):
        """ Just return the offtimes loaded from the config file by the `flyer`
        object.
        """
        ontimes = self._flyer.ontimes
        offtimes = self._flyer.offtimes
        self.t0 = ontimes[0]

        return offtimes-ontimes


    def geneBounds(self):
        """ Upper and lower bounds for the gene are taken from parameters
        `maxduration' and `minduration` in config.info.
        """
        try:
            maxOffTime = self._optprops['maxofftime']
            minOffTime = self._optprops['minofftime']
        except KeyError:
            log.critical('maxofftime or minofftime missing from config.info.')
            raise RuntimeError('Missing config values')

        return [12*[minOffTime], 12*[maxOffTime]]


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
    

    def saveGene(self, gene, filename):
        self._convertGene(gene)
        durations = self._offtimes - self._ontimes
        print np.transpose((self._ontimes, self._offtimes, durations))



    def _convertGene(self, gene):
        """ Convert the gene to ontimes and offtimes.
        """
        durations = np.require(gene[:12].copy(),
                requirements=['c', 'a', 'o', 'w'])

        self._ontimes = np.zeros((12,))
        self._offtimes = np.zeros((12,))
        # set the first coil on time to the stored value from config.
        self._ontimes[0] = self.t0
        self._offtimes[0] = self.t0 + durations[0]
        # The next coil ontime is 6 us before the previous coil is turned off.
        for i in range(1, len(durations)):
            self._ontimes[i] = self._offtimes[i-1] - 6
            self._offtimes[i] = self._ontimes[i] + durations[i]
