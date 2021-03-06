""" Set of classes to determine the fitness of a set of positions and velocities.
"""

import numpy as np
import logging

class MaxVelocity(object):
    """ Defines the fitness as the number of particles at the detection plane
    below a threshold velocity.
    """
    def __init__(self, geneFlyer, optprops):
        self.log = logging.getLogger(__name__)
        self.log.info('Maximum velocity fitness function')
        self.geneFlyer = geneFlyer
        self.detectorPos = geneFlyer.flyer.detectionProps['position']
        try:
            self.targetspeed = optprops['targetspeed']
        except KeyError:
            self.log.critical('No target speed in OPTIMISER options')
            raise RuntimeError('Missing fitness option')

    def __call__(self, gene):
        """ Fitness is the negative of the number of particles reaching the
        detection plane below 1.04 the target velocity.
        """
        pos, vel = self.geneFlyer.flyGene(gene)
        ind = np.where((pos[:, 2] >= self.detectorPos) & 
                    (vel[:, 2] < 1.04*self.targetspeed)
                    )[0]
        return -len(ind)

class VelocityWindow(object):
    """ Defines the fitness as the number of particles at the detection plane
    within a range of velocities centred at targetspeed+-windowwidth.
    """
    def __init__(self, geneFlyer, optprops):
        self.log = logging.getLogger(__name__)
        self.log.info('Velocity window fitness function')
        self.geneFlyer = geneFlyer
        self.detectorPos = geneFlyer.flyer.detectionProps['position']
        try:
            self.targetspeed = optprops['targetspeed']
            self.windowwidth = optprops['windowwidth']
        except KeyError:
            self.log.critical('No target speed or windowwidth in OPTIMISER options')
            raise RuntimeError('Missing fitness option')

    def __call__(self, gene):
        """ Fitness is the negative of the number of particles reaching the
        detection plane within +/- windowwidth of the targetspeed.
        """
        pos, vel = self.geneFlyer.flyGene(gene)
        ind = np.where((pos[:, 2] >= self.detectorPos) & 
                    (vel[:, 2] < self.targetspeed+self.windowwidth) &
                    (vel[:, 2] > self.targetspeed-self.windowwidth)
                    )[0]
        return -len(ind)
