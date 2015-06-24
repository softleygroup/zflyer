
import ConfigParser
import cma
import ctypes
import logging
import numpy as np

from sim_zeeman import ZeemanFlyer

def loadParameters(config_file, section):
    """ Load parameters from `section` in the config file into a dict.
    
    Parameters are parsed by calling `eval` builtin function. Those that fail
    the evaluation are logged and dropped from the parameters dict.
    """
    log = logging.getLogger('loadParameters')
    log.debug('Reading optimisation parameters from section %s' % section)
    config = ConfigParser.SafeConfigParser()
    config.read(config_file)
    d = {}
    try:
        items = config.items(section)
        for k, v in items:
            try:
                d[k] = eval(v)
            except(ValueError, NameError):
                log.error('Could not parse option named "%s", value "%s",' +
                        'removing from parameters dict'  % (str(k), str(v)))
        return d
    except ConfigParser.NoSectionError as e:
        log.critical('Input file does not contain a section named %s'
                % e.section)
        raise RuntimeError

class Fitness(object):
    """ Fitness class to hold properties and `ZeemanFlyer` object required to
    fly particles. Initialisation stores parameters and sets up logging, then
    returning this callable class which can be passed as the fitness function
    """
    def __init__(self, flyer, optprops):
        self.log = logging.getLogger('fitness')
        self.optstates = optprops['optstates']
        self.targetspeed = optprops['targetspeed']
        self.flyer = flyer

    def __call__(self, gene):
        """ Determine the fitness of a gene by the number of particles with
        acceptable properties at the end of the decelerator. The gene encodes the
        on-times for each coil, the off-times are given by the standard field-ramp
        overlap conditions.
        """
        #flyer.addparticles(checkskimmer=true, nparticlesoverride=1200)
        offtimes = np.require(gene[:12].copy(), requirements=['c', 'a', 'o', 'w'])
        ontimes = np.zeros((12,))
        ontimes[1:] = offtimes[:11] - 6
        ontimes[0] = offtimes[0] - 30
        c_double_p = ctypes.POINTER(ctypes.c_double)           # pointer type
        self.flyer.prop.overwriteCoils(ontimes.ctypes.data_as(c_double_p), 
                offtimes.ctypes.data_as(c_double_p))

        #currents = [243.]*12
        #flyer.preparepropagation(currents)
        self.flyer.preparePropagation()
        fval = 0
        #for i in np.arange(optprops['optstates']):
        for i in self.optstates:
            pos, vel, _ = self.flyer.propagate(i)
            ind = np.where((pos[:, 2] >= self.flyer.detectionProps['position']) & 
                    (vel[:, 2] < 1.04*self.targetspeed)
                    )[0]
            #r = np.sqrt(pos[ind, 0]**2 + pos[ind, 1]**2)
            #fval = ind.shape[0]**2/r.mean()
            fval += ind.shape[0]
        #print 'good particles:', fval
        return -fval

def optimise_cma_fixed(flyer, config_file):
    log = logging.getLogger('optimise')
    optprops = loadParameters(config_file, 'OPTIMISER')
    try:
        maxOffTime = optprops['maxofftime']
        minOffTime = optprops['minofftime']
        sigma0 = optprops['sigma0']
    except KeyError as e:
        log.critical('No parameter named %s in OPTIMISER section' % e)
        raise RuntimeError(e)

    cmaopt = loadParameters(config_file, 'CMA')
    cmaopt['bounds'] = [12*[minOffTime], 12*[maxOffTime]]

    fitness = Fitness(flyer, optprops)
    initval = flyer.offtimes[:]
    es = cma.CMAEvolutionStrategy(initval, sigma0, cmaopt)
    nh = cma.NoiseHandler(es.N, [1, 1, 30])

    while not es.stop():
        x, fit = es.ask_and_eval(fitness, evaluations=nh.evaluations)
        es.tell(x, fit)  # prepare for next iteration
        es.disp()
        es.eval_mean(fitness)
        print '========= evaluations: ', es.countevals, '========'
        print '========= current mean: ', es.fmean, '========'
        print es.mean.x
        print '========= current best: ', es.best.f, '========'
        print es.best.x
    print(es.stop())
    ontimes = np.zeros(12)
    offtimes = es.result()[-2][:12]
    ontimes[1:] = offtimes[:11] - 6
    ontimes[0] = offtimes[0] - 30
    print 'mean ontimes: ', ontimes  # take mean value, the best solution is totally off
    print 'mean durations: ', offtimes-ontimes  # take mean value, the best solution is totally off
    offtimes = x[np.argmin(fit)][:12]
    ontimes[1:] = offtimes[:11] - 6
    ontimes[0] = offtimes[0] - 30
    print 'best ontimes: ', ontimes  # not bad, but probably worse than the mean
    print 'best durations: ', offtimes-ontimes  # not bad, but probably worse than the mean

    #return es



if __name__ == '__main__':
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('wd', 
            help='The working directory containing the config.info file.')
    parser.add_argument('-c', 
            help='Console mode. Will not produce plots on completion',
            action='store_true')
    parser.add_argument('-q', 
            help='Quiet mode. Does not produce any output; still log messages to file.', 
            action='store_true')
    parser.add_argument('-d', 
            help='Debug mode', 
            action='store_true')
    args = parser.parse_args()
    folder = args.wd

    # Set up logging to console and file.
    log = logging.getLogger('main')
    logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)-8s : %(message)s',
            datefmt='%d%m%y %H:%M',
            filename=os.path.join(folder, 'log.txt'),
            filemode='w',
            level=logging.WARN)
    if not args.q:
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARN)
        ch.setFormatter(logging.Formatter('%(levelname)s - %(name)s - %(message)s'))
        logging.getLogger().addHandler(ch)

    config_file = os.path.join(folder, 'config.info')
    log.info('Running analysis in folder %s' % folder)
    if not os.path.exists(config_file):
        log.critical('Config file not found at %s' % config_file)
        sys.exit(1)

    flyer = ZeemanFlyer(verbose=False)
    flyer.loadParameters(config_file)
    flyer.addParticles(checkSkimmer=True, NParticlesOverride=3000)
    flyer.calculateCoilSwitching()
    flyer.loadBFields()
    flyer.preparePropagation()

    try:
        res = optimise_cma_fixed(flyer, config_file)
    except RuntimeError:
        log.critical('Optimisation failed.')
        sys.exit(1)

    print res
