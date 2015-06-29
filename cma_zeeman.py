
import ConfigParser
import cma
import ctypes
import importlib
import logging
import numpy as np

from sim_zeeman import ZeemanFlyer
import GeneFitness
import GeneFlyer

def loadParameters(config_file, section):
    """ Load parameters from `section` in the config file into a dict.
    Parameters are parsed by calling `eval` builtin function. Those that fail
    the evaluation are logged and dropped from the parameters dict.
    """
    log = logging.getLogger('loadParameters')
    log.debug('Reading optimisation parameters from section %s' % section)
    config = ConfigParser.SafeConfigParser()
    config.optionxform = str
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
        raise RuntimeError (e)


def optimise_cma_fixed(flyer, config_file, path):
    """ Load parameters and perform the optimisation.
    """
    log = logging.getLogger('optimise')
    # Load parameters from config.info.
    cmaopt = loadParameters(config_file, 'CMA')
    optprops = loadParameters(config_file, 'OPTIMISER')
    try:
        sigma0 = optprops['sigma0']
        optStates = optprops['optstates']
    except KeyError as e:
        log.critical('No parameter named %s in OPTIMISER section' % e)
        raise RuntimeError(e)

    # Attempt to load the GeneFlyer class specified in the input file.
    try:
        geneClass = GeneFlyer.__dict__[optprops['genetype']]
    except KeyError as e:
        log.critical('No GeneFlyer class named %s.' % e)
        raise RuntimeError(e)
    geneFlyer = geneClass(flyer, optprops)

    # Attempt to load the GeneFitness class specified in the input file.
    try:
        fitnessClass = GeneFitness.__dict__[optprops['fitnesstype']]
    except KeyError as e:
        log.critical('No GeneFitness class named %s' % e)
        raise RuntimeError(e)
    fitness = fitnessClass(geneFlyer, optprops)

    # Initialise the gene and get the limits for each parameter.
    initval = geneFlyer.createGene()
    cmaopt['bounds'] = geneFlyer.geneBounds()

    # Initialise the CMA optimiser.
    es = cma.CMAEvolutionStrategy(initval, sigma0, cmaopt)
    nh = cma.NoiseHandler(es.N, [1, 1, 30])
    cma_log = cma.CMADataLogger().register(es)

    while not es.stop():
        x, fit = es.ask_and_eval(fitness, evaluations=nh.evaluations)
        es.tell(x, fit)  # prepare for next iteration
        es.disp()
        es.eval_mean(fitness)
        cma_log.add()
        print '========= evaluations: ', es.countevals, '========'
        print '========= current mean: ', es.fmean, '========'
        print es.mean
        print '========= current best: ', es.best.f, '========'
        print es.best.x

    print(es.stop())

    geneFlyer.saveGene(es.result()[-2], path)

    #ontimes = np.zeros(12)
    #offtimes = es.result()[-2][:12]
    #ontimes[1:] = offtimes[:11] - 6
    #ontimes[0] = offtimes[0] - 30
    #print 'mean ontimes: ', ontimes  # take mean value, the best solution is totally off
    #print 'mean durations: ', offtimes-ontimes  # take mean value, the best solution is totally off
    #offtimes = x[np.argmin(fit)][:12]
    #ontimes[1:] = offtimes[:11] - 6
    #ontimes[0] = offtimes[0] - 30
    #print 'best ontimes: ', ontimes  # not bad, but probably worse than the mean
    #print 'best durations: ', offtimes-ontimes  # not bad, but probably worse than the mean

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
            level=logging.INFO)
    if not args.q:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(levelname)s - %(name)s - %(message)s'))
        logging.getLogger().addHandler(ch)

    config_file = os.path.join(folder, 'config.info')
    log.info('Running analysis in folder %s' % folder)
    if not os.path.exists(config_file):
        log.critical('Config file not found at %s' % config_file)
        sys.exit(1)

    flyer = ZeemanFlyer(verbose=False)
    flyer.loadParameters(config_file)
    flyer.addParticles(checkSkimmer=True) #, NParticlesOverride=3000)
    flyer.calculateCoilSwitching()
    flyer.loadBFields()
    flyer.preparePropagation()

    try:
        res = optimise_cma_fixed(flyer, config_file, folder)
    except RuntimeError:
        log.critical('Optimisation failed.')
        sys.exit(1)

    print res
