""" Optimise a Zeeman switching sequence using a covariance matrix adaptation
evolutionary strategy (`CMA-ES
<https://www.lri.fr/~hansen/html-pythoncma/frames.html>`_).

The gene is a list of numbers that represents some parameters of the coil
switching sequence, and the CMA algorithm seeks an optimum set of parameters
that minimise an objective function. This program uses a set of modules to
separate the meaning of a gene sequence, and the determination of its fitness.
The intention is to allow easy and flexible substitution of each part of the
optimisation. The structure is a heirarchy of abstract classes that pass the
gene down eventually to a :class:ZeemanFlyer to run the simulation, then the
final positions and velocities are passed back up the stack eventually to a
fitness useful to CMA.

Simulation Structure
====================

The genes are handled by a stack of abstract classes that compartmentalise the
gene interpretation and fitness. Each of these classes is abstract and concrete
forms implement the behavior.  Only the :class:`GeneFlyer` class knows how to
convert a gene to a switching sequence, and only the :class:`GeneFitness` class
knows how to convert final positions and velocities into fitness.::

                                  +-------------+
                                  |     CMA     |
                               |- +-------------+<-|
                  pass a gene  |                   |  Return fitness 
                               |->+-------------+ -|                 
                                  | GeneFitness |                    
                               |- +-------------+<-|                 
                 pass a gene   |                   |  Return pos, vel
                               |->+-------------+ -|                 
                                  |  GeneFlyer  |                    
                               |- +-------------+<-|                 
    convert gene to switching  |                   |  Return pos, vel
                               |->+-------------+ -|
                                  | ZeemanFlyer |
                                  +-------------+

The CMA optimisation routine expects a function that takes a gene and returns a
fitness. This is implemented by :func:`GeneFitness.__call__`, which passes the
gene on to :func:`GeneFlyer.GeneFlyer.flyGene`. The
:func:`GeneFlyer.GeneFlyer.flyGene` converts the gene to a set of ontimes and
durations, and calls :func:`sim_zeeman.ZeemanFlyer.propagate`. The positions
and velocties are passed back up through :class:`GeneFitness`, which calculates
the fitness (lower is fitter).  CMA uses this fitness to iterate the next
generation.

See Also:
    :mod:`GeneFlyer`
        Module to fly a simulation using a gene.
    :mod:`GeneFitness`
        Module to determine fitness from final positions and velocities.

"""
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
    the evaluation are logged and not included in the parameters dict.

    Args:
        config_file (string) : Full path to file.
        section (string) : Name of [SECTION] to parse.

    Returns:
        d (dict) : Dictionary of values stored by parameter name.
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

    # Set the working directory to save CMA log files.
    os.chdir(path)

    # Load parameters from config.info and get the initial sigma.
    cmaopt = loadParameters(config_file, 'CMA')
    optprops = loadParameters(config_file, 'OPTIMISER')
    try:
        sigma0 = optprops['sigma0']
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

    geneFlyer.saveGene(es.result()[-2], os.path.join(path, 'mean.info'))
    geneFlyer.saveGene(es.best.x, os.path.join(path, 'best.info'))

    return es


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
    log = logging.getLogger('cma_main')
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

    folder = os.path.abspath(folder)
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
