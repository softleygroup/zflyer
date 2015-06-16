import numpy as np
import logging

""" Check the config file entries against expected type and naming. After
loading and parsing by the ZeemanFlyer class, check each variable conforms to
an expected type. The types are stored in a dictionary with keys matching each
section of the config.info file. The values are lists of tuples containing the
parameter name, and a tuple of permitted types:
	(name, (float, int))
This allows float numbers to be specified as integers in the config.info file.
"""
# A dictionary of item names expected in the config.info file and the expected
# numeric type of each.
config_items = { 
		'particle' : [
			('mass', float)],
		'bunch' : [
			('TRadial', float),
			('TLong', float),
			('v0', (np.ndarray, list)),
			('NParticles', int),
			('useEGun', bool),
			('radius', (float, int)),
			('length', (float, int)),
			('egunPulseDuration', (float, int)),
			('zeemanStates', int)],
		'propagation' : [
			('timestep', (float, int)),
			('starttime', (float, int)),
			('timestepPulse', (float, int)),
			('stoptime', (float, int)),
			('phase', (float, int, type(None))),
			('ontimes', (np.ndarray, list)),
			('durations', (np.ndarray, list))],
		'coils' : [
			('NCoils', int),
			('position', np.ndarray),
			('radius', (float, int)),
			('H1', (float, int)),
			('H2', (float, int)),
			('ramp1', (float, int)),
			('timeoverlap', (float, int)),
			('rampcoil', (float, int)),
			('current', np.ndarray),
			('maxPulseLength', (float, int))], 
		'skimmer' : [
			('position', (float, int)),
			('radius', (float, int)),
			('length', (float, int)),
			('backradius', (float, int))],
		'detection' : [
			('position', (float, int))],
		'optimiser' : [
			('position', (float, int)),
			('targetSpeed', (float, int)),
			('optStates', int),
			('maxPulseDuration', (float, int))]
		}

def test_parameters(flyer):
	""" Check each of the parameters in class flyer for numerical type
	correctness and length of arrays.
	"""
	check_param_set(flyer.particleProps, 'particle')
	check_param_set(flyer.bunchProps, 'bunch')
	check_param_set(flyer.propagationProps, 'propagation')
	check_param_set(flyer.coilProps, 'coils')
	check_param_set(flyer.skimmerProps, 'skimmer')
	check_param_set(flyer.detectionProps, 'detection')
	check_param_set(flyer.optimiserProps, 'optimiser')


def check_param_set(d, name):
	allOK = True
	for k in config_items[name]:
		try:
			if not isinstance(d[k[0]], k[1]):
				logging.critical('Parameter "%s" is not of type %s. Found entry of type %s' % (k[0], k[1], type(d[k[0]])))
				allOK = False
		except KeyError:
			logging.critical('Parameter "%s" is missing from section [%s]' % (k[0], name))
			allOK = False
	if not allOK:
		logging.critical('Some items missing from configuration file.')
		raise RuntimeError('Error in config file.')


if __name__ == '__main__':
	from optparse import OptionParser               # reading command line arguments
	import sys
	import os
	from sim_zeeman import ZeemanFlyer
	# If this is run directly, test an input file without flying simulation.
	logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)

	parser = OptionParser('Usage: %prog "Working Directory"')
	(options, args) = parser.parse_args()
	if len(args) != 1:
		parser.error("Directory not specified")

	folder = args[0]
	config_file = os.path.join(folder, 'config.info')

	logging.info('Testing config file in folder %s' % folder)
	if not os.path.exists(config_file):
		logging.critical('Config file not found at %s' % config_file)
		sys.exit(1)

	flyer = ZeemanFlyer()
	flyer.loadParameters(config_file)

	try:
		test_parameters(flyer)
		logging.info('No Errors in config file.')
	except RuntimeError as e:
		logging.critical(e.message)
		sys.exit(1)
