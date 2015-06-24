from sim_zeeman import ZeemanFlyer
import numpy as np

import ctypes
from ctypes import c_double, c_uint, c_int 		# shorthand form for data types
c_double_p = ctypes.POINTER(c_double)			# pointer type
import logging
import sys
import os


def optimise_minuit(target_time):
	from iminuit import Minuit
	def minimizer(on1, on2, on3, on4, on5, on6, on7, on8, on9, on10, on11, on12, delta1, delta2, delta3, delta4, delta5, delta6, delta7, delta8, delta9, delta10, delta11, delta12):
		ontimes = np.array([on1, on2, on3, on4, on5, on6, on7, on8, on9, on10, on11, on12])
		offtimes = np.array([on1 + delta1, on2 + delta2, on3 + delta3, on4 + delta4, on5 + delta5, on6 + delta6, on7 + delta7, on8 + delta8, on9 + delta9, on10 + delta10, on11 + delta11, on12 + delta12])
		#currents = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12]
		currents = [243.]*12
		flyer.prop.overwriteCoils(ontimes.ctypes.data_as(c_double_p), offtimes.ctypes.data_as(c_double_p))
		flyer.preparePropagation(currents)
		flyer.propagate(0)

		pos = flyer.finalPositions[0]
		vel = flyer.finalVelocities[0]
		
		ind = np.where((pos[:, 2] > 268.) & (vel[:, 2] < 1.1*target_speed) & (vel[:, 2] > 0.9*target_speed))[0] # all particles that reach the end
		print 'good particles:', ind.shape[0]
		return -1.*ind.shape[0]

	initvals = {}
	for i in np.arange(12) + 1:
		initvals['on' + str(i)] = flyer.ontimes[i - 1]
		initvals['limit_on' + str(i)] = (0, 600)
		initvals['error_on' + str(i)] = 50
		initvals['delta' + str(i)] = flyer.offtimes[i - 1] - flyer.ontimes[i - 1]
		initvals['limit_delta' + str(i)] = (0, 85)
		initvals['error_delta' + str(i)] = 5
		#initvals['c' + str(i)] = 243.
		#initvals['limit_c' + str(i)] = (0, 300)
		#initvals['error_c' + str(i)] = 50

	m = Minuit(minimizer, **initvals)
	m.set_strategy(2)
	m.migrad(ncall=20)
	print m.values

def optimise_openopt(target_speed):
	from openopt import NLP
	def fitfun(gene):
		ontimes = np.require(gene[:12].copy(), requirements=['C', 'A', 'O', 'W'])
		offtimes = np.require(ontimes + gene[12:].copy(), requirements=['C', 'A', 'O', 'W'])
		currents = [243.]*12
		flyer.prop.overwriteCoils(ontimes.ctypes.data_as(c_double_p), offtimes.ctypes.data_as(c_double_p))
		flyer.preparePropagation(currents)
		flyer.propagate(0)

		pos = flyer.finalPositions[0]
		vel = flyer.finalVelocities[0]
		
		ind = np.where((pos[:, 2] > 268.) & (vel[:, 2] < 1.1*target_speed) & (vel[:, 2] > 0.9*target_speed))[0] # all particles that reach the end
		print 'good particles:', ind.shape[0]
		return -1.*ind.shape[0]

	initval = np.append(flyer.ontimes, flyer.offtimes - flyer.ontimes)
	lb = np.array(24*[0])
	ub = np.array(12*[600] + 12*[85])

	p = NLP(fitfun, initval, lb=lb, ub=ub)
	r = p.solve('bobyqa', plot=0)
	return r

def optimise_cma(): # no additional constraints beyond pulse duration
	import cma
	def fitfun(gene):
		#flyer.addParticles(checkSkimmer=True, NParticlesOverride=1200)
		offtimes = np.require(gene[:12].copy(), requirements=['C', 'A', 'O', 'W'])
		ontimes = np.require(offtimes - 0.1*gene[12:].copy(), requirements=['C', 'A', 'O', 'W'])
		currents = [243.]*12
		flyer.prop.overwriteCoils(ontimes.ctypes.data_as(c_double_p), offtimes.ctypes.data_as(c_double_p))
		flyer.preparePropagation(currents)
		fval = 0
		for i in np.arange(optStates):
			pos, vel, _ = flyer.propagate(i)
			ind = np.where((pos[:, 2] >= endplane) & (vel[:, 2] < 1.04*target_speed))[0] # all particles that reach the end
			#r = np.sqrt(pos[ind, 0]**2 + pos[ind, 1]**2)
			#fval = ind.shape[0]**2/r.mean()
			fval += ind.shape[0]
		#print 'good particles:', fval
		return -fval

	endplane = flyer.optimiserProps['position']
	target_speed = flyer.optimiserProps['targetSpeed']
	optStates = flyer.optimiserProps['optStates']
	maxPulseDuration = flyer.optimiserProps['maxPulseDuration']

	initval = np.append(flyer.offtimes, 10.*(flyer.offtimes - flyer.ontimes))
	sigma0 = 10.
	opts = {}
	opts['maxfevals'] = 12000
	opts['tolfun'] = 2
	opts['mindx'] = 0.5
	opts['bounds'] = [24*[0], 12*[800] + 12*[10.*maxPulseDuration]]
	opts['popsize'] = 22 # default is 13 for problem dimensionality 24; larger means more global search

	es = cma.CMAEvolutionStrategy(initval, sigma0, opts)
	nh = cma.NoiseHandler(es.N, [1, 1, 30])
	while not es.stop():
		X, fit_vals = es.ask_and_eval(fitfun, evaluations=nh.evaluations)
		es.tell(X, fit_vals)  # prepare for next iteration
		#es.sigma *= nh(X, fit_vals, fitfun, es.ask)  # see method __call__
		#es.countevals += nh.evaluations_just_done  # this is a hack, not important though
		es.disp()
		es.eval_mean(fitfun)
		print '========= evaluations: ', es.countevals, '========'
		print '========= current mean: ', es.fmean, '========'
		print es.mean
		print '========= current best: ', es.best.f, '========'
		print es.best.x

	print(es.stop())
	print 'mean ontimes: ', (es.result()[-2][:12] - es.result()[-2][12:]/10)  # take mean value, the best solution is totally off
	print 'mean durations: ', (es.result()[-2][12:]/10)  # take mean value, the best solution is totally off
	print 'best ontimes: ', (X[np.argmin(fit_vals)][:12] - X[np.argmin(fit_vals)][12:]/10)  # not bad, but probably worse than the mean
	print 'best durations: ', (X[np.argmin(fit_vals)][12:]/10)  # not bad, but probably worse than the mean

	return es



def optimise_cma_extra(): # with additional constraint of only one coil per box
	import cma
	def fitfun(gene):
		#flyer.addParticles(checkSkimmer=True, NParticlesOverride=1200)
		offtimes = np.require(gene[:12].copy(), requirements=['C', 'A', 'O', 'W'])
		ontimes = np.require(offtimes - 0.1*gene[12:].copy(), requirements=['C', 'A', 'O', 'W'])
		currents = [243.]*12
		flyer.prop.overwriteCoils(ontimes.ctypes.data_as(c_double_p), offtimes.ctypes.data_as(c_double_p))
		flyer.preparePropagation(currents)
		fval = 0
		for i in np.arange(optStates):
			pos, vel, _ = flyer.propagate(i)
			ind = np.where((pos[:, 2] >= endplane) & (vel[:, 2] < 1.04*target_speed))[0] # all particles that reach the end
			#r = np.sqrt(pos[ind, 0]**2 + pos[ind, 1]**2)
			#fval = ind.shape[0]**2/r.mean()
			fval += ind.shape[0]
		#print 'good particles:', fval
		return -fval

	endplane = flyer.optimiserProps['position']
	target_speed = flyer.optimiserProps['targetSpeed']
	optStates = flyer.optimiserProps['optStates']
	maxPulseDuration = 10.*flyer.optimiserProps['maxPulseDuration']

	initval = np.append(flyer.offtimes, 10.*(flyer.offtimes - flyer.ontimes))
	sigma0 = 8.
	opts = {}
	opts['maxfevals'] = 3000
	opts['tolfun'] = 2
	opts['mindx'] = 0.5
	opts['bounds'] = [24*[70], 12*[800] + 12*[maxPulseDuration]]
	opts['popsize'] = 22 # default is 13 for problem dimensionality 24; larger means more global search

	es = cma.CMAEvolutionStrategy(initval, sigma0, opts)
	nh = cma.NoiseHandler(es.N, [1, 1, 30])
	while not es.stop():
		X = []
		fit = []
		while len(X) < es.popsize:
			while True:
				x = es.ask(1)[0]
				res = np.where(x[2:12] - x[14:]/10. - x[:10] < 8)[0].shape
				if (res[0] == 0):
					X.append(x)
					fit.append(fitfun(x))
					break
		# X = es.ask(es.popsize) # this version shifts the bounds, but doesn't work well.
		# for x in X:
		# 	# first eliminate the ones where a later coil turn off before an earlier coil
		# 	origx = np.copy(x[:])

		# 	ind = np.where(x[1:12] < x[:11] + 8)[0]
		# 	while ind.shape[0] != 0:
		# 		x[ind + 1] = x[ind] + 9
		# 		ind = np.where(x[1:12] < x[:11] + 8)[0]
		# 	# and then the ones where more than two coils of the same supply are on at the same time
		# 	ind = np.where(x[2:12] - x[14:]/10. - x[:10] < 8)[0]
		# 	x[ind + 14] = (x[ind + 2] - x[ind] - 8)*10.
		# 	# # finally enforce bounds
		# 	# x[:12][x[:12] > 800] = 799.99
		# 	# x[12:][x[12:] > maxPulseDuration] = maxPulseDuration - 1.e-3
			
		# 	fit.append(fitfun(x))
		
		es.tell(X, fit)
		# X, fit_vals = es.ask_and_eval(fitfun, evaluations=nh.evaluations)
		# es.tell(X, fit_vals)  # prepare for next iteration
		#es.sigma *= nh(X, fit_vals, fitfun, es.ask)  # see method __call__
		#es.countevals += nh.evaluations_just_done  # this is a hack, not important though
		es.disp()
		es.eval_mean(fitfun)
		print '========= evaluations: ', es.countevals, '========'
		print '========= current mean: ', es.fmean, '========'
		print es.mean
		print '========= current best: ', es.best.f, '========'
		print es.best.x

	print(es.stop())
	print 'mean ontimes: ', (es.result()[-2][:12] - es.result()[-2][12:]/10)  # take mean value, the best solution is totally off
	print 'mean durations: ', (es.result()[-2][12:]/10)  # take mean value, the best solution is totally off
	print 'best ontimes: ', (X[np.argmin(fit)][:12] - X[np.argmin(fit)][12:]/10)  # not bad, but probably worse than the mean
	print 'best durations: ', (X[np.argmin(fit)][12:]/10)  # not bad, but probably worse than the mean

	return es



def optimise_cma_fixed(): # with fixed overlap time of 6 mus
	import cma
	ontimes = np.zeros((12,))
	def fitfun(gene):
		#flyer.addParticles(checkSkimmer=True, NParticlesOverride=1200)
		offtimes = np.require(gene[:12].copy(), requirements=['C', 'A', 'O', 'W'])
		ontimes[1:] = offtimes[:11] - 6
		ontimes[0] = offtimes[0] - 30
		currents = [243.]*12
		flyer.prop.overwriteCoils(ontimes.ctypes.data_as(c_double_p), offtimes.ctypes.data_as(c_double_p))
		flyer.preparePropagation(currents)
		fval = 0
		for i in np.arange(optStates):
			pos, vel, _ = flyer.propagate(i)
			ind = np.where((pos[:, 2] >= endplane) & (vel[:, 2] < 1.04*target_speed))[0] # all particles that reach the end
			#r = np.sqrt(pos[ind, 0]**2 + pos[ind, 1]**2)
			#fval = ind.shape[0]**2/r.mean()
			fval += ind.shape[0]
		#print 'good particles:', fval
		return -fval

	endplane = flyer.optimiserProps['position']
	target_speed = flyer.optimiserProps['targetSpeed']
	optStates = flyer.optimiserProps['optStates']
	maxPulseDuration = 10.*flyer.optimiserProps['maxPulseDuration']

	initval = flyer.offtimes[:]
	sigma0 = 10.
	opts = {}
	opts['maxfevals'] = 5000
	opts['tolfun'] = 2
	opts['mindx'] = 0.5
	opts['bounds'] = [12*[70], 12*[800]]
	opts['popsize'] = 22 # default is 13 for problem dimensionality 24; larger means more global search

	es = cma.CMAEvolutionStrategy(initval, sigma0, opts)
	nh = cma.NoiseHandler(es.N, [1, 1, 30])
	while not es.stop():
		X, fit = es.ask_and_eval(fitfun, evaluations=nh.evaluations)
		es.tell(X, fit)  # prepare for next iteration
		es.disp()
		es.eval_mean(fitfun)
		print '========= evaluations: ', es.countevals, '========'
		print '========= current mean: ', es.fmean, '========'
		print es.mean
		print '========= current best: ', es.best.f, '========'
		print es.best.x

        #np.savetxt(os.path.join(folder, 'mean.txt'), es.mean)
        #np.savetxt(os.path.join(folder, 'best.txt'), es.best.x)

	print(es.stop())
	ontimes = np.zeros(12)
	offtimes = es.result()[-2][:12]
	ontimes[1:] = offtimes[:11] - 6
	ontimes[0] = offtimes[0] - 30
	print 'mean ontimes: ', ontimes  # take mean value, the best solution is totally off
	print 'mean durations: ', offtimes-ontimes  # take mean value, the best solution is totally off
	np.savetxt(os.path.join(folder, 'mean.txt'), np.transpose((ontimes, offtimes , offtimes-ontimes)), fmt='%4.2f')
	offtimes = X[np.argmin(fit)][:12]
	ontimes[1:] = offtimes[:11] - 6
	ontimes[0] = offtimes[0] - 30
	print 'best ontimes: ', ontimes  # not bad, but probably worse than the mean
	print 'best durations: ', offtimes-ontimes  # not bad, but probably worse than the mean

	np.savetxt(os.path.join(folder, 'best.txt'), np.transpose((ontimes, offtimes , offtimes-ontimes)), fmt='%4.2f')
	return es

#folder = 'data/experiment_Ar/fixed50/460_380/'

#flyer = ZeemanFlyer(verbose=False)
#flyer.loadParameters(folder)
#flyer.addParticles(checkSkimmer=True, NParticlesOverride=3000)
#flyer.calculateCoilSwitching()
#flyer.loadBFields()
#flyer.preparePropagation()
#
#res = optimise_cma_fixed()
#print res

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('wd', 
			help='The working directory containing the config.info file.')
	parser.add_argument('-c', 
			help='Console mode. Will not produce plots on completion',
			action='store_true')
	parser.add_argument('-q', 
			help='Quiet mode. Does not produce any output; still log messages to file.', 
			action='store_true')
	args = parser.parse_args()
	folder = args.wd

	# Set up logging to console and file.
	logging.basicConfig(
			format='%(asctime)s - %(levelname)-8s : %(message)s',
			datefmt='%d%m%y %H:%M',
			filename=os.path.join(folder, 'log.txt'),
			filemode='w',
			level=logging.DEBUG)
	if not args.q:
		ch = logging.StreamHandler()
		ch.setLevel(logging.DEBUG)
		ch.setFormatter(logging.Formatter('%(levelname)-8s - %(message)s'))
		logging.getLogger().addHandler(ch)

	config_file = os.path.join(folder, 'config.info')
	logging.info('Running analysis in folder %s' % folder)
	if not os.path.exists(config_file):
		logging.critical('Config file not found at %s' % config_file)
		sys.exit(1)

	flyer = ZeemanFlyer()
	# Load parameters from config file and test that all is present and
	# correct. Exit if there is a problem.
	try:
		flyer.loadParameters(config_file)
	except RuntimeError as e:
		logging.critical(e)
		sys.exit(1)

	# Initialise the flyer calculation.  Generate the cloud of starting
	# positions and velocities
	flyer.addParticles(checkSkimmer=True)
	# Generate the switching sequence for the selected phase angle.
	flyer.calculateCoilSwitching(folder)
	# Load pre-calculated magnetic field mesh.
	flyer.loadBFields()
	# Transfer data to propagation library.
	flyer.preparePropagation()

	res = optimise_cma_fixed()
	print res
