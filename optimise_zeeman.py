from sim_zeeman import ZeemanFlyer
import numpy as np

import ctypes
from ctypes import c_double, c_uint, c_int 		# shorthand form for data types
c_double_p = ctypes.POINTER(c_double)			# pointer type


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

def optimise_cma(target_speed):
	import cma
	def fitfun(gene):
		#flyer.addParticles(checkSkimmer=True, NParticlesOverride=1200)
		ontimes = np.require(gene[:12].copy(), requirements=['C', 'A', 'O', 'W'])
		offtimes = np.require(ontimes + 0.1*gene[12:].copy(), requirements=['C', 'A', 'O', 'W'])
		currents = [243.]*12
		flyer.prop.overwriteCoils(ontimes.ctypes.data_as(c_double_p), offtimes.ctypes.data_as(c_double_p))
		flyer.preparePropagation(currents)
		fval = 0
		for i in np.arange(2):
			pos, vel, _ = flyer.propagate(i)
			ind = np.where((pos[:, 2] >= 255.) & (vel[:, 2] < 1.04*target_speed))[0] # all particles that reach the end
			#r = np.sqrt(pos[ind, 0]**2 + pos[ind, 1]**2)
			#fval = ind.shape[0]**2/r.mean()
			fval += ind.shape[0]
		#print 'good particles:', fval
		return -fval


	initval = np.append(flyer.ontimes, 10.*(flyer.offtimes - flyer.ontimes))
	sigma0 = 10.
	opts = {}
	opts['maxfevals'] = 5000
	opts['tolfun'] = 2
	opts['mindx'] = 0.5
	opts['bounds'] = [24*[0], 12*[600] + 12*[850]]

	es = cma.CMAEvolutionStrategy(initval, sigma0, opts)
	nh = cma.NoiseHandler(es.N, [1, 1, 30])
	while not es.stop():
		X, fit_vals = es.ask_and_eval(fitfun, evaluations=nh.evaluations)
		es.tell(X, fit_vals)  # prepare for next iteration
		#es.sigma *= nh(X, fit_vals, fitfun, es.ask)  # see method __call__
		#es.countevals += nh.evaluations_just_done  # this is a hack, not important though
		es.disp()
		print '========= evaluations: ', es.countevals, '========'
		print '========= current mean: ', fitfun(es.mean), '========'
		print es.mean
		print '========= current best: ', es.best.f, '========'
		print es.best.x
		# nh.maxevals = ...  it might be useful to start with smaller values and then increase

	print(es.stop())
	print 'mean ontimes: ', (es.result()[-2][:12])  # take mean value, the best solution is totally off
	print 'mean durations: ', (es.result()[-2][12:]/10)  # take mean value, the best solution is totally off
	print 'best ontimes: ', (X[np.argmin(fit_vals)][:12])  # not bad, but probably worse than the mean
	print 'best durations: ', (X[np.argmin(fit_vals)][12:]/10)  # not bad, but probably worse than the mean




	#res = cma.fmin(fitfun, initval, 10, options=opts)
	#cma.plot()
	#return res
	return es



folder = 'easy/opt_500_450/'
target_speed = 0.45

flyer = ZeemanFlyer(verbose=False)
flyer.loadParameters(folder)
flyer.addParticles(checkSkimmer=True, NParticlesOverride=3000)
flyer.calculateCoilSwitching()
flyer.loadBFields()
flyer.preparePropagation()



#r = optimise_openopt(0.288)
#print r.xf
# optimise_minuit()
#r = optimise_cma(0.288)
#print r[0]

res = optimise_cma(target_speed)
print res
raise RuntimeError


# show results for all zeeman states and more particles
flyer.addParticles(checkSkimmer=True, NParticlesOverride=500000)

# ontimes = []
# offtimes = []
# for i in range(1, 13):
# 	ontimes.append(m.values['on' + str(i)])
# 	offtimes.append(m.values['delta' + str(i)])
# offtimes = np.array(offtimes)
# ontimes = np.array(ontimes)
# offtimes += ontimes
# flyer.prop.overwriteCoils(ontimes.ctypes.data_as(c_double_p), offtimes.ctypes.data_as(c_double_p))
# currents = [243.]*12
# flyer.preparePropagation(currents)
	
totalGood1 = 0
allvel1 = []
alltimes1 = []
for z in range(4):
	flyer.propagate(z)
	vel = flyer.finalVelocities[z]
	pos = flyer.finalPositions[z]
	times = flyer.finalTimes[z]
	ind = np.where((pos[:, 2] > 255.)) # all particles that reach the end
	if z in [0, 1]:
		plt.figure(0)
		plt.hist(vel[ind, 2].flatten(), bins = np.arange(0, 1, 0.005), histtype='step', color='r', label='optimised')
		plt.figure(1)
		plt.hist(times[ind], bins=np.linspace(200, 1200, 101), histtype='step', color='r', label='optimised')
	allvel1.extend(vel[ind, 2].flat)
	alltimes1.extend(times[ind])
	indg1 = np.where((pos[:, 2] > 255.) & (vel[:, 2] < 1.1*target_speed) & (vel[:, 2] > 0.9*target_speed))[0]
	print indg1.shape[0]
	totalGood1 += indg1.shape[0]
	#plt.hist(pos[indg1, 0], histtype='step', label='optimized', normed=True)

	np.save(folder + 'finalpos' + str(z) + '.npy', pos)
	np.save(folder + 'finalvel' + str(z) + '.npy', vel)
	np.save(folder + 'finaltimes' + str(z) + '.npy', times)

np.save(folder + 'initialpos.npy', flyer.initialPositions)
np.save(folder + 'initialvel.npy', flyer.initialVelocities)

raise RuntimeError

# and do if for the default version with fixed phase
flyer.propagationProps['phase'] = 90
flyer.calculateCoilSwitching()
flyer.preparePropagation()
totalGood2 = 0
allvel2 = []
alltimes2 = []
for z in range(4):
	flyer.propagate(z)
	vel2 = flyer.finalVelocities[z]
	pos2 = flyer.finalPositions[z]
	times2 = flyer.finalTimes[z]
	ind2 = np.where((pos2[:, 2] > 268.)) # all particles that reach the end
	if z in [0, 1]:
		plt.figure(0)
		plt.hist(vel2[ind2, 2].flatten(), bins = np.arange(0, 1, 0.005), histtype='step', color='b', label='default')
		plt.figure(1)
		plt.hist(times2[ind2], bins=np.linspace(200, 1200, 101), histtype='step', color='b', label='default')
	allvel2.extend(vel2[ind2, 2].flat)
	alltimes2.extend(times2[ind2])
	indg2 = np.where((pos2[:, 2] > 268.) & (vel2[:, 2] < 1.1*0.25) & (vel2[:, 2] > 0.9*0.25))[0]
	totalGood2 += indg2.shape[0]
	print indg2.shape[0]
	#plt.hist(pos2[indg2, 0], histtype='step', label='default', normed=True)


plt.figure()
plt.hist(allvel1, bins = np.arange(0, 1, 0.005), histtype='step', color='r', label='optimised')
plt.hist(allvel2, bins = np.arange(0, 1, 0.005), histtype='step', color='b', label='default')
plt.legend()
plt.figure()
plt.hist(alltimes1, bins=np.linspace(200, 1200, 101), histtype='step', color='r', label='optimised')
plt.hist(alltimes2, bins=np.linspace(200, 1200, 101), histtype='step', color='b', label='default')
plt.show()
