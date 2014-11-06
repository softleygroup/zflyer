# Zeeman Flyer
#
#  # Introduction
#    
#	This is a python wrapper for propagator_particle.c, the library used
#	for efficient propagation of particles through the zeeman decelerator.
#	The wrapper is responsible for
#		- reading of settings from the config file
#		- creating initial positions and velocities for the particle bunch
#		- loading magnetic field values from disk
#		- passing field values and parameters to the propagator
#		  (with all memory management being done in python)
#		- starting the simulation, and providing an interface to the results
#  
# @author Atreju Tauschinsky
# @copyright Copyright 2014 University of Oxford.



import numpy as np 								# used for numeric arrays, and passing data to c library 
from numpy import sqrt, pi 						# shorthand form for these functions
import time 									# only used to time execution
from matplotlib import pyplot as plt 			# only used if executed as standalone app, to display simulation results
import os 										# used for compilation of propagator library
from subprocess import call 					# also used for compilation
from ConfigParser import SafeConfigParser 		# reading config file

import ctypes 									# used to interface with the c library (propagator_particle.c)
from ctypes import c_double, c_uint, c_int 		# shorthand form for data types
c_double_p = ctypes.POINTER(c_double)			# pointer type

np.random.seed(1)								# initialize random number generator

kB = 1.3806504E-23								# Boltzmann constant (in J/K)
muB = 9.2740154E-24 							# Bohr magneton in J/T
HBAR = 1.054571628E-34 							# Planck constant (in Js)
A = 1420405751.768*2*pi/HBAR 					# in 1/((s^2)*J)


class ZeemanFlyer(object):
	def __init__(self, verbose=True):
		self.verbose = verbose
		
		# create dictionaries for final results
		self.finalPositions = {}
		self.finalVelocities = {}
		self.finalTimes = {}
		
		# load C library
		# and recompile if necessary
		target = 'propagator_particle'
		if not os.path.exists(target + '.so') or os.stat(target + '.c').st_mtime > os.stat(target + '.so').st_mtime: # we need to recompile
			COMPILE = ['PROF'] # 'PROF', 'FAST', both or neither
			# include branch prediction generation. compile final version with only -fprofile-use
			commonopts = ['-c', '-fPIC', '-Ofast', '-march=native', '-std=c99', '-Wall', '-Wextra', '-Wconversion', '-Wshadow', '-Wcast-qual', '-Werror', '-fno-exceptions', '-fomit-frame-pointer']
			profcommand = ['gcc', '-fprofile-arcs', '-fprofile-generate', target + '.c']
			profcommand[1:1] =  commonopts
			fastcommand = ['gcc', '-fprofile-use', target + '.c']
			fastcommand[1:1] = commonopts
			debugcommand = ['gcc', target + '.c']
			debugcommand[1:1] = ['-c', '-fPIC', '-O0', '-g', '-std=c99']

			print
			print
			print '==================================='
			print 'compilation target: ', target
			if 'PROF' in COMPILE:
				if call(profcommand) != 0:
					raise RuntimeError("COMPILATION FAILED!")
				call(['gcc', '-shared', '-fprofile-generate', target + '.o', '-o', target + '.so'])
				print 'COMPILATION: PROFILING RUN'
			if 'FAST' in COMPILE:
				if call(fastcommand) != 0:
					raise RuntimeError("COMPILATION FAILED!")
				call(['gcc', '-shared', target + '.o', '-o', target + '.so'])
				print 'COMPILATION: FAST RUN'
			if 'DEBUG' in COMPILE:
				if call(debugcommand) != 0:
					raise RuntimeError("COMPILATION FAILED!")
				call(['gcc', '-shared', target + '.o', '-o', target + '.so'])
				print 'COMPILATION: DEBUG RUN'
			if not ('PROF' in COMPILE or 'FAST' in COMPILE or 'DEBUG' in COMPILE):
				print 'DID NOT RECOMPILE C SOURCE'
			print '==================================='
			print
			print
		
		# define interface to propagator library
		self.prop = ctypes.cdll.LoadLibrary('./' + target + '.so')
		self.prop.setSynchronousParticle.argtypes = [c_double, c_double_p, c_double_p]
		self.prop.setSynchronousParticle.restype = None
		self.prop.setBFields.argtypes = [c_double_p, c_double_p, c_double_p, c_double_p, c_double, c_double, c_double, c_int, c_int, c_int]
		self.prop.setBFields.restype = None
		self.prop.setCoils.argtypes = [c_double_p, c_double, c_double, c_int]
		self.prop.setCoils.restype = None
		self.prop.setSkimmer.argtypes = [c_double, c_double, c_double, c_double]
		self.prop.setSkimmer.restype = None
		self.prop.doPropagate.argtypes = [c_double_p, c_double_p, c_double_p, c_int, c_int]
		self.prop.doPropagate.restype = None
		self.prop.setTimingParameters.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double]
		self.prop.setTimingParameters.restype = None
		self.prop.calculateCoilSwitching.argtypes = [c_double, c_double, c_double_p, c_double_p, c_double_p, c_double_p]
		self.prop.calculateCoilSwitching.restype = int
		self.prop.precalculateCurrents.argtypes = [c_double_p, c_double_p]
		self.prop.precalculateCurrents.restype = int
		self.prop.setPropagationParameters.argtypes = [c_double, c_double, c_int]
		self.prop.setPropagationParameters.restype = None
		self.prop.overwriteCoils.argtypes = [c_double_p, c_double_p]
		self.prop.overwriteCoils.restype = None
	
	def loadParameters(self, folder):
		# function to load parameters from file, and make them easily accessible to the class
		def configToDict(items):
			# sub-function turning a set of config entries to a dict, 
			# automatically converting strings to numbers where possible
			d = {}						# initialize empty dict
			for k, v in items:			# traverse all settings
				try:
					d[k] = eval(v)		# try to evaluate (essentially turning strings to numbers, but allowing things like multiplication in the config file)
				except ValueError:		# if this goes wrong for some reason we simply keep this entry as a string
					if self.verbose:
						print 'Could not parse option "', k, '", keeping value "', v, '" as string'
					d[k] = v
			return d
		

		config = SafeConfigParser()
		config.optionxform = lambda option : option 						# no processing in parser, in particular no change of capitalisation
		config.read(folder + 'config.info')									# read config
		self.particleProps = configToDict(config.items('PARTICLE'))			# here we read the different sections, turning the entries from each section 
		self.bunchProps = configToDict(config.items('BUNCH'))				# into a dictionary that we can easily access
		self.propagationProps = configToDict(config.items('PROPAGATION'))
		self.coilProps = configToDict(config.items('COILS'))
		self.skimmerProps = configToDict(config.items('SKIMMER'))
		self.detectionProps = configToDict(config.items('DETECTION'))
	
	def addParticles(self, includeSyn=True, checkSkimmer=False, NParticlesOverride = None):
		# add particles with position and velocity spread given by settings
		# create random initial positions and velocities;
		# if checkSkimmer = True, we also immediately test whether the particles would make it through
		# the skimmer, and discard those that won't

		if NParticlesOverride is not None:
			self.bunchProps['NParticles'] = NParticlesOverride		# allow manually overriding the particle number specified in the config
		
		nGenerated = 0												# keep track of total number of generated particle
		nGeneratedGood = 0 											# number of particles passing through the skimmer
		
		# make the parameters used here available in shorthand
		nParticles = self.bunchProps['NParticles']
		v0 = self.bunchProps['v0']
		x0 = self.bunchProps['x0']
		radius = self.bunchProps['radius']
		length = self.bunchProps['length']
		TRadial = self.bunchProps['TRadial']
		TLong = self.bunchProps['TLong']
		mass = self.particleProps['mass']
		skimmerDist = self.skimmerProps['position']
		skimmerRadius = self.skimmerProps['radius']
		
		if includeSyn:
			# if includeSyn == True, the first particle in the array
			# will be the synchronous particle as given in the config file
			initialPositions = np.array([x0])
			initialVelocities = np.array([v0])
			nGenerated += 1
			nGeneratedGood += 1
		else:
			# otherwise we still have to initialise the arrays, but they are empty now (right shape only).
			initialPositions = np.zeros((0, 3))
			initialVelocities = np.zeros((0, 3))
		
		while nGeneratedGood < nParticles:							# keep going as long as we don't have as many good particles as we need
			nParticlesToSim = nParticles - nGeneratedGood			# we'll create the difference between the number of particles we need and the number of particles we have
			# (a) for positions
			# random uniform distribution within a cylinder
			# r0 and phi0 span up a disk; z0 gives the height
			r0_rnd = sqrt(np.random.uniform(0, radius, nParticlesToSim))*sqrt(radius)
			phi0_rnd = np.random.uniform(0, 2*pi, nParticlesToSim)
			
			# transformation polar coordinates <--> cartesian coordinates
			x0_rnd = r0_rnd*np.cos(phi0_rnd)
			y0_rnd = r0_rnd*np.sin(phi0_rnd)
			
			# in z direction it's just a box
			z0_rnd = 5. + np.random.uniform(-length/2, length/2, nParticlesToSim)
			
			# (b) for velocities
			# normally distributed random numbers
			# if you want to generate normally distributed vx-vy random numbers
			# that are centered at vx = 0 mm/mus and vy = 0 mm/mus, use bivar_rnd = 1
			# else use bivar_rnd = 0
			sigmavr0 = sqrt(kB*TRadial/mass)/1000 # standard deviation self.vr0 component
			
			# normally distributed random numbers centered at 0 mm/mus
			# generate bi(multi)variate Gaussian data for vx and vy
			# rand_data = mvnrnd(mu, sigma,num of data)
			muvr = [0, 0] # mean values centered around 0 mm/mus
			# sigma1 = [1 0  # covariance matrix, diagonals = variances of each variable,
			#          0 1]  # off-diagonals = covariances between the variables
			# if no correlation, then off-diagonals = 0 and Sigma can also be written as a row array
			SigmaM = [[sigmavr0**2, 0], [0, sigmavr0**2]]
			vx0_rnd, vy0_rnd = np.random.multivariate_normal(muvr, SigmaM, [nParticlesToSim]).T
			
			sigmavz0 = sqrt(kB*TLong/mass)/1000 # standard deviation vz0 component
			vz0_rnd = np.random.normal(v0[2], sigmavz0, nParticlesToSim)
			#vz0_rnd = np.random.uniform(synVel[2] - sigmavz0, synVel[2] + sigmavz0, nParticlesToSim)
			
			if checkSkimmer:
				xatskimmer = x0_rnd + (vx0_rnd/vz0_rnd)*(skimmerDist-z0_rnd)
				yatskimmer = y0_rnd + (vy0_rnd/vz0_rnd)*(skimmerDist-z0_rnd)
				ratskimmer = sqrt(xatskimmer**2 + yatskimmer**2)
				ts = np.where(ratskimmer<=skimmerRadius)[0]
			else:
				ts = slice(0, x0_rnd.shape[0])

			
			initialPositions = np.vstack((initialPositions, np.array([x0_rnd[ts], y0_rnd[ts], z0_rnd[ts]]).T))
			initialVelocities = np.vstack((initialVelocities, np.array([vx0_rnd[ts], vy0_rnd[ts], vz0_rnd[ts]]).T))
			
			nGenerated += nParticlesToSim
			nGeneratedGood  = initialPositions.shape[0]
		
		self.initialPositions = np.array(initialPositions)
		self.initialVelocities = np.array(initialVelocities)
		
		if self.verbose:
			skimmerloss_no = 100.*nGeneratedGood/nGenerated
			print 'particles coming out of the skimmer (in percent): %.2f\n' % skimmerloss_no
	
	def addSavedParticles(self, folder):
		A = np.genfromtxt(folder + 'init_cond.txt', dtype=np.float)
		self.initialPositions = A[:, :3]
		self.initialVelocities=  A[:, 3:]/1000.
	
	def calculateCoilSwitching(self, phaseAngleOverride = None):
		if phaseAngleOverride is not None:
			self.propagationProps['phase'] = phaseAngleOverride
		
		bunchpos = np.array(self.bunchProps['x0'])
		bunchspeed = np.array(self.bunchProps['v0'])
		self.prop.setSynchronousParticle(self.particleProps['mass'], bunchpos.ctypes.data_as(c_double_p), bunchspeed.ctypes.data_as(c_double_p))
		
		coilpos = np.array(self.coilProps['position'])
		self.prop.setCoils(coilpos.ctypes.data_as(c_double_p), self.coilProps['radius'], self.detectionProps['position'], self.coilProps['NCoils'])
		
		self.prop.setTimingParameters(self.coilProps['H1'], self.coilProps['H2'], self.coilProps['ramp1'], self.coilProps['timeoverlap'], self.coilProps['rampcoil'], self.coilProps['maxPulseLength'])
		
		## B field along z axis
		# from FEMM or Comsol file
		 
		# analytic solution
		bfieldz = np.require(np.genfromtxt('sim_files/bonzaxis.txt', delimiter='\t'), requirements=['C', 'A', 'O', 'W'])
		# bfieldz = np.genfromtxt('sim_files/baxis_Zurich.txt', delimiter='\t') # Zurich Comsol calculation
		

		if self.propagationProps['phase'] == None:
			self.ontimes = self.propagationProps['ontimes']
			self.offtimes = self.propagationProps['ontimes'] + self.propagationProps['durations']
			self.prop.overwriteCoils(self.ontimes.ctypes.data_as(c_double_p), self.offtimes.ctypes.data_as(c_double_p))
		else:
			currents = self.coilProps['current']
			self.ontimes = np.zeros(self.coilProps['NCoils'], dtype=np.double)
			self.offtimes = np.zeros(self.coilProps['NCoils'], dtype=np.double)	
			
			if not self.prop.calculateCoilSwitching(self.propagationProps['phase'], self.propagationProps['timestepPulse'], bfieldz.ctypes.data_as(c_double_p), self.ontimes.ctypes.data_as(c_double_p), self.offtimes.ctypes.data_as(c_double_p), currents.ctypes.data_as(c_double_p)) == 0:
				raise RuntimeError("Error while calculating coil switching times")
	
	def resetParticles(self, initialZeemanState):
		self.finalPositions[initialZeemanState] = np.require(self.initialPositions.copy(), requirements=['C', 'A', 'O', 'W'])
		self.finalVelocities[initialZeemanState] = np.require(self.initialVelocities.copy(), requirements=['C', 'A', 'O', 'W'])
		
		self.nParticles = self.initialPositions.shape[0]
		
		self.finalTimes[initialZeemanState] = np.require(np.empty((self.nParticles, )))

		return 0
	
	def loadBFields(self):
		## B field coil
		Bz_n = np.genfromtxt('sim_files/Bz_n.txt', delimiter='\t').T # contains Bz field as a grid with P(r,z) (from analytic solution)
		Br_n = np.genfromtxt('sim_files/Br_n.txt', delimiter='\t').T # contains Br field as a grid with P(r,z) (from analytic solution)
		
		self.raxis = np.genfromtxt('sim_files/raxis.txt', delimiter='\t') # raxis as one column
		self.zaxis = np.genfromtxt('sim_files/zaxis.txt', delimiter='\t') # zaxis as one row
		
		zdist = self.zaxis[1] - self.zaxis[0] # spacing B field z axis (in mm)
		rdist = self.raxis[1] - self.raxis[0] # spacing B field r axis (in mm)
		bzextend = -self.zaxis[0] # dimension B field along decelerator z axis (in mm)
		sizB = Bz_n.shape[1]
		
		self.Bz_n_flat = Bz_n.flatten()
		self.Br_n_flat = Br_n.flatten()
		
		sizZ = self.zaxis.shape[0]
		sizR = self.raxis.shape[0]
		
		self.prop.setBFields(self.Bz_n_flat.ctypes.data_as(c_double_p), self.Br_n_flat.ctypes.data_as(c_double_p), self.zaxis.ctypes.data_as(c_double_p), self.raxis.ctypes.data_as(c_double_p), bzextend, zdist, rdist, sizZ, sizR, sizB)
	
	def preparePropagation(self, overwrite_currents=None):
		sradius = self.skimmerProps['radius']
		sbradius = self.skimmerProps['backradius']
		slength = self.skimmerProps['length']
		spos = self.skimmerProps['position']
		alpha = np.arctan((sbradius - sradius)/slength)
		self.prop.setSkimmer(spos, slength, sradius, alpha)
		
		self.coilpos = np.array(self.coilProps['position']) - 15.5 # zshiftdetect??
		cradius = self.coilProps['radius']
		nCoils = int(self.coilProps['NCoils'])
		self.prop.setCoils(self.coilpos.ctypes.data_as(c_double_p), cradius, self.detectionProps['position'], nCoils)
		
		tStart = self.propagationProps['starttime']
		tStop = self.propagationProps['stoptime']
		dT =  self.propagationProps['timestep']
		
		self.prop.setPropagationParameters(tStart, dT, (tStop - tStart)/dT)
		self.current_buffer = np.zeros(((tStop - tStart)/dT, nCoils), dtype=np.double)

		if overwrite_currents is None:
			self.currents = self.coilProps['current']
		else:
			self.currents = np.array(overwrite_currents)
		if not self.prop.precalculateCurrents(self.current_buffer.ctypes.data_as(c_double_p), self.currents.ctypes.data_as(c_double_p)) == 0:
			raise RuntimeError("Error precalculating currents!")
		
	def propagate(self, zeemanState = -1):
		self.resetParticles(zeemanState)				
		pos = self.finalPositions[zeemanState]
		vel = self.finalVelocities[zeemanState]
		times = self.finalTimes[zeemanState]
		self.prop.doPropagate(pos.ctypes.data_as(c_double_p), vel.ctypes.data_as(c_double_p),  times.ctypes.data_as(c_double_p), flyer.nParticles, zeemanState)

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
		ontimes = np.require(gene[:12].copy(), requirements=['C', 'A', 'O', 'W'])
		offtimes = np.require(ontimes + 0.1*gene[12:].copy(), requirements=['C', 'A', 'O', 'W'])
		currents = [243.]*12
		flyer.prop.overwriteCoils(ontimes.ctypes.data_as(c_double_p), offtimes.ctypes.data_as(c_double_p))
		flyer.preparePropagation(currents)
		flyer.propagate(0)

		pos = flyer.finalPositions[0]
		vel = flyer.finalVelocities[0]
		
		ind = np.where((pos[:, 2] > 268.) & (vel[:, 2] < 1.1*target_speed) & (vel[:, 2] > 0.9*target_speed))[0] # all particles that reach the end
		print 'good particles:', ind.shape[0]
		return -1.*ind.shape[0]

	initval = np.append(flyer.ontimes, 10.*(flyer.offtimes - flyer.ontimes))
	print initval
	lb = np.array(24*[0])
	ub = np.array(12*[600] + 12*[850])

	opts = {}
	opts['maxfevals'] = 10000
	opts['tolfun'] = 2
	opts['mindx'] = 0.5

	res = cma.fmin(fitfun, initval, 50, options=opts)
	cma.plot()
	return res


if __name__ == '__main__':

	folder = 'test_60/'
	flyer = ZeemanFlyer()
	flyer.loadParameters(folder)
	flyer.addParticles(checkSkimmer=True)
	#flyer.addSavedParticles('./output_500_60_1/')
	flyer.calculateCoilSwitching()
	flyer.loadBFields()
	flyer.preparePropagation()

	#r = optimise_openopt(0.288)
	#print r.xf
	# optimise_minuit()
	#r = optimise_cma(0.288)
	#print r[0]
	#raise RuntimeError
	

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
		ind = np.where((pos[:, 2] > 268.)) # all particles that reach the end
		if z in [0, 1]:
			plt.figure(0)
			plt.hist(vel[ind, 2].flatten(), bins = np.arange(0, 1, 0.005), histtype='step', color='r', label='optimised')
			plt.figure(1)
			plt.hist(times[ind], bins=np.linspace(200, 1200, 101), histtype='step', color='r', label='optimised')
		allvel1.extend(vel[ind, 2].flat)
		alltimes1.extend(times[ind])
		indg1 = np.where((pos[:, 2] > 268.) & (vel[:, 2] < 1.1*0.25) & (vel[:, 2] > 0.9*0.25))[0]
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
	flyer.propagationProps['phase'] = 72
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

	raise RuntimeError
