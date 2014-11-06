import numpy as np
from numpy import sqrt, pi
import time
from matplotlib import pyplot as plt
import os
from ConfigParser import SafeConfigParser

import ctypes
from ctypes import c_double, c_uint, c_int
c_double_p = ctypes.POINTER(c_double)

np.random.seed(1)

kB = 1.3806504E-23 # Boltzmann constant (in J/K)
muB = 9.2740154E-24 # Bohr magneton in J/T
HBAR = 1.054571628E-34 # Planck constant (in Js)
A = 1420405751.768*2*pi/HBAR # in 1/((s^2)*J)


class ZeemanFlyer(object):
	def __init__(self, verbose=True):
		self.verbose = verbose
		
		self.finalPositions = {}
		self.finalVelocities = {}
		self.finalTimes = {}
		
		# load C library
		# and recompile if necessary
		from subprocess import call
		target = 'propagator_particle'
		if not os.path.exists(target + '.so') or os.stat(target + '.c').st_mtime > os.stat(target + '.so').st_mtime: # we need to recompile
			COMPILE = ['PROF'] # 'PROF', 'FAST', both or neither
			# include branch prediction generation. compile final version with only -fprofile-use
			commonopts = ['-c', '-fPIC', '-Ofast', '-march=native', '-std=c99', '-fopenmp', '-Wall', '-fno-exceptions', '-fomit-frame-pointer']
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
				call(['gcc', '-shared', '-fopenmp', '-lgomp', '-fprofile-generate', target + '.o', '-o', target + '.so'])
				print 'COMPILATION: PROFILING RUN'
			if 'FAST' in COMPILE:
				if call(fastcommand) != 0:
					raise RuntimeError("COMPILATION FAILED!")
				call(['gcc', '-shared', '-lgomp', target + '.o', '-o', target + '.so'])
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
		
		self.prop = ctypes.cdll.LoadLibrary('./' + target + '.so')
		self.prop.setSynchronousParticle.argtypes = [c_double, c_double_p, c_double_p]
		self.prop.setSynchronousParticle.restype = None
		self.prop.setBFields.argtypes = [c_double_p, c_double_p, c_double_p, c_double_p, c_double, c_double, c_double, c_uint, c_uint, c_uint]
		self.prop.setBFields.restype = None
		self.prop.setCoils.argtypes = [c_double_p, c_double, c_double, c_uint, c_double]
		self.prop.setCoils.restype = None
		self.prop.setSkimmer.argtypes = [c_double, c_double, c_double, c_double]
		self.prop.setSkimmer.restype = None
		self.prop.doPropagate.argtypes = [c_double_p, c_double_p, c_double_p, c_uint, c_int]
		self.prop.doPropagate.restype = None
		self.prop.setTimingParameters.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double]
		self.prop.setTimingParameters.restype = None
		self.prop.calculateCoilSwitching.argtypes = [c_double, c_double, c_double_p, c_double_p, c_double_p]
		self.prop.calculateCoilSwitching.restype = int
		self.prop.precalculateCurrents.argtypes = [c_double_p]
		self.prop.precalculateCurrents.restype = int
		self.prop.setPropagationParameters.argtypes = [c_double, c_double, c_double]
		self.prop.setPropagationParameters.restype = None
	
	def loadParameters(self, folder):
		def configToDict(items):
			d = {}
			for k, v in items:
				try:
					d[k] = eval(v)
				except ValueError:
					if self.verbose:
						print 'Could not parse option "', k, '", keeping value "', v, '" as string'
					d[k] = v
			return d
			
		config = SafeConfigParser()
		config.optionxform = lambda option : option
		config.read(folder + 'config.info')
		self.particleProps = configToDict(config.items('PARTICLE'))
		self.bunchProps = configToDict(config.items('BUNCH'))
		self.propagationProps = configToDict(config.items('PROPAGATION'))
		self.coilProps = configToDict(config.items('COILS'))
		self.skimmerProps = configToDict(config.items('SKIMMER'))
		self.detectionProps = configToDict(config.items('DETECTION'))
	
	def addParticles(self, includeSyn=True, checkSkimmer=False, NParticlesOverride = None):
		if NParticlesOverride is not None:
			self.bunchProps['NParticles'] = NParticlesOverride
		
		nGenerated = 0
		nGeneratedGood = 0
		
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
			# synchronous particle (set by default)
			initialPositions = np.array([x0])
			initialVelocities = np.array([v0])
			nGenerated += 1
			nGeneratedGood += 1
		else:
			initialPositions = np.zeros((0, 3))
			initialVelocities = np.zeros((0, 3))
		
		while nGeneratedGood < nParticles:
			nParticlesToSim = nParticles - nGeneratedGood
			# (a) for positions
			# random uniform distribution within a cylinder
			# r0 and phi0 span up a disk; z0 gives the height
			r0_rnd = sqrt(np.random.uniform(0, radius, nParticlesToSim))*sqrt(radius)
			phi0_rnd = np.random.uniform(0, 2*pi, nParticlesToSim)
			
			# transformation polar coordinates <--> cartesian coordinates
			# [x,y] = pol2cart(phi,r)
			# [x0_rnd,y0_rnd] = pol2cart(phi0_rnd,r0_rnd)
			x0_rnd = r0_rnd*np.cos(phi0_rnd)
			y0_rnd = r0_rnd*np.sin(phi0_rnd)
			
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
		self.prop.setCoils(coilpos.ctypes.data_as(c_double_p), self.coilProps['radius'], self.detectionProps['position'], self.coilProps['NCoils'], self.coilProps['current'])
		
		self.prop.setTimingParameters(self.coilProps['H1'], self.coilProps['H2'], self.coilProps['ramp1'], self.coilProps['timeoverlap'], self.coilProps['rampcoil'], self.coilProps['maxPulseLength'])
		
		## B field along z axis
		# from FEMM or Comsol file
		 
		# analytic solution
		bfieldz = np.require(np.genfromtxt('sim_files/bonzaxis.txt', delimiter='\t'), requirements=['C', 'A', 'O', 'W'])
		# bfieldz = np.genfromtxt('sim_files/baxis_Zurich.txt', delimiter='\t') # Zurich Comsol calculation
		
		self.ontimes = np.zeros(self.coilProps['NCoils'], dtype=np.double)
		self.offtimes = np.zeros(self.coilProps['NCoils'], dtype=np.double)	
		if not self.prop.calculateCoilSwitching(self.propagationProps['phase'], self.propagationProps['timestepPulse'], bfieldz.ctypes.data_as(c_double_p), self.ontimes.ctypes.data_as(c_double_p), self.offtimes.ctypes.data_as(c_double_p)) == 0:
			raise RuntimeError("Error while calculating coil switching times")
	
	def resetParticles(self, initialZeemanState):
		self.finalPositions[initialZeemanState] = np.require(self.initialPositions.copy(), requirements=['C', 'A', 'O', 'W'])
		self.finalVelocities[initialZeemanState] = np.require(self.initialVelocities.copy(), requirements=['C', 'A', 'O', 'W'])
		
		self.nParticles = self.initialPositions.shape[0]
		
		self.finalTimes[initialZeemanState] = np.require(np.empty((self.nParticles, )))
		
		#self.currentZeemanState = initialZeemanState*np.ones((self.nParticles,1))
		#self.currentTime = np.zeros((self.nParticles, 1))
		#self.currentIndex = np.arange(self.nParticles)
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
	
	def preparePropagation(self):
		sradius = self.skimmerProps['radius']
		sbradius = self.skimmerProps['backradius']
		slength = self.skimmerProps['length']
		spos = self.skimmerProps['position']
		alpha = np.arctan((sbradius - sradius)/slength)
		self.prop.setSkimmer(spos, slength, sradius, alpha)
		
		self.coilpos = np.array(self.coilProps['position']) - 18.6 # zshiftdetect??
		cradius = self.coilProps['radius']
		nCoils = int(self.coilProps['NCoils'])
		current = self.coilProps['current']			# the current at which we want to run the coils in the simulation
		self.prop.setCoils(self.coilpos.ctypes.data_as(c_double_p), cradius, self.detectionProps['position'], nCoils, current)
		
		tStart = self.propagationProps['starttime']
		tStop = self.propagationProps['stoptime']
		dT =  self.propagationProps['timestep']
		
		self.prop.setPropagationParameters(tStart, dT, (tStop - tStart)/dT)
		
		self.currents = np.zeros(((tStop - tStart)/dT, nCoils), dtype=np.double)
		if not self.prop.precalculateCurrents(self.currents.ctypes.data_as(c_double_p)) == 0:
			raise RuntimeError("Error precalculating currents!")
		
	def propagate(self, zeemanState = -1):
		self.resetParticles(zeemanState)
		
		pos = self.finalPositions[zeemanState]
		vel = self.finalVelocities[zeemanState]
		times = self.finalTimes[zeemanState]
		
		self.prop.doPropagate(pos.ctypes.data_as(c_double_p), vel.ctypes.data_as(c_double_p),  times.ctypes.data_as(c_double_p), flyer.nParticles, zeemanState)
	
if __name__ == '__main__':
	folder = 'test_omp/'
	flyer = ZeemanFlyer()
	flyer.loadParameters(folder)
	flyer.addParticles(checkSkimmer=True)	# bunch.addSavedParticles('./output_600_21_0/')
	
	flyer.loadBFields()	
	flyer.calculateCoilSwitching()
	
	flyer.preparePropagation()
	
	print 'starting propagation'
	start = time.time()
	for z in range(0, 4): # simulate all possible zeeman states, -1 is decelerator off
		flyer.propagate(z)
		print flyer.finalPositions[0][0]
		print flyer.finalVelocities[0][0]	
	print 'time for propagation was', time.time()-start, 'seconds'	
	
	plt.figure(0)
	plt.title('final velocities')
	
	for z in range(0, 4): # analysis for all zeeman states, including gaspulse
		pos = flyer.finalPositions[z]
		vel = flyer.finalVelocities[z]
		times = flyer.finalTimes[z]
		print pos[0]
		print vel[0]
		
		np.save(folder + 'finalpos' + str(z) + '.npy', pos)
		np.save(folder + 'finalvel' + str(z) + '.npy', vel)
		np.save(folder + 'finaltime' + str(z) + '.npy', times)
		
		
	plt.legend()
	plt.show()
	raise RuntimeError('DONE')
	
	allpos = []
	allvel = []
	alltimes = []
	alldelta = []
	
	for k in flyer.finalPositions.iterkeys():
		allpos.extend(flyer.finalPositions[k])
		allvel.extend(flyer.finalVelocities[k])
		alltimes.extend(flyer.finalTimes[k])
		alldelta.extend(flyer.initialVelocities[:, 2] - flyer.finalVelocities[k][:, 2])
	allpos = np.array(allpos)
	allvel = np.array(allvel)
	alltimes = np.array(alltimes)
	alldelta = np.array(alldelta)
	
	ind = np.where(allpos[:, 2] > DETECTIONPLANE)
	#ind2 = np.where((allpos[:, 2] > DETECTIONPLANE) & (alldelta > 0.98*deltav0))
	
	plt.figure()
	plt.title('final velocities, all states')
	plt.hist(allvel[ind, 2].flatten(), bins = np.arange(0, 1, 0.01))
	#plt.hist(allvel[ind2, 2].flatten(), bins = np.arange(0, 1, 0.01))
	
	plt.figure()
	plt.title('arrival times, all states')
	plt.hist(alltimes[ind].flatten(), bins=range(0, STOPTIME, 10))
	#plt.hist(alltimes[ind2].flatten(), bins=range(0, STOPTIME, 10))
	
	
	plt.show()
