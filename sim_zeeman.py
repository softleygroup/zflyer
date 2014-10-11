import numpy as np
from numpy import sqrt, pi
import time
from matplotlib import pyplot as plt
import os

import ctypes
from ctypes import c_double, c_uint, c_int
c_double_p = ctypes.POINTER(c_double)

np.random.seed(1)

from subprocess import call
target = 'propagator_particle'
if not os.path.exists(target + '.so') or os.stat(target + '.c').st_mtime > os.stat(target + '.so').st_mtime: # we need to recompile
	COMPILE = ['PROF'] # 'PROF', 'FAST', both or neither
	# include branch prediction generation. compile final version with only -fprofile-use
	commonopts = ['-c', '-fPIC', '-Ofast', '-march=native', '-std=c99', '-Wall', '-fno-exceptions', '-fomit-frame-pointer']
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


prop = ctypes.cdll.LoadLibrary('./' + target + '.so')
prop.setSynchronousParticle.argtypes = [c_double, c_double_p, c_double_p]
prop.setSynchronousParticle.restype = None
prop.setBFields.argtypes = [c_double_p, c_double_p, c_double_p, c_double_p, c_double, c_double, c_double, c_uint, c_uint, c_uint]
prop.setBFields.restype = None
prop.setCoils.argtypes = [c_double_p, c_double, c_double, c_uint]
prop.setCoils.restype = None
prop.setSkimmer.argtypes = [c_double, c_double, c_double, c_double]
prop.setSkimmer.restype = None
prop.doPropagate.argtypes = [c_double_p, c_double_p, c_double_p, c_uint, c_int]
prop.doPropagate.restype = None
prop.setTimingParameters.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double]
prop.setTimingParameters.restype = None
prop.calculateCoilSwitching.argtypes = [c_double, c_double, c_double_p, c_double_p, c_double_p]
prop.calculateCoilSwitching.restype = int
prop.precalculateCurrents.argtypes = [c_double_p]
prop.precalculateCurrents.restype = int
prop.setPropagationParameters.argtypes = [c_double, c_double, c_double]
prop.setPropagationParameters.restype = None

kB = 1.3806504E-23 # Boltzmann constant (in J/K)
muB = 9.2740154E-24 # Bohr magneton in J/T
HBAR = 1.054571628E-34 # Planck constant (in Js)
A = 1420405751.768*2*pi/HBAR # in 1/((s^2)*J)
TIMESTEP = 1
STARTTIME = 31
STOPTIME = 1000
NPARTICLES = 20000
TRAD = 0.01
TLONG = 1.1
PARTICLEMASS = 1.00782503*1.6605402e-27 # in amu, H atom
#PARTICLEMASS = 2*14.007*1.6605402e-27 # in amu, N2 molecule
BUNCHRADIUS = 1.
BUNCHLENGTH = 15.
BUNCHPOS = [0., 0., 0.]
BUNCHSPEED = [0., 0., 600e-3]
COILPOS = np.array([108.9, 119.6, 130.3, 141., 151.7, 162.4, 173.1, 183.8, 194.5, 205.2, 215.9, 226.6])# - 18.6 # zshiftdetect?
COILRADIUS = 3.0
NCOILS = 12
SKIMMERDIST = 49.
SKIMMERRADIUS = 1.0
SKIMMERLENGTH = 20.88  
SKIMMERALPHA = np.arctan((22.94/2.-SKIMMERRADIUS)/SKIMMERLENGTH)
DETECTIONPLANE = 268.

# stuff for coil current calculation
H1 = 0.94
H2 = 0.9
RAMP1 = 7.1
TIMEOVERLAP = 6.
RAMPCOIL = 8.
CURRENT = 243.
SIMCURRENT = 300.

# stuff for coil time calculation
TIMESTEPPULSE = 4e-3 # do this with smaller timesteps than real propagation
PHASEANGLE = 21
MAXPULSELENGTH = 85


class ParticleBunch:
	def __init__(self, radial_temp, long_temp, particle_mass):
		self.initial_Tr = radial_temp
		self.initial_Tz = long_temp
		self.mass = particle_mass
		
		self.nGenerated = 0
		self.nGeneratedGood = 0
		
		self.finalPositions = {}
		self.finalVelocities = {}
		self.finalTimes = {}
	
	def addParticles(self, nParticles, diam, length, includeSyn=False, synPos = [0, 0, 0], synVel = [0, 0, 0], checkSkimmer=False, skimmerDist = 0, skimmerRadius = 0):
		if includeSyn:
			# synchronous particle (set by default)
			initialPositions = np.array([synPos])
			initialVelocities = np.array([synVel])
			self.nGenerated += 1
			self.nGeneratedGood += 1
		else:
			initialPositions = np.zeros((0, 3))
			initialVelocities = np.zeros((0, 3))
		
		while self.nGeneratedGood < nParticles:
			nParticlesToSim = nParticles - self.nGeneratedGood
			# (a) for positions
			# random uniform distribution within a cylinder
			# r0 and phi0 span up a disk; z0 gives the height
			r0_rnd = sqrt(np.random.uniform(0, diam/2, nParticlesToSim))*sqrt(diam/2)
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
			sigmavr0 = sqrt(kB*self.initial_Tr/self.mass)/1000 # standard deviation self.vr0 component
			
			# normally distributed random numbers centered at 0 mm/mus
			# generate bi(multi)variate Gaussian data for vx and vy
			# rand_data = mvnrnd(mu, sigma,num of data)
			muvr = [0, 0] # mean values centered around 0 mm/mus
			# sigma1 = [1 0  # covariance matrix, diagonals = variances of each variable,
			#          0 1]  # off-diagonals = covariances between the variables
			# if no correlation, then off-diagonals = 0 and Sigma can also be written as a row array
			SigmaM = [[sigmavr0**2, 0], [0, sigmavr0**2]]
			vx0_rnd, vy0_rnd = np.random.multivariate_normal(muvr, SigmaM, [nParticlesToSim]).T
			
			sigmavz0 = sqrt(kB*self.initial_Tz/self.mass)/1000 # standard deviation vz0 component
			vz0_rnd = np.random.normal(synVel[2], sigmavz0, nParticlesToSim)
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
			
			self.nGenerated += nParticlesToSim
			self.nGeneratedGood  = initialPositions.shape[0]
		
		self.initialPositions = np.array(initialPositions)
		self.initialVelocities = np.array(initialVelocities)
		
		
	def addUniformParticlesOnAxis(self, nParticles, diam, length, includeSyn=False, synPos = [0, 0, 0], synVel = [0, 0, 0], checkSkimmer=False, skimmerDist = 0, skimmerRadius = 0):
		if includeSyn:
			# synchronous particle (set by default)
			initialPositions = np.array([synPos])
			initialVelocities = np.array([synVel])
			self.nGenerated += 1
			self.nGeneratedGood += 1
		else:
			initialPositions = np.zeros((0, 3))
			initialVelocities = np.zeros((0, 3))
		
		while self.nGeneratedGood < nParticles:
			nParticlesToSim = nParticles - self.nGeneratedGood
			
			x0_rnd = np.zeros((nParticlesToSim,))
			y0_rnd = np.zeros((nParticlesToSim,))
			
			z0_rnd = 5. + np.random.uniform(-length/2, length/2, nParticlesToSim)
			
			
			vx0_rnd = np.zeros((nParticlesToSim,))
			vy0_rnd = np.zeros((nParticlesToSim,))
			
			sigmavz0 = sqrt(kB*self.initial_Tz/self.mass)/1000 # standard deviation vz0 component
			vz0_rnd = np.random.uniform(synVel[2] - sigmavz0, synVel[2] + sigmavz0, nParticlesToSim)
			
			if checkSkimmer:
				xatskimmer = x0_rnd + (vx0_rnd/vz0_rnd)*(skimmerDist-z0_rnd)
				yatskimmer = y0_rnd + (vy0_rnd/vz0_rnd)*(skimmerDist-z0_rnd)
				ratskimmer = sqrt(xatskimmer**2 + yatskimmer**2)
				ts = np.where(ratskimmer<=skimmerRadius)[0]
			else:
				ts = slice(0, x0_rnd.shape[0])

			
			initialPositions = np.vstack((initialPositions, np.array([x0_rnd[ts], y0_rnd[ts], z0_rnd[ts]]).T))
			initialVelocities = np.vstack((initialVelocities, np.array([vx0_rnd[ts], vy0_rnd[ts], vz0_rnd[ts]]).T))
			
			self.nGenerated += nParticlesToSim
			self.nGeneratedGood  = initialPositions.shape[0]
		
		self.initialPositions = np.array(initialPositions)
		self.initialVelocities = np.array(initialVelocities)
		
		
	def addSavedParticles(self, folder, checkSkimmer=False, skimmerDist = 0, skimmerRadius = 0):
		A = np.genfromtxt(folder + 'init_cond.txt', dtype=np.float)
		self.initialPositions = A[:, :3]
		self.initialVelocities=  A[:, 3:]/1000.
		self.nGeneratedGood = A.shape[0]
		self.nGenerated = 3371498#A.shape[0]
		
		#global ONTIME, OFFTIME
		#A = np.genfromtxt(folder + 'coil.txt', dtype = np.float)
		#ONTIME = A[:, 1]
		#OFFTIME = A[:, 2]
		
	def reset(self, initialZeemanState):
		self.finalPositions[initialZeemanState] = np.require(self.initialPositions.copy(), requirements=['C', 'A', 'O', 'W'])
		self.finalVelocities[initialZeemanState] = np.require(self.initialVelocities.copy(), requirements=['C', 'A', 'O', 'W'])
		
		self.nParticles = self.initialPositions.shape[0]
		
		self.finalTimes[initialZeemanState] = np.require(np.empty((self.nParticles, )))
		
		#self.currentZeemanState = initialZeemanState*np.ones((self.nParticles,1))
		#self.currentTime = np.zeros((self.nParticles, 1))
		#self.currentIndex = np.arange(self.nParticles)
		return 0

class Fields:
	def loadBFields(self):
		## B field along z axis
		# from FEMM or Comsol file
		bfieldz = np.require(np.genfromtxt('sim_files/bonzaxis.txt', delimiter='\t'), requirements=['C', 'A', 'O', 'W'])  # analytic solution
		
		# bfieldz = np.genfromtxt('sim_files/baxis_Zurich.txt', delimiter='\t') # Zurich Comsol calculation
		bdist = bfieldz[1, 0] - bfieldz[0, 0]   # spacing B field (in T)
		bextend = -bfieldz[0, 0]                # dimension B field along decelerator axis (in mm)
		
		self.bdist = bdist
		self.bextend = bextend
		self.bfieldz = bfieldz
		
		
		## B field in coil centre
		# for spin flip calc, at sim.current
		B0ind = np.where(bfieldz[:, 0] >= 0)[0][0]
		B0 = bfieldz[B0ind, 1]
		
		
		## B field coil
		Bz_n = np.genfromtxt('sim_files/Bz_n.txt', delimiter='\t').T # contains Bz field as a grid with P(r,z) (from analytic solution)
		Br_n = np.genfromtxt('sim_files/Br_n.txt', delimiter='\t').T # contains Br field as a grid with P(r,z) (from analytic solution)
		
		self.raxis = np.genfromtxt('sim_files/raxis.txt', delimiter='\t') # raxis as one column
		self.zaxis = np.genfromtxt('sim_files/zaxis.txt', delimiter='\t') # zaxis as one row
		
		self.zdist = self.zaxis[1] - self.zaxis[0] # spacing B field z axis (in mm)
		self.rdist = self.raxis[1] - self.raxis[0] # spacing B field r axis (in mm)
		self.bzextend = -self.zaxis[0] # dimension B field along decelerator z axis (in mm)
		self.sizB = Bz_n.shape[1]
		
		self.Bz_n_flat = Bz_n.flatten()
		self.Br_n_flat = Br_n.flatten()
		
		self.sizZ = self.zaxis.shape[0]
		self.sizR = self.raxis.shape[0]

if __name__ == '__main__':
	bunch = ParticleBunch(TRAD, TLONG, PARTICLEMASS)
	#bunch.addUniformParticlesOnAxis(NPARTICLES, BUNCHRADIUS, BUNCHLENGTH, True, BUNCHPOS, BUNCHSPEED, False, SKIMMERDIST, SKIMMERRADIUS)
	bunch.addParticles(NPARTICLES, BUNCHRADIUS, BUNCHLENGTH, True, BUNCHPOS, BUNCHSPEED, True, SKIMMERDIST, SKIMMERRADIUS)
	#bunch.addSavedParticles('./output_600_21_0/', True, SKIMMERDIST, SKIMMERRADIUS)
	
	bunchpos = np.array(BUNCHPOS)
	bunchspeed = np.array(BUNCHSPEED)
	prop.setSynchronousParticle(bunch.mass, bunchpos.ctypes.data_as(c_double_p), bunchspeed.ctypes.data_as(c_double_p))
	
	skimmerloss_no = 100.*bunch.nGeneratedGood/bunch.nGenerated
	print 'particles coming out of the skimmer (in percent): %.2f\n' % skimmerloss_no
	
	
	fields = Fields()
	fields.loadBFields()
	
	coilpos =  np.array(COILPOS)
	
	prop.setTimingParameters(H1, H2, RAMP1, TIMEOVERLAP, RAMPCOIL, MAXPULSELENGTH)
	prop.setBFields(fields.Bz_n_flat.ctypes.data_as(c_double_p), fields.Br_n_flat.ctypes.data_as(c_double_p), fields.zaxis.ctypes.data_as(c_double_p), fields.raxis.ctypes.data_as(c_double_p), fields.bzextend, fields.zdist, fields.rdist, fields.sizZ, fields.sizR, fields.sizB)
	prop.setCoils(coilpos.ctypes.data_as(c_double_p), COILRADIUS, DETECTIONPLANE, NCOILS)
	prop.setSkimmer(SKIMMERDIST, SKIMMERLENGTH, SKIMMERRADIUS, SKIMMERALPHA)

	ontimes = np.zeros(NCOILS, dtype=np.double)
	offtimes = np.zeros(NCOILS, dtype=np.double)
	
	if not prop.calculateCoilSwitching(PHASEANGLE, TIMESTEPPULSE, fields.bfieldz.ctypes.data_as(c_double_p), ontimes.ctypes.data_as(c_double_p), offtimes.ctypes.data_as(c_double_p)) == 0:
		raise RuntimeError("Error while calculating coil switching times")
		
	coilpos -= 18.6 # zshiftdetect??
	prop.setCoils(coilpos.ctypes.data_as(c_double_p), COILRADIUS, DETECTIONPLANE, NCOILS)
	prop.setPropagationParameters(STARTTIME, TIMESTEP, (STOPTIME-STARTTIME)/TIMESTEP)
	currents = np.zeros(((STOPTIME-STARTTIME)/TIMESTEP, NCOILS), dtype=np.double)
	if not prop.precalculateCurrents(currents.ctypes.data_as(c_double_p)) == 0:
		raise RuntimeError("Error precalculating currents!")
	
	print 'starting propagation'
	start = time.clock()
	
	for z in range(-1, 4): # simulate all possible zeeman states, -1 is decelerator off
		bunch.reset(z)
		
		pos = bunch.finalPositions[z]
		vel = bunch.finalVelocities[z]
		times = bunch.finalTimes[z]
		
		prop.doPropagate(pos.ctypes.data_as(c_double_p), vel.ctypes.data_as(c_double_p),  times.ctypes.data_as(c_double_p), bunch.nParticles, z)
		ind = np.where(vel[:, 0] != 0)[0]
		print 1.*ind.shape[0]/bunch.nGenerated*1e6
			
	print 'time for propagation was', time.clock()-start, 'seconds'	
	
	plt.figure(0)
	plt.title('final velocities')
	
	for z in range(-1, 4): # analysis for all zeeman states, including gaspulse
		pos = bunch.finalPositions[z]
		vel = bunch.finalVelocities[z]
		times = bunch.finalTimes[z]
		ind = np.where((pos[:, 2] > DETECTIONPLANE)) # all particles that reach the end
		plt.figure(0)
		plt.hist(vel[ind, 2].flatten(), bins = np.arange(0, 1, 0.01), histtype='step', label=str(z))
		
		#plt.figure(1)
		#deltav = bunch.initialVelocities[:, 2] - vel[:, 2]
		#ind2 = np.where((deltav > 0.98*deltav0) & (pos[:, 2] > DETECTIONPLANE))
		#plt.plot(bunch.initialPositions[ind2, 2].T - BUNCHPOS[2], (BUNCHSPEED[2] - bunch.initialVelocities[ind2, 2].T)*1000, 'x')
		#plt.plot(pos[ind2, 2].T - pos[0, 2], (vel[ind2, 2].T - vel[0, 2])*1000, 'x'
	
	plt.legend()
	plt.show()
	raise RuntimeError('DONE')
	
	allpos = []
	allvel = []
	alltimes = []
	alldelta = []
	
	for k in bunch.finalPositions.iterkeys():
		allpos.extend(bunch.finalPositions[k])
		allvel.extend(bunch.finalVelocities[k])
		alltimes.extend(bunch.finalTimes[k])
		alldelta.extend(bunch.initialVelocities[:, 2] - bunch.finalVelocities[k][:, 2])
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
