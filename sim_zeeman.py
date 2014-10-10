import numpy as np
from numpy import sqrt, pi
import time
from matplotlib import pyplot as plt
import os

import ctypes
from ctypes import c_double, c_ulong
c_double_p = ctypes.POINTER(c_double)

np.random.seed(1)

from subprocess import call
target = 'propagator_particle'
if not os.path.exists(target + '.so') or os.stat(target + '.c').st_mtime > os.stat(target + '.so').st_mtime: # we need to recompile
	COMPILE = ['DEBUG'] # 'PROF', 'FAST', both or neither
	# include branch prediction generation. compile final version with only -fprofile-use
	commonopts = ['-c', '-fPIC', '-Ofast', '-march=native', '-std=c99', '-fno-exceptions', '-fomit-frame-pointer']
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
		call(profcommand)
		call(['gcc', '-shared', '-fprofile-generate', target + '.o', '-o', target + '.so'])
		print 'COMPILATION: PROFILING RUN'
	if 'FAST' in COMPILE:
		call(fastcommand)
		call(['gcc', '-shared', target + '.o', '-o', target + '.so'])
		print 'COMPILATION: FAST RUN'
	if 'DEBUG' in COMPILE:
		call(debugcommand)
		call(['gcc', '-shared', target + '.o', '-o', target + '.so'])
	if not ('PROF' in COMPILE or 'FAST' in COMPILE or 'DEBUG' in COMPILE):
		print 'DID NOT RECOMPILE C SOURCE'
	print '==================================='
	print
	print


prop = ctypes.cdll.LoadLibrary('./' + target + '.so')

prop.setInitialBunch.argtypes = [c_double_p, c_double_p, c_double_p, c_ulong, c_double, c_ulong]
prop.setInitialBunch.restype = None
prop.setBFields.argtypes = [c_double_p, c_double_p, c_double_p, c_double_p, c_double, c_double, c_double, c_ulong, c_ulong, c_ulong]
prop.setBFields.restype = None
prop.setCoils.argtypes = [c_double_p, c_double_p, c_double, c_double, c_ulong]
prop.setCoils.restype = None
prop.setSkimmer.argtypes = [c_double, c_double, c_double, c_double]
prop.setSkimmer.restype = None
prop.doPropagate.argtypes = [c_double, c_double, c_double]
prop.doPropagate.restype = None
prop.setTimingParameters.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double]
prop.setTimingParameters.restype = None
prop.calculateCoilSwitching.argtypes = [c_double, c_double, c_double, c_double, c_double_p, c_double_p, c_double_p]
prop.calculateCoilSwitching.restype = int


kB = 1.3806504E-23 # Boltzmann constant (in J/K)
muB = 9.2740154E-24 # Bohr magneton in J/T
HBAR = 1.054571628E-34 # Planck constant (in Js)
A = 1420405751.768*2*pi/HBAR # in 1/((s^2)*J)
TIMESTEP = 1
STARTTIME = 31
STOPTIME = 1000
NPARTICLES = 10000
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
MAXPULSELENGTH = 85


# 12-coil version ??
ONTIME = [148.83, 161.64, 179.71, 197.96, 216.38, 235., 253.81, 272.82, 292.03, 311.47, 331.14, 351.04] # 21 degree 
OFFTIME =[167.64, 185.71, 203.96, 222.38, 241., 259.81, 278.82, 298.03, 317.47, 337.14, 357.04, 376.28] 


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

			
			initialPositions = np.vstack((initialPositions, numpy.array([x0_rnd[ts], y0_rnd[ts], z0_rnd[ts]]).T))
			initialVelocities = np.vstack((initialVelocities, numpy.array([vx0_rnd[ts], vy0_rnd[ts], vz0_rnd[ts]]).T))
			
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
		
		global ONTIME, OFFTIME
		A = np.genfromtxt(folder + 'coil.txt', dtype = np.float)
		ONTIME = A[:, 1]
		OFFTIME = A[:, 2]
		
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

class Coils:
	def calculateRampFactor(self, jj, time, ontimes, offtimes):
		
		m1 = H1/RAMP1
		m2 = (1-H1)/TIMEOVERLAP
		n2 = H1-m2*RAMP1
		m3 = -(1-H2)/TIMEOVERLAP
		m4 = -H2/RAMP1
		
		ontime = ontimes[jj]
		offtime = offtimes[jj]
		timediff = offtime - ontime
		
		rampfactor = 0
		if jj == 0:
			if time >= ontime and time < ontime+RAMPCOIL: # normal rise
				rampfactor = (CURRENT/SIMCURRENT)*(1/RAMPCOIL)*(time-ontime)
			elif time >= ontime+RAMPCOIL and time < offtime-TIMEOVERLAP: # constant level
				rampfactor = (CURRENT/SIMCURRENT)
			elif time >= offtime-TIMEOVERLAP and time < offtime: # overlap fall
				rampfactor = (CURRENT/SIMCURRENT)*(m3*(time-ontime)+(H2-m3*timediff))
			elif time >= offtime and time < offtime+RAMP1: # rise 1 fall
				rampfactor = (CURRENT/SIMCURRENT)*(m4*(time-ontime)-m4*(timediff+RAMP1))
		elif jj == NCOILS - 1:
			if time >= ontime and time < ontime+RAMP1: # rise 1 rise
				rampfactor = (CURRENT/SIMCURRENT)*(m1*(time-ontime))
			elif time >= ontime+RAMP1 and time < ontime+RAMP1+TIMEOVERLAP: # overlap rise
				rampfactor = (CURRENT/SIMCURRENT)*(m2*(time-ontime)+n2)
			elif time >= ontime+RAMP1+TIMEOVERLAP and time < offtime: # constant level
				rampfactor = (CURRENT/SIMCURRENT)
			elif time >= offtime and time < offtime+RAMPCOIL: # normal fall
				rampfactor = (CURRENT/SIMCURRENT)*(1/RAMPCOIL)*(offtime+RAMPCOIL-time)
		else:
			if time >= ontime and time < ontime+RAMP1: # rise 1 rise
				rampfactor = (CURRENT/SIMCURRENT)*(m1*(time-ontimes[jj]))
			elif time >= ontime+RAMP1 and time < ontime+RAMP1+TIMEOVERLAP: # overlap rise
				rampfactor = (CURRENT/SIMCURRENT)*(m2*(time-ontime)+n2)
			elif time >= ontime+RAMP1+TIMEOVERLAP and time < offtime-TIMEOVERLAP: # constant level
				rampfactor = (CURRENT/SIMCURRENT)
			elif time >= offtime-TIMEOVERLAP and time < offtime: # overlap fall
				rampfactor = (CURRENT/SIMCURRENT)*(m3*(time-ontime)+(H2-m3*timediff))
			elif time >= offtime and time < offtime+RAMP1: # rise 1 fall
				rampfactor = (CURRENT/SIMCURRENT)*(m4*(time-ontime)-m4*(timediff+RAMP1))
		return rampfactor
	
	def precalcCoilCurrents(self, starttime, stoptime, timestep):
		times = np.arange(starttime, stoptime, timestep)
		coilval = np.zeros((times.shape[0], NCOILS))
		for c in np.arange(NCOILS):
			for i, t in enumerate(times):
				coilval[i, c] = self.calculateRampFactor(c, t, ONTIME, OFFTIME)
		return coilval
	
	def calculateCoilSwitching(self, phasedeg, fields):
		##############################################################################
		## Generate the pulse sequence
		# using Zeeman effect = 1 (lfs, linear)
		# note: any Halbach hexapole arrays will not affect the motion of the
		# synchronous particle since Br=Bphi=Bz= 0 on axis
		
		# other variables:
		# gradB = sum(gradBz) = total B field gradient in a self.timestep
		# |B| = sum(Bz) = total B field in a self.timestep
		# rampfactor = to account for the finite rise/fall times
		#       of the coil B fields
		# accel, acc = acceleration of the synchronous particle
		# particle = generates a matrix for the particle [position velocity]
		# tol = tolerance for difference in switching time from previous and
		#       self.current iteration
		# s = self.timestep counter
		# oldcoil = switch-off time from last loop
		# lastcount = saves the last counter from the self.current loop
		
		
		## Zeeman effect
		# derivative of the Zeeman effect in atomic hydrogen:
		# for pulse generation: take only lfs state |+,+> (linear):
		dEZee = muB
		
		
		## Coil properties and switching positions along decelerator axis
		# coil = matrix for the coil positions
		# [position switch-on_time switch-off_time pulse_duration]
		coilon = np.zeros((NCOILS,))
		coiloff = np.zeros((NCOILS,))
		
		coildist = COILPOS[1] - COILPOS[0]
		
		# position of the synchr. particle when coils are switched off
		phaseangle = COILPOS - (coildist/180)*(90-phasedeg)
		#tfirstcoil = (phaseangle[0]-coildist)/BUNCHSPEED[2] - RAMPCOIL
		tfirstcoil = (COILPOS[0]-fields.bextend)/BUNCHSPEED[2] + RAMPCOIL
		# version pre-25/01/2013, shall be used again when going for "real"
		# deceleration with 12 coils
		
		ff = 2500.
		tolz = 0.01 # phase angle tolerance (in mm)
		
		
		
		coilon[0] = tfirstcoil
		coiloff[0] = coilon[0] + RAMPCOIL + ff*TIMESTEPPULSE
		coilon[1] = coiloff[0] - TIMEOVERLAP
		coiloff[1] = coilon[1] + RAMP1 + TIMEOVERLAP + ff*TIMESTEPPULSE
		
		## Preallocating vectors for speeding up the program
		time = 0
		
		## Optimization
		s = 0 # time counter
		zabs = BUNCHPOS[2] # initial position
		vz = BUNCHSPEED[2] # initial velocity
		vhz = vz # dummy velocity half-step
		cycles = 0 # number of cycles needed to get the pulse sequence
		
		gotvzis0 = 1
		
		for j in np.arange(NCOILS):
			ii = 1
			gottime = 0
			stopit = 0
			print 'analysing coil', j
			counter = 0
			neighbourcoils = np.arange(max(j-1, 0), min(j+3, NCOILS))
			
			while stopit == 0:
				if cycles > 1E3:
					raise RuntimeError('Maximum number of iterations reached without convergence. Change initial parameters.')
				
				s += 1 # begin with s = 1 since time(s=1) = 0*self.timestep
				time = s*TIMESTEPPULSE # time in microsec
				
				## B field ramp, B field and gradient
				# determines B field strength (Bz) along decelerator axis
				# next coil can only be switched on after the switch-off trigger for the
				# previous coil
				# only coil j, j-1 and j+1 since coils are not overlapped
				# predefine B field values!!
				# gradBtot_z = 0
				Bz1 = 0
				Bz2 = 0
				
				field = False
				for jj in neighbourcoils:
					if coilon[jj] != 0 and abs(zabs - COILPOS[jj]) < fields.bextend and time >= coilon[jj] and time <= coiloff[jj] + RAMPCOIL:
						field = True
						rampfactor = self.calculateRampFactor(jj, time, coilon, coiloff)
						index = np.ceil((zabs - COILPOS[jj] + fields.bextend)/fields.bdist)
						Bz1 += rampfactor*fields.bfieldz[index-1, 1]
						Bz2 += rampfactor*fields.bfieldz[index, 1]
				
				if field:
					# total gradient of B (only Bz component needed)
					gradBtot_z = (sqrt(Bz2**2)-sqrt(Bz1**2))/fields.bdist
					## Determine acceleration
					accsum_z = -(gradBtot_z/0.001)*dEZee/PARTICLEMASS*1E-9
				
				## Numerical integration of the equations of motion
				# using the Velocity Verlet method
				# remember: zabs = zabs(s), vz = vz(s-1)
				if time >= (coiloff[j]-TIMEOVERLAP-ff*TIMESTEPPULSE) and gottime == 0:
					sagain = s-1
					vzlast = vz
					vhzlast = vhz
					zabslast = zabs
					gottime = 1
				
				if field:
					vz = vhz + 0.5*TIMESTEPPULSE*accsum_z
					vhz = vz + 0.5*TIMESTEPPULSE*accsum_z
					
				zabs = zabs + TIMESTEPPULSE*vhz
				counter += 1
				
				
				if vz < 20/1000:
					raise RuntimeError('Particle is decelerated to v_z < 20 m/s: reduce phaseangle, increase initial velocity or decrease the number of coils to be switched.')
				
				# scheme: want phaseangle(j) to be the position at which the coil is
				# completely switched off --> hence, guess coil(j, 3) and then iterate until
				# coil(j, 3) + c.rampcoil, if phaseangle(j) is not reached, add some time to
				# coil(j, 3) and go back to old coil(j, 3) and integrate from there again
				
				if phaseangle[j] != 0:
					if j == NCOILS - 1:
						co = coiloff[j] + RAMPCOIL
					else:
						co = coiloff[j] + RAMP1
					
					if time >= co: # Oxford
						if j+2 != NCOILS:
							addramp = RAMP1+TIMEOVERLAP
						else:
							addramp = RAMPCOIL
						
						if zabs < phaseangle[j]: # particle position < phaseangle
							coiloffold = coiloff[j]
							cycles += 1
							sold = sagain
							vzold = vzlast
							vhzold = vhzlast
							zabsold = zabslast
							coiloff[j] += ff*TIMESTEPPULSE
							
							if j+2 <= NCOILS:
								# prevent generation of coil 12
								# if coil 12 is the bias coil
								# in order to avoid Majorana transitions, pulses will be overlapped
								# if trap is on, overlap the pulses of coil n and
								# coil n+2 (coil n+1 = trap coil)
								coilon[j+1] = coiloff[j] - TIMEOVERLAP
								
								# this yields the switch-on time for the next coil
								coiloff[j+1] = coilon[j+1] + addramp + ff*TIMESTEPPULSE
								
								# next coil turned off after the (shorter) ramptime plus some arb.
								# shift (guess)
								# if trap is on, turn off coil n+2 (coil n+1 = trap coil)
							
							s = sagain
							vz = vzlast
							vhz = vhzlast
							zabs = zabslast
							gottime = 0
						
						elif zabs >= phaseangle[j] and zabs <= phaseangle[j] + tolz: # particle position = phaseangle
							stopit = 1
						elif zabs > phaseangle[j] + tolz: # particle position >> phaseangle
							coiloff[j] = coiloffold + (ff/(2**ii))*TIMESTEPPULSE # try smaller stepsize
							ii += 1
							
							if j+2 <= NCOILS:
								# in order to avoid Majorana transitions, pulses will be overlapped:
								# if trap is on, overlap the pulses of coil n and
								# coil n+2 (coil n+1 = trap coil)
								coilon[j+1] = coiloff[j] - TIMEOVERLAP
								
								# this yields the switch-on time for the next coil
								coiloff[j+1] = coilon[j+1] + addramp + ff*TIMESTEPPULSE
								
								# next coil turned off after the (shorter) ramptime plus some arb.
								# shift (guess)
								# if trap is on, turn off coil n+2 (coil n+1 = trap coil)
							
							s = sold
							vz = vzold
							vhz = vhzold
							zabs = zabsold
							gottime = 0
			print 'counter for this round:', counter
		
		# pulse duration:
		duration = coiloff - coilon
		
		# pulses must not be longer than "maxpulselength",
		# else coils might explode
		mpl = np.where(duration > MAXPULSELENGTH)[0]
		print coilon
		print coiloff
		if len(mpl) > 0:
			if len(mpl) == 1 and mpl == NCOILS - 1:
				coil[-1, 2] = coil[-1, 1] + maxpulselength
				print('Length of last pulse reduced to maximum pulse length. Final velocity higher than expected!')
			else:
				raise RuntimeError('Maximum pulse duration exceeded for more than one coil! Change initial velocity and/or phi0.')
		
		vzfin = vz # final velocity of synchronous particle
		print 'final velocity: ', str(round(vzfin*1000)), 'm/s'
		
		# round coil timings to multiples of 10 ns to prevent pulseblaster from
		# ignoring times shorter than that
		coilon = np.round(coilon*100)/100
		coiloff = np.round(coiloff*100)/100
		
		print coilon
		print coiloff
		
		return coilon, coiloff
	
	
if __name__ == '__main__':
	bunch = ParticleBunch(TRAD, TLONG, PARTICLEMASS)
	#bunch.addUniformParticlesOnAxis(NPARTICLES, BUNCHRADIUS, BUNCHLENGTH, True, BUNCHPOS, BUNCHSPEED, False, SKIMMERDIST, SKIMMERRADIUS)
	#bunch.addParticles(NPARTICLES, BUNCHRADIUS, BUNCHLENGTH, True, BUNCHPOS, BUNCHSPEED, True, SKIMMERDIST, SKIMMERRADIUS)
	bunch.addSavedParticles('./output_600_61_0/', True, SKIMMERDIST, SKIMMERRADIUS)
	bunch.reset(0)

	pos = bunch.finalPositions[0]
	vel = bunch.finalVelocities[0]
	times = bunch.finalTimes[0]
	prop.setInitialBunch(pos.ctypes.data_as(c_double_p), vel.ctypes.data_as(c_double_p),  times.ctypes.data_as(c_double_p), bunch.nParticles, bunch.mass, 0)

	skimmerloss_no = 100.*bunch.nGeneratedGood/bunch.nGenerated
	print 'particles coming out of the skimmer (in percent): %.2f\n' % skimmerloss_no
	
	
	fields = Fields()
	fields.loadBFields()
	
	# calculate coil current matrix
	coils = Coils()
	#start = time.clock()
	#res = coils.calculateCoilSwitching(21, fields)
	#print 'time for coil calculation was', time.clock()-start, 'seconds'	
	#print res
	#raise RuntimeError
	
	currents = coils.precalcCoilCurrents(STARTTIME, STOPTIME, TIMESTEP)
	coilpos =  np.array(COILPOS)

	prop.setTimingParameters(H1, H2, RAMP1, TIMEOVERLAP, RAMPCOIL, MAXPULSELENGTH)
	prop.setBFields(fields.Bz_n_flat.ctypes.data_as(c_double_p), fields.Br_n_flat.ctypes.data_as(c_double_p), fields.zaxis.ctypes.data_as(c_double_p), fields.raxis.ctypes.data_as(c_double_p), fields.bzextend, fields.zdist, fields.rdist, fields.sizZ, fields.sizR, fields.sizB)
	prop.setCoils(currents.ctypes.data_as(c_double_p), coilpos.ctypes.data_as(c_double_p), COILRADIUS, DETECTIONPLANE, NCOILS)
	prop.setSkimmer(SKIMMERDIST, SKIMMERLENGTH, SKIMMERRADIUS, SKIMMERALPHA)

	ontimes = np.zeros(NCOILS)
	offtimes = np.zeros(NCOILS)
	
	print prop.calculateCoilSwitching(0., 0.6, 61., 4.e-3, fields.bfieldz.ctypes.data_as(c_double_p), ontimes.ctypes.data_as(c_double_p), offtimes.ctypes.data_as(c_double_p))
	print ontimes
	print offtimes
	raise RuntimeError
	
	prop.setCoils(currents.ctypes.data_as(c_double_p), coilpos.ctypes.data_as(c_double_p), COILRADIUS, DETECTIONPLANE, NCOILS)
	currents = coils.precalcCoilCurrents(STARTTIME, STOPTIME, TIMESTEP)
	coilpos =  np.array(COILPOS)
	
	
	print 'starting propagation'
	start = time.clock()
	
	for z in range(4): # simulate all possible zeeman states
		bunch.reset(z)
		
		pos = bunch.finalPositions[z]
		vel = bunch.finalVelocities[z]
		times = bunch.finalTimes[z]
		
		prop.setInitialBunch(pos.ctypes.data_as(c_double_p), vel.ctypes.data_as(c_double_p),  times.ctypes.data_as(c_double_p), bunch.nParticles, bunch.mass, z)
		prop.doPropagate(STARTTIME, STOPTIME, TIMESTEP)
		ind = np.where(vel[:, 0] != 0)[0]
		print 1.*ind.shape[0]/bunch.nGenerated*1e6
			
	print 'time for propagation was', time.clock()-start, 'seconds'	
	
	plt.figure(0)
	plt.title('final velocities')
	plt.hist(bunch.initialVelocities[:, 2], bins = np.arange(0.25, 0.45, 0.005), histtype='step')
	
	deltav0 = bunch.initialVelocities[0, 2] - bunch.finalVelocities[0][0, 2]
	
	for z in range(4): # analysis for all zeeman states
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
	ind2 = np.where((allpos[:, 2] > DETECTIONPLANE) & (alldelta > 0.98*deltav0))
	
	plt.figure()
	plt.title('final velocities, all states')
	plt.hist(allvel[ind, 2].flatten(), bins = np.arange(0, 1, 0.01))
	#plt.hist(allvel[ind2, 2].flatten(), bins = np.arange(0, 1, 0.01))
	
	plt.figure()
	plt.title('arrival times, all states')
	plt.hist(alltimes[ind].flatten(), bins=range(0, STOPTIME, 10))
	#plt.hist(alltimes[ind2].flatten(), bins=range(0, STOPTIME, 10))
	
	
	plt.show()
