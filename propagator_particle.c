#include "propagator.h"

#define PROPAGATE 0
#define DETECTED 1
#define LOST 2

#define HYDROGEN 1
#define NITROGEN 2
#define ELEMENT HYDROGEN

#define muB 9.2740154E-24 // Bohr Magneton, J/T
#define HBAR 1.054571628E-34 // Planck constant (in Js)
#define PI 3.141592653589793
#define A 1420405751.768*2*PI/HBAR //hf splitting in 1/((s^2)*J)

#define SIMCURRENT 300.
#define CURRENT 243.

static unsigned long nParticles, nDetected, nLost;
static double particleMass;
static unsigned long zeemanState;

static double *restrict pos = NULL;
static double *restrict vel = NULL;

static double r;

static double *restrict finaltime = NULL;
static double *restrict finalpos = NULL, *restrict finalvel = NULL;

static double *restrict coilpos = NULL;
static double coilrad;
static double endpos;
static unsigned long nCoils = 0;
static double *restrict coilon = NULL;
static double *restrict coiloff = NULL; 
static double *restrict currents = NULL;

static double skimmerdist, skimmerradius, skimmeralpha, skimmerlength;

static double *restrict Bz=NULL, *restrict Br=NULL;
static double *restrict raxis=NULL, *restrict zaxis=NULL;
static double bzextend, zdist, rdist;
static unsigned long sizZ, sizR, sizB;

static double h1, h2, ramp1, rampcoil, timeoverlap, maxpulselength;

// get particle bunch information from python
void setInitialBunch(double *pos, double *vel, double *time, unsigned long nParticles_l, double particleMass_l, unsigned long zeemanState_l)
{
	nParticles = nParticles_l;
	particleMass = particleMass_l;
	
	zeemanState = zeemanState_l;
	
	nDetected = 0;
	nLost = 0;
	
	finalpos = __builtin_assume_aligned(pos, 16);
	finalvel = __builtin_assume_aligned(vel, 16);
	finaltime = __builtin_assume_aligned(time, 16);
}

// get field information from python
void setBFields(double *Bz_l, double *Br_l, double *zaxis_l, double *raxis_l, double bzextend_l, double zdist_l, double rdist_l, unsigned long sizZ_l, unsigned long sizR_l, unsigned long sizB_l)
{
	Bz = __builtin_assume_aligned(Bz_l, 16);
	Br = __builtin_assume_aligned(Br_l, 16);
	zaxis = __builtin_assume_aligned(zaxis_l, 16);
	raxis = __builtin_assume_aligned(raxis_l, 16);
	bzextend = bzextend_l;
	zdist = zdist_l;
	rdist = rdist_l;
	sizZ = sizZ_l;
	sizR = sizR_l;
	sizB = sizB_l;
}
	
// get coil information from python
void setCoils(double *currents_l, double *coilpos_l, double coilrad_l, double endpos_l, unsigned long nCoils_l)
{
	currents = __builtin_assume_aligned(currents_l, 16);
	coilpos = coilpos_l;
	coilrad = coilrad_l;
	endpos = endpos_l;
	nCoils = nCoils_l;
}

// set skimmer geometry
void setSkimmer(const double skimmerdist_l, const double skimmerlength_l, const double skimmerradius_l, const double skimmeralpha_l)
{
	skimmerdist = skimmerdist_l;
	skimmerlength = skimmerlength_l;
	skimmerradius = skimmerradius_l;
	skimmeralpha = skimmeralpha_l;
}


void setTimingParameters(const double h1_l, const double h2_l, const double ramp1_l, const double timeoverlap_l, const double rampcoil_l, const double maxpulselength_l)
{
	h1 = h1_l;
	h2 = h2_l;
	ramp1 = ramp1_l;
	timeoverlap = timeoverlap_l;
	rampcoil = rampcoil_l;
	maxpulselength = maxpulselength_l;
}

static inline double calculateRampFactor(unsigned int j, const double time)
{
		const double m1 = h1/ramp1;
		const double m2 = (1-h1)/timeoverlap;
		const double n2 = h1-m2*ramp1;
		const double m3 = -(1-h2)/timeoverlap;
		const double m4 = -h2/ramp1;
		
		const double ontime = coilon[j];
		const double offtime = coiloff[j];
		
		const double timediff = offtime - ontime;
		
		double rampfactor = 0;
		switch (j)
		{
		case 0:
			if (time >= ontime && time < ontime+rampcoil) // normal rise
				rampfactor = (CURRENT/SIMCURRENT)*(1./rampcoil)*(time-ontime);
			else if (time >= ontime+rampcoil && time < offtime-timeoverlap) // constant level
				rampfactor = (CURRENT/SIMCURRENT);
			else if (time >= offtime-timeoverlap && time < offtime) // overlap fall
				rampfactor = (CURRENT/SIMCURRENT)*(m3*(time-ontime)+(h2-m3*timediff));
			else if (time >= offtime && time < offtime+ramp1) // rise 1 fall
				rampfactor = (CURRENT/SIMCURRENT)*(m4*(time-ontime)-m4*(timediff+ramp1));
			break;
		case 11: //WARN: this should be nCoils - 1, but switch/case doesn't take variables, only constants!
			if (time >= ontime && time < ontime+ramp1) // rise 1 rise
				rampfactor = (CURRENT/SIMCURRENT)*(m1*(time-ontime));
			else if (time >= ontime+ramp1 && time < ontime+ramp1+timeoverlap) // overlap rise
				rampfactor = (CURRENT/SIMCURRENT)*(m2*(time-ontime)+n2);
			else if (time >= ontime+ramp1+timeoverlap && time < offtime) // constant level
				rampfactor = (CURRENT/SIMCURRENT);
			else if (time >= offtime && time < offtime+rampcoil) // normal fall
				rampfactor = (CURRENT/SIMCURRENT)*(1./rampcoil)*(offtime+rampcoil-time);
			break;
		default:
			if (time >= ontime && time < ontime+ramp1) // rise 1 rise
				rampfactor = (CURRENT/SIMCURRENT)*(m1*(time-ontime));
			else if (time >= ontime+ramp1 && time < ontime+ramp1+timeoverlap) // overlap rise
				rampfactor = (CURRENT/SIMCURRENT)*(m2*(time-ontime)+n2);
			else if (time >= ontime+ramp1+timeoverlap && time < offtime-timeoverlap) // constant level
				rampfactor = (CURRENT/SIMCURRENT);
			else if (time >= offtime-timeoverlap && time < offtime) // overlap fall
				rampfactor = (CURRENT/SIMCURRENT)*(m3*(time-ontime)+(h2-m3*timediff));
			else if (time >= offtime && time < offtime+ramp1) // rise 1 fall
				rampfactor = (CURRENT/SIMCURRENT)*(m4*(time-ontime)-m4*(timediff+ramp1));
		}
		
		return rampfactor;
}

int calculateCoilSwitching(const double x0, const double v0, const double phase, const double dT, const double * bfieldz, double * coilon_l, double * coiloff_l)
{
	/* GENERATE THE PULSE SEQUENCE
	* using Zeeman effect = 1 (lfs, linear)
	* note: any Halbach hexapole arrays will not affect the motion of the
	* synchronous particle since Br=Bphi=Bz= 0 on axis
	*
	* other variables:
	* gradB = sum(gradBz) = total B field gradient in a self.timestep
	* |B| = sum(Bz) = total B field in a self.timestep
	* rampfactor = to account for the finite rise/fall times
	*       of the coil B fields
	* accel, acc = acceleration of the synchronous particle
	* particle = generates a matrix for the particle [position velocity]
	* tol = tolerance for difference in switching time from previous and
	*       self.current iteration
	* s = self.timestep counter
	* oldcoil = switch-off time from last loop
	* lastcount = saves the last counter from the self.current loop
	*/

	// Zeeman effect
	// derivative of the Zeeman effect in atomic hydrogen:
	// for pulse generation: take only lfs state |+,+> (linear):
	
	coilon = __builtin_assume_aligned(coilon_l, 16);
	coiloff = __builtin_assume_aligned(coiloff_l, 16);

	
	const double dEZee = muB;
	
	double Bz1, Bz2;
	double gradBtot_z, accsum_z;
	int field, index;
	double rampfactor;
	int sagain, sold;
	
	// Optimization
	unsigned int s = 0; // time counter
	double zabs = x0; // initial position
	double vz = v0; // initial velocity
	double vzlast, vhzlast, vzold, vhzold;
	double zabslast, zabsold;
	double co;
	double addramp;
	double coiloffold;
	
	const double coildist = coilpos[1] - coilpos[0];
	const double bextend = -bfieldz[0];
	const double bdist = bfieldz[2] - bfieldz[0];
	
	// position of the synchr. particle when coils are switched off
	double phaseangle[nCoils];
	for (unsigned int i = 0; i < nCoils; i++)
	{
		phaseangle[i] = coilpos[i] - (coildist/180)*(90-phase);
	}
	// tfirstcoil = (phaseangle[0]-coildist)/BUNCHSPEED[2] - RAMPCOIL
	const double tfirstcoil = (coilpos[0] - bextend)/v0 + rampcoil;
	// version pre-25/01/2013, shall be used again when going for "real"
	// deceleration with 12 coils
	
	const double ff = 2500.0;
	const double tolz = 0.005; // phase angle tolerance (in mm)
	
	coilon[0] = tfirstcoil;
	coiloff[0] = coilon[0] + rampcoil + ff*dT;
	coilon[1] = coiloff[0] - timeoverlap;
	coiloff[1] = coilon[1] + ramp1 + timeoverlap + ff*dT;
	
	// Preallocating vectors for speeding up the program
	double time = 0.;
	
	double vhz = vz; // dummy velocity half-step
	unsigned int cycles = 0; // number of cycles needed to get the pulse sequence
	
	unsigned int gotvzis0 = 1;
	
	for (unsigned int j = 0; j < nCoils; j++)
	{
		unsigned int ii = 1;
		unsigned int gottime = 0;
		printf("analysing coil %d\n", j);
		
		int leftcoil = j > 1 ? j - 1 : 0;
		int rightcoil = j + 3 < nCoils ? j + 3 : nCoils - 1;
		
		while (1)
		{
			if (cycles > 1E3)
			{
				printf("Maximum number of iterations reached without convergence. Change initial parameters.\n");
				return(-1);
			}
			
			s++; // begin with s = 1 since time(s=1) = 0*self.timestep
			time = s*dT; // time in microsec
			
			/* 
			* B field ramp, B field and gradient
			* determines B field strength (Bz) along decelerator axis
			* next coil can only be switched on after the switch-off trigger for the
			* previous coil
			* only coil j, j-1 and j+1 since coils are not overlapped
			* predefine B field values!!
			* gradBtot_z = 0
			*/
			
			Bz1 = 0;
			Bz2 = 0;
			
			field = 0;
			for (int jj = leftcoil; jj <= rightcoil; jj++)
			{
				if ((coilon[jj] != 0) && (abs(zabs - coilpos[jj]) < bextend) && (time >= coilon[jj]) && (time <= coiloff[jj] + rampcoil))
				{
					field = 1;
					rampfactor = calculateRampFactor(jj, time);
					index = ceil((zabs - coilpos[jj] + bextend)/bdist);
					Bz1 += rampfactor*bfieldz[2*(index-1) + 1];
					Bz2 += rampfactor*bfieldz[2*(index) + 1];
				}
			}
			
			
			if (field == 1)
			{
				// total gradient of B (only Bz component needed)
				gradBtot_z = (sqrt(Bz2*Bz2)-sqrt(Bz1*Bz1))/bdist;
				// Determine acceleration
				accsum_z = -(gradBtot_z/0.001)*dEZee/particleMass*1E-9;
			}
			
			// Numerical integration of the equations of motion
			// using the Velocity Verlet method (??)
			// remember: zabs = zabs(s), vz = vz(s-1)
			if (time >= (coiloff[j]-timeoverlap-ff*dT) && gottime == 0)
			{
				sagain = s-1;
				vzlast = vz;
				vhzlast = vhz;
				zabslast = zabs;
				gottime = 1;
			}
			
			if (field == 1)
			{
				vz = vhz + 0.5*dT*accsum_z;
				vhz = vz + 0.5*dT*accsum_z;
			}
			
			zabs = zabs + dT*vhz;
			
			if (vz < 20/1000)
			{
				printf("Particle is decelerated to v_z < 20 m/s: reduce phaseangle, increase initial velocity or decrease the number of coils to be switched.\n");
				return (-1);
			}
			
			/* 
			* scheme: want phaseangle(j) to be the position at which the coil is
			* completely switched off --> hence, guess coil(j, 3) and then iterate until
			* coil(j, 3) + c.rampcoil, if phaseangle(j) is not reached, add some time to
			* coil(j, 3) and go back to old coil(j, 3) and integrate from there again
			*/
			if (phaseangle[j] != 0)
			{
				if (j == nCoils - 1)
				{
					co = coiloff[j] + rampcoil;
				}
				else
				{
					co = coiloff[j] + ramp1;
				}
				if (time >= co) // Oxford
				{
					if (j+2 != nCoils)
					{
						addramp = ramp1+timeoverlap;
					}
					else
					{
						addramp = rampcoil;
					}
					
					if (zabs < phaseangle[j]) // particle position < phaseangle
					{
						coiloffold = coiloff[j];
						cycles++;
						sold = sagain;
						vzold = vzlast;
						vhzold = vhzlast;
						zabsold = zabslast;
						coiloff[j] += ff*dT;
						
						if (j+2 <= nCoils)
						{
							/*
							 *  prevent generation of coil 12
							 * if coil 12 is the bias coil
							 * in order to avoid Majorana transitions, pulses will be overlapped
							 * if trap is on, overlap the pulses of coil n and
							 * coil n+2 (coil n+1 = trap coil)
							 */
							coilon[j+1] = coiloff[j] - timeoverlap;
							
							// this yields the switch-on time for the next coil
							coiloff[j+1] = coilon[j+1] + addramp + ff*dT;
							
							// next coil turned off after the (shorter) ramptime plus some arb.
							// shift (guess)
							// if trap is on, turn off coil n+2 (coil n+1 = trap coil)
						}
						s = sagain;
						vz = vzlast;
						vhz = vhzlast;
						zabs = zabslast;
						gottime = 0;
					}
					else 
					{
						if (zabs >= phaseangle[j] && zabs <= phaseangle[j] + tolz) // particle position = phaseangle
						{
							break;
						}
						else 
						{
							if (zabs > phaseangle[j] + tolz) // particle position >> phaseangle
							{
								coiloff[j] = coiloffold + (ff/pow(2, ii))*dT; // try smaller stepsize
								ii++;
								
								if (j+2 <= nCoils)
								{
									/*
									 *  in order to avoid Majorana transitions, pulses will be overlapped:
									 *if trap is on, overlap the pulses of coil n and
									 * coil n+2 (coil n+1 = trap coil)
									 */
									coilon[j+1] = coiloff[j] - timeoverlap;
									
									// this yields the switch-on time for the next coil
									coiloff[j+1] = coilon[j+1] + addramp + ff*dT;
									
									// next coil turned off after the (shorter) ramptime plus some arb.
									// shift (guess)
									// if trap is on, turn off coil n+2 (coil n+1 = trap coil)
								}
								s = sold;
								vz = vzold;
								vhz = vhzold;
								zabs = zabsold;
								gottime = 0;
							}
						}
					}
				}
			}
		}
	}
	
	/* pulse duration:
	 * pulses may not be longer than "maxpulselength",
	 * else coils might explode
	 */
	for (int k = 0; k < nCoils; k++)
	{
		double duration = coiloff[k] - coilon[k];
		
		if (duration > maxpulselength)
		{
			if (k == nCoils - 1)
			{
				coiloff[k] = coilon[k] + maxpulselength;
				printf("Length of last pulse reduced to maximum pulse length. Final velocity higher than expected!");
			}
			else
			{
				printf("Maximum pulse duration exceeded for more than one coil! Change initial velocity and/or phi0.");
				return (-1);
			}
		}
	}
	
	printf("final velocity: %5.2f m/s\n", vz*1000);
	
	// round coil timings to multiples of 10 ns to prevent pulseblaster from
	// ignoring times shorter than that
	//coilon = np.round(coilon*100)/100
	//coiloff = np.round(coiloff*100)/100
	
	return(1);
}

// update current positions, based on current velocity
static inline void update_p(const double timestep)
{
	for (unsigned int i = 0; i < 3; i++)
	{
		pos[i] += vel[i]*timestep;
	}
	r = sqrt(pos[0]*pos[0] + pos[1]*pos[1]);
}

// check whether we are outside radius boundary or behind detection plane
// if either of these is true, we'll stop propagation
// otherwise we continue
static inline unsigned int check_positions()
{
	if (pos[2] >= skimmerdist && pos[2] <= skimmerdist + skimmerlength)
	{	//skimmer
		if (atan((r-skimmerradius)/(pos[2] - skimmerdist)) > skimmeralpha)
		{
			//printf("skimmer!\n");
			nLost++;
			return LOST;
		}
	}
	else if (r > coilrad && pos[2] > coilpos[0]-5 && pos[2] < coilpos[nCoils-1]+5) // 5 is due to width of coils
	{ //coils
		nLost++;
		return LOST;
	}
	else
	{
		if (pos[2] > endpos)
		{
			nDetected++;
			return DETECTED;
		}
	}
	return PROPAGATE;
}

// most of the difficult bits are in here
// this function both calculates the acceleration
// and updates the velocity based on that acceleration
// we first need to determine the magnetic field at the given
// position for the current time, where all coils can 
// potentially contribute
// we then calculate the acceleration for that field, and
// for the current zeeman state
// and finally update the velocity for that acceleration
static inline void update_v(unsigned long step, double timestep)
{
	double Bz_tot = 0;
	double Br_tot = 0;
	
	double Babs;
	
	double zrel;
	
	long z1, r1, idx1, idx3;
	double z1pos, r1pos, zp1pos, rp1pos;
	
	double QA_z, QB_z, QC_z, QD_z, QA_r, QB_r, QC_r, QD_r;
	double C1, C2, C3, C4;
	
	double QAB_z, QCD_z, QAC_z, QBD_z;
	double QAB_r, QCD_r, QAC_r, QBD_r;
	double gradBz_z, gradBz_r, gradBr_z, gradBr_r;
	double gradBabs_z, gradBabs_r;
	
	double accsum_r;
	double acc[3];
	
	double dEZee;
	
	double rampfactor;
	
	char pchanged = 0;
	
	unsigned long maxcoil = nCoils;
	
	// check if we see any field at this position and at this time
	for (unsigned long coil=0; coil < nCoils; coil++) // loop over all coils, calculate B-field contribution // not vectorized
	{
		rampfactor = currents[step*nCoils + coil];
		zrel = pos[2]-coilpos[coil];
		
		
		if (rampfactor > 0 && fabs(zrel) <= bzextend) // only look at this coil if there is current running through it at this time
		{
			pchanged = 1;
			// -zdist and -rdist to account for use of "ceil" interpolation between grid
			// positions z1/zp1 and r1/rp1
			// for particles inside the coil:
			// indices on grid (check by showing that zrel is between z1pos and zp1pos):
			z1 = lrint(ceil(zrel+bzextend)/zdist);
			r1 = lrint(ceil(r/rdist));
			
			// shift particles at the boundaries of the grid back one position into the grid
			if (z1 == sizZ) z1--;
			if (z1 == 0) z1++;
			if (r1 == sizR) r1--;
			if (r1 == 0) r1++;
			
			// positions on grid:
			z1pos = zaxis[z1-1];
			r1pos = raxis[r1-1];
			
			// tcalculate index into field array
			idx1 = r1+(z1-1)*sizB;
			idx3 = r1+z1*sizB;
			
			// grid points Bz field:
			QA_z = Bz[idx1-1];
			QB_z = Bz[idx1];
			QC_z = Bz[idx3-1];
			QD_z = Bz[idx3];
			
			// grid points Br field:
			QA_r = Br[idx1-1];
			QB_r = Br[idx1];
			QC_r = Br[idx3-1];
			QD_r = Br[idx3];
			
			C1 = (r-r1pos)/rdist;
			C2 = 1-C1;
			C3 = (zrel-z1pos)/zdist;
			C4 = 1-C3;
			
			QAB_z = C1*QB_z + C2*QA_z;
			QCD_z = C1*QD_z + C2*QC_z;
			QAC_z = C3*QC_z + C4*QA_z;
			QBD_z = C3*QD_z + C4*QB_z;
			
			QAB_r = C1*QB_r + C2*QA_r;
			QCD_r = C1*QD_r + C2*QC_r;
			QAC_r = C3*QC_r + C4*QA_r;
			QBD_r = C3*QD_r + C4*QB_r;
			
			gradBz_z += rampfactor*(QCD_z-QAB_z);
			gradBz_r += rampfactor*(QBD_z-QAC_z);
			gradBr_z += rampfactor*(QCD_r-QAB_r);
			gradBr_r += rampfactor*(QBD_r-QAC_r);
			
			// add to total Bz and Br
			Bz_tot += rampfactor*(C1*QBD_z + C2*QAC_z); // Bz
			Br_tot += rampfactor*(C1*QBD_r + C2*QAC_r); // Br
			
		}
	}
	
	// update velocity based on the field calculated above
	if (pchanged) // but only if we did see any magnetic field
	{
		gradBz_z /= zdist;
		gradBz_r /= rdist;
		gradBr_z /= zdist;
		gradBr_r /= rdist;
		
		// total B field:
		Babs = sqrt(Bz_tot*Bz_tot + Br_tot*Br_tot);
		if (Babs > 0)
		{
			gradBabs_z = (Bz_tot*gradBz_z + Br_tot*gradBr_z)/Babs; // chain rule
			gradBabs_r = (Bz_tot*gradBz_r + Br_tot*gradBr_r)/Babs;
			
			#if ELEMENT == HYDROGEN
			switch (zeemanState)
			{
				case 0: 
					dEZee = muB/particleMass*(1E-9/0.001); // already include mass here
					break;
				case 1: 
					dEZee = 1/sqrt(A*A*HBAR*HBAR*HBAR*HBAR/4/Babs/Babs/muB/muB/muB/muB + 1/muB/muB)/particleMass*(1E-9/0.001);
					break;
				case 2:
					dEZee = -muB/particleMass*(1E-9/0.001); // already include mass here
					break;
				case 3:
					dEZee = -1/sqrt(A*A*HBAR*HBAR*HBAR*HBAR/4/Babs/Babs/muB/muB/muB/muB + 1/muB/muB)/particleMass*(1E-9/0.001);
					break;
			}
			#elif ELEMENT == NITROGEN
			switch (zeemanState)
			{
				case 0: 
					dEZee = 5/2*muB/particleMass*(1E-9/0.001); // already include mass here // 0.001 belongs to gradient, 1e-9 to mass (?)
					break;
				case 1: 
					dEZee = 3/2*muB/particleMass*(1E-9/0.001);
					break;
				case 2:
					dEZee = 1/2*muB/particleMass*(1E-9/0.001);
					break;
				case 3:
					dEZee = -1/2*muB/particleMass*(1E-9/0.001);
					break;
				case 4:
					dEZee = -3/2*muB/particleMass*(1E-9/0.001);
					break;
				case 5:
					dEZee = -5/2*muB/particleMass*(1E-9/0.001);
					break;
			}
			#endif
			
			// if r = 0, then accsum_x = accsum_y = 0
			// (accsum_r and accsum_phi irrelevant)
			// prevent division by zero
			acc[2] = -gradBabs_z*dEZee;
			vel[2] += acc[2]*timestep;
			if (r != 0)
			{
				accsum_r = -gradBabs_r*dEZee;
				acc[0] = accsum_r/r*pos[0];
				vel[0] += acc[0]*timestep;
				acc[1] = accsum_r/r*pos[1];
				vel[1] += acc[1]*timestep;
			}
		}
		
	}
}

void doPropagate(double starttime, double maxtime, double timestep)
{
	//_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	//_mm_setcsr(_mm_getcsr() | 0x8040);
	
	double currentTime;
	unsigned long step;
	
	unsigned int result;
	
	for (unsigned long p = 0; p < nParticles; p++)
	{
		currentTime = starttime;
		step = 0;
		
		pos = &finalpos[3*p];
		vel = &finalvel[3*p];
		
		result = PROPAGATE;
		
		// no half step necessary at start, as a(t=0) always 0
		while (currentTime < maxtime - timestep)
		{
			step++;
			currentTime += timestep;
			
			// update x0 to x1 based on v0 and a0
			update_p(timestep);
			// check positions
			result = check_positions();
			
			if (result != PROPAGATE)
			{
				finaltime[p] = currentTime;
				//if (result != DETECTED)
					//vel[0] = 0;
					//vel[1] = 0;
					//vel[2] = 0;
				break;
			}
			
			// calculate acceleration and update v
			update_v(step, timestep);
		}
		
	}
	
	printf("--------calculations for zeeman state %zu--------\n", zeemanState);
	printf("number of particles lost: %zu\n", nLost);
	printf("number of particles reaching detection plane: %zu\n", nDetected);
	printf("number of particles timed out: %zu\n", nParticles-nDetected-nLost);
	
}
