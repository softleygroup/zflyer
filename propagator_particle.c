/** Zeeman propagator
 *
 *  # Introduction
 *
 *  This program integrates the equations of motion for a collection of atoms
 *  ('bunch') flying through the zeeman decelerator from pulse-valve to detector.
 *  
 *
 * @author Atreju Tauschinsky
 * @copyright Copyright 2014 University of Oxford.
 */


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

static double particleMass = 0;
static int zeemanState = 0;

static double startTime = 0, timestep = 0;
static int nSteps = 0;
static int firstStep = 0;

static double *restrict pos0 = NULL;
static double *restrict vel0 = NULL;
static double *restrict pos = NULL;
static double *restrict vel = NULL;

static double r = 0;

static double *restrict finaltime = NULL;
static double *restrict finalpos = NULL, *restrict finalvel = NULL;

static double *restrict coilpos = NULL;
static double coilrad = 0;
static double endpos = 0;
static int nCoils = 0;
static double *restrict coilon = NULL; 
static double *restrict coiloff = NULL; 
static double *restrict current_buffer = NULL;

static double skimmerdist = 0, skimmerradius = 0, skimmeralpha = 0, skimmerlength = 0;

static double *restrict Bz=NULL, *restrict Br=NULL;
static double *restrict raxis=NULL, *restrict zaxis=NULL;
static double bzextend = 0, zdist = 0, rdist = 0;
static int sizZ = 0, sizR = 0, sizB = 0;

static double h1 = 0, h2 = 0, ramp1 = 0, rampcoil = 0, timeoverlap = 0, maxpulselength = 0;

void setSynchronousParticle(double particleMass_l, double p0[3], double v0[3])
{
    // helper to set the particle mass, initial position and initial velocity from python wrapper
    pos0 = p0;
    vel0 = v0;
    particleMass = particleMass_l;
}

void setBFields(double *Bz_l, double *Br_l, double *zaxis_l, double *raxis_l, double bzextend_l, double zdist_l, double rdist_l, int sizZ_l, int sizR_l, int sizB_l)
{
    // helper to set magnetic field properties from python wrapper
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

void setCoils(double *coilpos_l, const double coilrad_l, const double endpos_l, int nCoils_l)
{
    // helper to set coil properties from python wrapper
    coilpos = coilpos_l;
    coilrad = coilrad_l;
    endpos = endpos_l;
    nCoils = nCoils_l;
}

void setSkimmer(const double skimmerdist_l, const double skimmerlength_l, const double skimmerradius_l, const double skimmeralpha_l)
{
    // helper to set skimmer geometry from python wrapper
    skimmerdist = skimmerdist_l;
    skimmerlength = skimmerlength_l;
    skimmerradius = skimmerradius_l;
    skimmeralpha = skimmeralpha_l;
}

void setPropagationParameters(const double startTime_l,
        const double timestep_l, const int firstStep_l, const int nSteps_l) {
    // helper to set propagation parameters from python wrapper
    startTime = startTime_l;
    timestep = timestep_l;
    firstStep = firstStep_l;
    nSteps = nSteps_l;
}

void setTimingParameters(const double h1_l, const double h2_l,
        const double ramp1_l, const double timeoverlap_l,
        const double rampcoil_l, const double maxpulselength_l) {
    // helper to set coil timing parameters from python wrapper
    h1 = h1_l;
    h2 = h2_l;
    ramp1 = ramp1_l;
    timeoverlap = timeoverlap_l;
    rampcoil = rampcoil_l;
    maxpulselength = maxpulselength_l;
}

static inline double calculateRampFactor(int j, const double time,
        const double *restrict currents) {
    /* function to calculate the ramp factor given
     * a certain current time, coil, and on/offtimes for 
     * that coil, depending on coil ramping times and overlap
     * (mutual inductance) from other coilds
     */

    const double m1 = h1/ramp1;
    const double m2 = (1-h1)/timeoverlap;
    const double n2 = h1-m2*ramp1;
    const double m3 = -(1-h2)/timeoverlap;
    const double m4 = -h2/ramp1;

    const double ontime = coilon[j];
    const double offtime = coiloff[j];

    const double timediff = offtime - ontime;

    double rampfactor = 0;
    if (j == 0)
    {
        if (time <= ontime || fabs(ontime - offtime) < DBL_EPSILON)
            rampfactor = 0;
        else if (time > ontime && time < ontime+rampcoil) // normal rise
            rampfactor = (currents[j]/SIMCURRENT)*(1./rampcoil)*(time-ontime);
        else if (time >= ontime+rampcoil && time < offtime-timeoverlap) // constant level
            rampfactor = (currents[j]/SIMCURRENT);
        else if (time >= offtime-timeoverlap && time < offtime) // overlap fall
            rampfactor = (currents[j]/SIMCURRENT)*(m3*(time-ontime)+(h2-m3*timediff));
        else if (time >= offtime && time < offtime+ramp1) // rise 1 fall
            rampfactor = (currents[j]/SIMCURRENT)*(m4*(time-ontime)-m4*(timediff+ramp1));
    }
    else if (j == nCoils - 1)
    {
        if (time <= ontime || fabs(ontime - offtime) < DBL_EPSILON)
            rampfactor = 0;
        else if (time > ontime && time < ontime+ramp1) // rise 1 rise
            rampfactor = (currents[j]/SIMCURRENT)*(m1*(time-ontime));
        else if (time >= ontime+ramp1 && time < ontime+ramp1+timeoverlap) // overlap rise
            rampfactor = (currents[j]/SIMCURRENT)*(m2*(time-ontime)+n2);
        else if (time >= ontime+ramp1+timeoverlap && time < offtime) // constant level
            rampfactor = (currents[j]/SIMCURRENT);
        else if (time >= offtime && time < offtime+rampcoil) // normal fall
            rampfactor = (currents[j]/SIMCURRENT)*(1./rampcoil)*(offtime+rampcoil-time);
    }
    else
    {
        if (time <= ontime || fabs(ontime - offtime) < DBL_EPSILON)
            rampfactor = 0;
        else if (time > ontime && time < ontime+ramp1) // rise 1 rise
            rampfactor = (currents[j]/SIMCURRENT)*(m1*(time-ontime));
        else if (time >= ontime+ramp1 && time < ontime+ramp1+timeoverlap) // overlap rise
            rampfactor = (currents[j]/SIMCURRENT)*(m2*(time-ontime)+n2);
        else if (time >= ontime+ramp1+timeoverlap && time < offtime-timeoverlap) // constant level
            rampfactor = (currents[j]/SIMCURRENT);
        else if (time >= offtime-timeoverlap && time < offtime) // overlap fall
            rampfactor = (currents[j]/SIMCURRENT)*(m3*(time-ontime)+(h2-m3*timediff));
        else if (time >= offtime && time < offtime+ramp1) // rise 1 fall
            rampfactor = (currents[j]/SIMCURRENT)*(m4*(time-ontime)-m4*(timediff+ramp1));
    }

    return rampfactor;
}

void overwriteCoils(double * coilon_l, double * coiloff_l)
{

    coilon = __builtin_assume_aligned(coilon_l, 16);
    coiloff = __builtin_assume_aligned(coiloff_l, 16);
}

int precalculateCurrents(double * current_buffer_l, const double * currents)
{
    /* function to precalculate the currents
     * in each coil at a certain timestep.
     * doing this in advance significantly speeds
     * up propagation
     */

    if (coilon == NULL || coiloff == NULL)
    {
        printf("You have to calculate coil switching before calling this function!\n");
        return (-1);
    }
    if (nCoils == 0)
    {
        printf("You have to set coil properties before calling this function!\n");
        return (-1);
    }
    if (timestep < DBL_EPSILON || nSteps == 0)
    {
        printf("You have to set propagation parameter before calling this function!\n");
        return(-1);
    }

    current_buffer = __builtin_assume_aligned(current_buffer_l, 16);

    for (int coil = 0; coil < nCoils; coil++)
    {
        for (int s = 0; s < nSteps; s++)
        {
            current_buffer[s*nCoils + coil] = calculateRampFactor(coil, startTime+s*timestep, currents);
        }
    }
    return (0);
}

int calculateCoilSwitching(const double phase, const double dT, const double * bfieldz, double * coilon_l, double * coiloff_l, const double * currents)
{
    /* GENERATE THE PULSE SEQUENCE
    * using Zeeman effect = 1 (lfs, linear)
    * note: any Halbach hexapole arrays which are on axis
    * will not affect the motion of the
    * synchronous particle since Br = Bphi = Bz = 0 on axis
    * 
    * other variables:
    * gradB = sum(gradBz) = total B field gradient in a timestep
    * |B| = sum(Bz) = total B field in a timestep
    * rampfactor = to account for the finite rise/fall times
    *       of the coil B fields
    * accel, acc = acceleration of the synchronous particle
    * particle = generates a matrix for the particle [position velocity]
    * tol = tolerance for difference in switching time from previous and
    *       current iteration
    * s = timestep counter
    * oldcoil = switch-off time from last loop
    * lastcount = saves the last counter from the current loop
    */

    // Zeeman effect
    // derivative of the Zeeman effect in atomic hydrogen:
    // for pulse generation: take only lfs state |+,+> (linear):

    if (particleMass == 0 || pos0 == NULL || vel0 == NULL)
    {
        printf("Set properties of synchronous particle before calling this function\n!");
        return (-1);
    }
    if (coilpos == NULL)
    {
        printf("Set coil geometry before calling this function!\n");
        return (-1);
    }
    if (maxpulselength == 0)
    {
        printf("Set timing parameters before calling this function!\n");
        return (-1);
    }

    coilon = __builtin_assume_aligned(coilon_l, 16);
    coiloff = __builtin_assume_aligned(coiloff_l, 16);

    # if ELEMENT == HYDROGEN
    const double dEZee = muB;
    # elif ELEMENT == NITROGEN
    const double dEZee = 1.2*5./2*muB;
    #endif
    const int Oxsim = 1; // currently we only support fixed phase

    double Bz1, Bz2;
    double gradBtot_z, accsum_z;
    int field, index;
    double rampfactor;
    int sagain = 0, sold = 0;
    int foundalltimes = 0;

    // Optimization
    int s = 0; // time counter
    double zabs = pos0[2]; // initial position
    double vz = vel0[2]; // initial velocity
    double vzlast = 0, vhzlast = 0, vzold = 0, vhzold = 0;
    double zabslast = 0, zabsold = 0;
    double co;
    double coiloffold = 0;

    const double coildist = coilpos[1] - coilpos[0];
    const double bextend = -bfieldz[0];
    const double bdist = bfieldz[2] - bfieldz[0];

    // position of the synchr. particle when coils are switched off
    double phaseangle[nCoils];
    for (int i = 0; i < nCoils; i++)
    {
        phaseangle[i] = coilpos[i] - (coildist/180)*(90-phase);
    }

    const double tfirstcoil = (phaseangle[0]-coildist)/vz - 1.5*rampcoil;
    //const double tfirstcoil = (coilpos[0] - bextend)/vz + rampcoil;
    // version pre-25/01/2013, shall be used again when going for "real"
    // deceleration with 12 coils; I don't know what the other one is meant for!

    const double ff = 500.0;
    const double tolz = 0.005; // phase angle tolerance (in mm)

    coilon[0] = tfirstcoil;
    coiloff[0] = coilon[0] + rampcoil + ff*dT;
    coilon[1] = coiloff[0] - timeoverlap;
    coiloff[1] = coilon[1] + ramp1 + timeoverlap + ff*dT;

    // Preallocating vectors for speeding up the program
    double time = 0.;

    double vhz = vz; // dummy velocity half-step
    int cycles = 0; // number of cycles needed to get the pulse sequence

    for (int j = 0; j < nCoils; j++)
    {
        int ii = 1;
        int gottime = 0;

        int leftcoil = j > 1 ? j - 1 : 0;
        int rightcoil = j + 3 < nCoils ? j + 3 : nCoils - 1;

        while (1)
        {
            if (cycles > 1E3)
            {
                printf("Maximum number of iterations reached without convergence. Change initial parameters.\n");
                return (-1);
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
                if ((coilon[jj] != 0) && (fabs(zabs - coilpos[jj]) < bextend) && (time >= coilon[jj]) && (time <= coiloff[jj] + rampcoil))
                {
                    field = 1;
                    rampfactor = calculateRampFactor(jj, time, currents);
                    index = (int)ceil((zabs - coilpos[jj] + bextend)/bdist);
                    Bz1 += rampfactor*bfieldz[2*index - 1];
                    Bz2 += rampfactor*bfieldz[2*index + 1];
                }
            }


            // Numerical integration of the equations of motion
            // using the Velocity Verlet method (??)
            // remember: zabs = zabs(s), vz = vz(s-1)
            //if (time >= (coiloff[j]-timeoverlap-ff*dT) && gottime == 0)
            if (gottime == 0 && time >= coilon[j])
            {
                sagain = s - 1;
                vzlast = vz;
                vhzlast = vhz;
                zabslast = zabs;
                gottime = 1;
            }

            if (field == 1)
            {
                // total gradient of B (only Bz component needed)
                gradBtot_z = (sqrt(Bz2*Bz2)-sqrt(Bz1*Bz1))/bdist;
                // Determine acceleration
                accsum_z = -(gradBtot_z/0.001)*dEZee/particleMass*1E-9;

                vz = vhz + 0.5*dT*accsum_z;
                vhz = vz + 0.5*dT*accsum_z;

            }
            zabs = zabs + dT*vhz;


            /* 
            * scheme: want phaseangle(j) to be the position at which the coil is
            * completely switched off --> hence, guess coil(j, 3) and then iterate until
            * coil(j, 3) + c.rampcoil, if phaseangle(j) is not reached, add some time to
            * coil(j, 3) and go back to old coil(j, 3) and integrate from there again
            */

            if (j == nCoils - 1)
            {
                co = coiloff[j] + rampcoil;
            }
            else
            {
                co = coiloff[j] + ramp1;
            }

            if (time >= co && foundalltimes == 0)
            {
                if (zabs < phaseangle[j])
                {
                    coiloffold = coiloff[j];
                    cycles++;
                    sold = sagain;
                    vzold = vzlast;
                    vhzold = vhzlast;
                    zabsold = zabslast;

                    coiloff[j] += ff*dT;

                    s =  sagain;
                    vz = vzlast;
                    vhz = vhzlast;
                    zabs = zabslast;
                    gottime = 0;
                }
                else if (zabs >= phaseangle[j] && zabs <= phaseangle[j] + tolz)
                {
                    if (Oxsim == 2 && j < nCoils - 1)
                    {
                        printf("Adaptive phase angle is currently not supported!");
                        return (-1);
                    }

                    if (j == nCoils - 1)
                    {
                        foundalltimes = 1;
                    }
                    else
                    {
                        coilon[j + 1] = coiloff[j] - timeoverlap;
                        coiloff[j + 1] = coilon[j + 1] + rampcoil + ff*dT;
                    }
                    break;
                }
                else if (zabs > phaseangle[j] + tolz)
                {
                    coiloff[j] = coiloffold + (ff/pow(2, ii)) * dT;
                    ii++;
                    s = sold;
                    vz = vzold;
                    vhz = vhzold;
                    zabs = zabsold;
                    gottime = 0;
                }

                if (j < nCoils - 1)
                {
                    coilon[j + 1] = coiloff[j] - timeoverlap;
                    coiloff[j + 1] = coilon[j + 1] + rampcoil + ff*dT;
                }
            }

            if (vz < 20/1000)
            {
                printf("Particle is decelerated to v_z < 20 m/s: reduce phaseangle, increase initial velocity or decrease the number of coils to be switched.\n");
                return (-1);

            }
        }
    }

    /* pulse duration:
     * pulses may not be longer than "maxpulselength",
     * else coils might explode
     */
    for (int k = 0; k < nCoils; k++)
    {
        // round to two decimal places, as the pulseblaster only has 10ns resolution
        // coiloff[k] = roundf(100*coiloff[k])/100;
        // coilon[k] = roundf(100*coilon[k])/100;

        // calculate pulse duration and check length
        if (ELEMENT==NITROGEN){
            coiloff[k] += 15;
            coilon[k] += 15;
        }
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

    return(0);
}

static inline void update_p() {
    // update current positions, based on current velocity
    for (int i = 0; i < 3; i++) {
        pos[i] += vel[i]*timestep;
    }
    r = sqrt(pos[0]*pos[0] + pos[1]*pos[1]);
}

static inline int check_positions() {
    /* check whether we are outside radius boundary or behind detection plane
     * if either of these is true, we'll stop propagation
     * otherwise we continue
     */

    if (pos[2] >= skimmerdist && pos[2] <= skimmerdist + skimmerlength)
    {   // if this is true we're inside the skimmer along z
        if (atan((r-skimmerradius)/(pos[2] - skimmerdist)) > skimmeralpha)
        {
            // and if this is true we're hitting the wall of the skimmer
            return LOST;
        }
    }
    // else if (pos[2] > 227. && pos[2] < 228. && r > .6)
    // {
    //  return LOST; // aperture! just for testing!
    // }
    else if (r > coilrad && pos[2] > coilpos[0]-5 && pos[2] < coilpos[nCoils-1]+5) // 5 is due to width of coils
    { 
        // here we're somewhere in the decelerator (i.e. between first and last coil)
        // and hitting the wall
        return LOST;
    }
    else
    {
        if (pos[2] > endpos)
        {
            // we've made it to the detection plane
            return DETECTED;
        }
    }
    // nothing stopping us, so we'll continue to propagate this particle
    return PROPAGATE;
}

static inline void update_v(int step) {
    /* most of the difficult bits are in here. This function both calculates
     * the acceleration and updates the velocity based on that acceleration. We
     * first need to determine the magnetic field at the given position for the
     * current time, where all coils can potentially contribute. We then
     * calculate the acceleration for that field, and for the current Zeeman
     * state and finally update the velocity for that acceleration.
     */

    double Bz_tot = 0;
    double Br_tot = 0;
    double Babs;
    double zrel;
    int z1, r1, idx1, idx3;
    double z1pos, r1pos;
    double QA_z, QB_z, QC_z, QD_z, QA_r, QB_r, QC_r, QD_r;
    double C1, C2, C3, C4;
    double QAB_z, QCD_z, QAC_z, QBD_z;
    double QAB_r, QCD_r, QAC_r, QBD_r;
    double gradBz_z = 0, gradBz_r = 0, gradBr_z = 0, gradBr_r = 0;
    double gradBabs_z = 0, gradBabs_r = 0;
    double accsum_r;
    double acc[3];
    double dEZee = 0;
    double rampfactor;
    char pchanged = 0;

    // check if we see any field at this position and at this time
    for (int coil=0; coil < nCoils; coil++) {
        // loop over all coils, calculate B-field contribution
        // not vectorized
        rampfactor = current_buffer[step*nCoils + coil];
        zrel = pos[2]-coilpos[coil];

        if (rampfactor > 0 && fabs(zrel) <= bzextend) {
            // only look at this coil if there is current running through it at
            // this time
            pchanged = 1;
            // -zdist and -rdist to account for use of "ceil" interpolation
            // between grid positions z1/zp1 and r1/rp1 for particles inside
            // the coil: indices on grid (check by showing that zrel is between
            // z1pos and zp1pos):
            z1 = (int)ceil((zrel+bzextend)/zdist);
            r1 = (int)ceil(r/rdist);

            // shift particles at the boundaries of the grid back one position
            // into the grid.
            z1 = z1 >= sizZ ? sizZ - 1 : z1;
            z1 = z1 == 0 ? 1 : z1;
            r1 = r1 >= sizR ? sizR - 1 : r1;
            r1 = r1 == 0 ? 1 : r1;

            // positions on grid:
            z1pos = zaxis[z1 - 1];
            r1pos = raxis[r1 - 1];

            // tcalculate index into field array
            idx1 = r1 + (z1 - 1)*sizB;
            idx3 = r1 + z1*sizB;

            // grid points Bz field:
            QA_z = Bz[idx1 - 1];
            QB_z = Bz[idx1];
            QC_z = Bz[idx3 - 1];
            QD_z = Bz[idx3];

            // grid points Br field:
            QA_r = Br[idx1 - 1];
            QB_r = Br[idx1];
            QC_r = Br[idx3 - 1];
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
    if (pchanged) {
        // but only if we did see any magnetic field
        gradBz_z /= zdist;
        gradBz_r /= rdist;
        gradBr_z /= zdist;
        gradBr_r /= rdist;

        // total B field:
        Babs = sqrt(Bz_tot*Bz_tot + Br_tot*Br_tot);
        if (Babs > 0) {
            gradBabs_z = (Bz_tot*gradBz_z + Br_tot*gradBr_z)/Babs; // chain rule
            gradBabs_r = (Bz_tot*gradBz_r + Br_tot*gradBr_r)/Babs;

            #if ELEMENT == HYDROGEN
            switch (zeemanState) {
                // already include mass here
                case 0:
                    dEZee = muB/particleMass*(1E-9/0.001);
                    break;
                case 1: 
                    dEZee = 1/sqrt(A*A*HBAR*HBAR*HBAR*HBAR/4/Babs/Babs/muB/muB/muB/muB + 1/muB/muB)/particleMass*(1E-9/0.001);
                    break;
                case 2:
                    dEZee = -muB/particleMass*(1E-9/0.001);
                    break;
                case 3:
                    dEZee = -1/sqrt(A*A*HBAR*HBAR*HBAR*HBAR/4/Babs/Babs/muB/muB/muB/muB + 1/muB/muB)/particleMass*(1E-9/0.001);
                    break;
            }
            #elif ELEMENT == NITROGEN
            // we assume that all states shift linearly, so we don't need a
            // switch/case, we can simply calculate the coefficient directly
            // from zeeman state (0...5) only for j=5/2 at the moment, 1.2 = gj

            // already include mass here
            // 0.001 belongs to gradient, 1e-9 to mass (?)
            dEZee = 1.200318*(5.-2.*zeemanState)/2.*muB/particleMass*(1E-9/0.001);
            #endif

            // if r = 0, then accsum_x = accsum_y = 0
            // (accsum_r and accsum_phi irrelevant)
            // prevent division by zero
            acc[2] = -gradBabs_z*dEZee;
            vel[2] += acc[2]*timestep;
            if (r != 0) {
                accsum_r = -gradBabs_r*dEZee;
                acc[0] = accsum_r/r*pos[0];
                vel[0] += acc[0]*timestep;
                acc[1] = accsum_r/r*pos[1];
                vel[1] += acc[1]*timestep;
            }
        }
    }
}

void doPropagate(double * finalpos_l, double * finalvel_l,
        double * finaltime_l, int nParticles, int zeemanState_l) {
    /* main function to be called from python wrapper to start propagation
     * finalpos and finalvel contain the *initial* position and velocities of
     * the particles to be propagated, and will be updated to the final
     * positions and velocities by this function; finaltime is initially zero,
     * and will be updated with the last time at which a particle could move
     * (because it was either detected or lost in the next step) a zeeman state
     * of -1 indicates gas-pulse propagation with the decelerator off. other
     * zeeman states are 0...3 in order from lfs to hfs for hydrogen or 0...5
     * for N
     */

    int nDetected = 0;
    int nLost = 0;

    finalpos = __builtin_assume_aligned(finalpos_l, 16);
    finalvel = __builtin_assume_aligned(finalvel_l, 16);
    finaltime = __builtin_assume_aligned(finaltime_l, 16);
    zeemanState = zeemanState_l;

    double currentTime;
    int step;
    int result;

    for (int p = 0; p < nParticles; p++) {
        currentTime = startTime;

        pos = &finalpos[3*p];
        vel = &finalvel[3*p];

        // no half step necessary at start, as a(t=0) always 0
        for (step = firstStep; step < firstStep + nSteps; step++) {
            currentTime += timestep;

            // update x0 to x1 based on v0 and a0
            update_p();
            // check positions
            result = check_positions();

            if (result != PROPAGATE) {
                finaltime[p] = currentTime;
                if (result == DETECTED) {
                    nDetected++;
                } else {
                    nLost++;
                    // mark lost particles by setting their velocity to zero
                    vel[0] = 0;
                    vel[1] = 0;
                    vel[2] = 0;
                }
                break;
            }

            /* For zeemanState < 0 we don't change the velocity, because we're
             * calculating the pulse with the decelerator turned off. Just
             * including this state with an if-clause will increase execution
             * time slightly.
             */
            if (zeemanState >= 0) {
                // calculate acceleration and update v
                update_v(step);
            }
        }
    }

    // printf("--------calculations for zeeman state %d--------\n", zeemanState);
    // printf("number of particles lost: %d\n", nLost);
    // printf("number of particles reaching detection plane: %d\n", nDetected);
    // printf("number of particles timed out: %d\n", nParticles-nDetected-nLost);
}
