#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include <xmmintrin.h>

void setSynchronousParticle(double, double[3], double[3]);
void setBFields(double *, double *, double *, double *, double, double, double, unsigned int, unsigned int, unsigned int);
void setCoils(double *, double, double, unsigned int);
void doPropagate(double *, double *, double *, unsigned int, int);
void setSkimmer(const double, const double, const double, const double);
int calculateCoilSwitching(const double, const double, const double *, double *, double *);
void setTimingParameters(const double, const double, const double, const double, const double, const double);
int precalculateCurrents(double *);
void setPropagationParameters(const double, const double, const double);
