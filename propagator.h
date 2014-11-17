#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <float.h>

#include <xmmintrin.h>

void setSynchronousParticle(double, double *, double *);
void setBFields(double *, double *, double *, double *, double, double, double, int, int, int);
void setCoils(double *, const double, const double, int);
void setSkimmer(const double, const double, const double, const double);
void doPropagate(double *, double *, double *, int, int);
void setTimingParameters(const double, const double, const double, const double, const double, const double);
int calculateCoilSwitching(const double, const double, const double *, double *, double *, const double *);
int precalculateCurrents(double *, const double *);
void setPropagationParameters(const double, const double, const int, const int);
void overwriteCoils(double *, double *);