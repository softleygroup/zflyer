#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include <xmmintrin.h>

void setInitialBunch(double *, double *, double *, unsigned long, double, unsigned long);
void setBFields(double *, double *, double *, double *, double, double, double, unsigned long, unsigned long, unsigned long);
void setCoils(double *, double *, double, double, unsigned long);
void doPropagate(double, double, double);
void setSkimmer(const double, const double, const double, const double);
int calculateCoilSwitching(const double, const double, const double, const double, const double *, double *, double *);
void setTimingParameters(const double, const double, const double, const double, const double, const double);
