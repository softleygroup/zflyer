Sim Zeeman
*********************

The :mod:`sim_zeeman` module contains the :class:`ZemmanFlyer` class that propagates a cloud
of particles through the Oxford Zeeman decelerator. The python class handles
the reading the configuration parameters, generation of the initial positions
and velocities, and saving the results. The equations of motion are integrated
by a library written in C `propagator_particle.c` which is automatically
compiled when the Python class is instantiated.

:mod:`sim_zeeman.py` can be run from the command line as a script to run a simulation from parameters in an input file, or the :class:`ZeemanFlyer` class can be used in other scripts, such as :mod:`optimise_zeeman.py`.

Running a Simulation
====================

If the :mod:`sim_zeeman.py` is run directly from the command line, the arguments are used to load configuration from an input file and start a simulation. 


Zeeman Decelerator Simulation
=============================

.. py:currentmodule:: sim_zeeman

Running a simulation with the :class:`ZeemanFlyer` class requires calling
functions in a sequence to initialise the simulation, then propagate particles
for each of a set of Zeeman levels. The seqence is described below.

Initialisation
--------------

First, the :class:`ZeemanFlyer` is instantiated:

.. automethod:: ZeemanFlyer.__init__

The parameters are expected to be stored in a file called `config.info` in
the working directory, given on the command line. This is given to the
:class:`ZeemanFlyer` object to load its simulation parameters. The script
exits if any parameters are missing or incorrect.

.. automethod:: ZeemanFlyer.loadParameters

Next, the object is instructed to generate all initial velocities and
positions.

.. automethod:: ZeemanFlyer.addParticles

The switching sequence is generated in one of two ways: either generating the sequence for a fixed phase angle, or loading a pre-computed sequence from the input file.

If a switching sequence needs to be calculated, it is done by the C library `propagatorParticle`

.. automethod:: ZeemanFlyer.calculateCoilSwitching

Now, the grid of magnetic field components through a coil is loaded. These have
been pre-calculated analytically.

.. automethod:: ZeemanFlyer.loadBFields

Finaly, the object is prepared for a calculation.

.. automethod:: ZeemanFlyer.preparePropagation

Flying Particles
----------------

The propagation must be called explicitly for each Zeeman state. States are
enumerated, starting from -1 for no Zeeman effect, then 0, 1, 2, 3 for the 4
Zeeman levels of hydrogen. The no Zeeman effect state is equivalent to running
with the decelerator off: particles will still be lost by collisions with
coils, but no acceleration is applied.

.. automethod:: ZeemanFlyer.propagate
