Sim Zeeman
*********************

The :mod:`sim_zeeman` module contains the :class:`ZemmanFlyer` class that propagates a cloud
of particles through the Oxford Zeeman decelerator. The python class handles
the reading the configuration parameters, generation of the initial positions
and velocities, and saving the results. The equations of motion are integrated
by a library written in C – propagator_particle.c – which is automatically
compiled when the Python class is instantiated.

:mod:`sim_zeeman.py` can be run from the command line as a script to run a simulation from parameters in an input file, or the :class:`ZeemanFlyer` class can be used in other scripts, such as :mod:`optimise_zeeman.py`.

Running a Simulation
====================



Zeeman Decelerator Simulation
=============================
.. py:module:: sim_zeeman

If the :mod:`sim_zeeman.py` is run directly from the command line, the arguments are used to load configuration from an input file and start a simulation.

First, the :class:`ZeemanFlyer` is instantiated:

.. automethod:: ZeemanFlyer.__init__

The parameters are expected to be stored in a file called `config.info` in the working directory, given on the command line. This is given to the :class:`ZeemanFlyer` object to load its simulation parameters.

.. automethod:: ZeemanFlyer.loadParameters


