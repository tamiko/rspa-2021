Computational resources for "Lorentz Resonance in the Homogenization of Plasmonic Crystals"
====

This repository contains all computational resources that were used for
computations in the aforementioned publication.

All compuations were done with small C++ programs based on the deal.II
finite element library, freely available at https://www.dealii.org and
https://github.com/dealii/dealii. See
https://www.dealii.org/current/readme.html and
https://www.dealii.org/current/doxygen/deal.II/Tutorial.html for more
information about how to install deal.II and run a program based on
deal.II.

The repository structure is as follows:

sobek
-----

This subdirectory contains the sources for a program computing the
effective permitivity tensor with the help of the cell problem and
averaging. Parameter files for two prototypical configurations (nanotubes
and nanoribbons) used in the publication are located in the `prm`
directory.

sobek-eigenvalue
----------------

This subdirectory contains the sources for a program computing the spectrum
and associated coupling constants for the corresponding eigenvalue problem.
Parameter files for two prototypical configurations (nanotubes and
nanoribbons) used in the publication are located in the `prm` directory.

How to compile and run the programs
-----------------------------------

Both programs are written in the form of simple deal.II example steps,
simply configure via `cmake .` and compile via `make`. After successful
compilation an executable `sobek` is located in a `run` directory. Simply
copy and rename one of the two example parameter files to `run/sobek.prm`
and invoke the executable.
