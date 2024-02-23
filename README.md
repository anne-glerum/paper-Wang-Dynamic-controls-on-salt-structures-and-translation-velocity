This repository belongs to the paper

*Dynamic controls on salt structures and translation velocity at continental rifted margins*

by

Wang, Z.,
Brune, S.,
Neuharth, D.,
Glerum, A.,
Rouby, D., and
Fernandez, N.

currently in preparation.

# Documentation
The numerical simulations presented in this paper were run with the geodynamics code ASPECT ([https://aspect.geodynamics.org/](https://aspect.geodynamics.org/)) coupled to the surface processes code FastScape ([https://fastscape.org/fastscapelib-fortran/](https://fastscape.org/fastscapelib-fortran/)).


## ASPECT version
The ASPECT input files provided in this repository correspond to commit 984cca1 of the ASPECT branch 

[fastscape_update_again_erosional_base_level-undo2780-before-rebase-on13mai22-Kais-and-ZhiChens-Version](https://github.com/EstherHeck/aspect/tree/fastscape_update_again_erosional_base_level-undo2780-before-rebase-on13mai22-Kais-and-ZhiChens-Version)

This branch is built on commit 84d40e745328f62df1a09e15a9f1bb4fdc86141a of the ASPECT 2.4.0-pre development branch,
which can be found at [https://github.com/geodynamics/aspect](https://github.com/geodynamics/aspect). 
A copy of 984cca1 can be found in the folder /src_ASPECT.

## Additional ASPECT plugins
For the initial model conditions, we used the ASPECT plugins in the folder /plugins. 
The file CMakeLists.txt can be used to install these plugins as shared libraries
against your ASPECT installation with:

1. Enter plugins directory
2. cmake -DAspect_DIR=/path/to/ASPECT/installation/ .
3. make

## FastScape version

The FastScape source code provided in this repository corresponds to commit 03a056c8673cba01f53875495c76c1fd72259e66 
of the FastScape branch [https://github.com/EstherHeck/fastscapelib-fortran/tree/fastscape-with-stratigraphy-for-aspect](fastscape-with-stratigraphy-for-aspect) 
and can be found in the folder /src_FastScape. This branch is built on commit 18f25888b16bf4cf23b00e79840bebed8b72d303 of 
[https://github.com/fastscape-lem/fastscapelib-fortran](https://github.com/fastscape-lem/fastscapelib-fortran).


## ASPECT input files
The ASPECT input files can be found in the respective folders of each simulation presentated in the manuscript in the folder /prm_files. For each simulation, three prm files are provided, corresponding to the pre, syn and post salt deposition phases.

## FastScape installation details
The FastScape version in this repository can by installed by:
1. Cloning this repository
2. Creating a build directory and entering it 
3. cmake -DBUILD_FASTSCAPELIB_SHARED=ON /path/to/fastscape/dir/
4. make

## ASPECT Installation details
ASPECT was built using the underlying library deal.II 9.3.0
on the German HLRN cluster Lise. deal.II used:
* 32 bit indices and vectorization level 3 (512 bits)
* Trilinos 12.18.1
* p4est 2.2.0

ASPECT used the following settings:

        ASPECT_VERSION:            2.4.0-pre
        CMAKE_BUILD_TYPE:          Release
        DEAL_II VERSION:           9.3.0
        ASPECT_USE_FP_EXCEPTIONS:  ON
        ASPECT_RUN_ALL_TESTS:      OFF
        ASPECT_USE_SHARED_LIBS:    ON
        ASPECT_HAVE_LINK_H:        ON
        ASPECT_WITH_LIBDAP:        OFF
        ASPECT_PRECOMPILE_HEADERS: ON
        ASPECT_UNITY_BUILD:        ON

        CMAKE_CXX_COMPILER:        GNU 9.2.0 on platform Linux x86_64
                                   /sw/comm/openmpi/3.1.5/skl/gcc/bin/mpicxx

        LINKAGE:                   DYNAMIC

        COMPILE_FLAGS:             
        _WITH_CXX14:               ON
        _WITH_CXX17:               FALSE
        _MPI_VERSION:              3.1
        _WITH_64BIT_INDICES:       OFF

The ASPECT version in this repository can be installed by (assuming deal.II is installed):
1. Creating a build directory and entering it
2. cmake -DEAL_II_DIR=/path/to/dealii/dir/ -DFASTSCAPE_DIR=/path/to/fastscape/build/dir/ path/to/aspect/dir/
3. make

## Postprocessing
Snapshot images of model results were created with the ParaView 5.7.0 statefile paraview_statefile.pvsm in the folder /post_processing_scripts.
Plots of velocity, salt thickness, gradient and sediment thickness over time were created with python scripts that can also be found in the /post_processing_scripts folder, with the scripts in /post_processing_scripts/postprocess/scatterplot_1-extract_data used to extract the data and post_processing_scripts/postprocess/scatterplot_2-plot.py used to plot them.
