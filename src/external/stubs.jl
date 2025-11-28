# Stubs for conditionally defined functions

"""
Wannerize the obtained bands using wannier90. By default all converged
bands from the `scfres` are employed (change with `n_bands` kwargs)
and `n_wannier = n_bands` wannier functions are computed.
Random Gaussians are used as guesses by default, can be changed
using the `projections` kwarg. All keyword arguments supported by
Wannier90 for the disentanglement may be added as keyword arguments.
The function returns the `fileprefix`.

!!! warning "Experimental feature"
    Currently this is an experimental feature, which has not yet been tested
    to full depth. The interface is considered unstable and may change
    incompatibly in the future. Use at your own risk and please report bugs
    in case you encounter any.
"""
function run_wannier90 end

"""
Run a SIRIUS input file in its standard JSON format. Pseudopotential files must be
provided according to the paths specified in the input file. The SIRIUS.jl package
must be imported with `using SIRIUS` prior to calling this function.
"""
function run_sirius_input end