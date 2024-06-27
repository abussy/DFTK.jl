### A collection of utility functions for SIRIUS in DFTK

function set_sirius_density(basis::SiriusBasis, ρ)

    SIRIUS.set_periodic_function(basis.sirius_gs, "rho"; f_rg=ρ, size_x=size(ρ)[1], size_y=size(ρ)[2],
                                 size_z=size(ρ)[3], offset_z=-1)
    #make PW density available in SIRIUS
    SIRIUS.fft_transform(basis.sirius_gs, "rho", -1)
end

function sirius_set_occupation(basis::SiriusBasis, occupation)
    nkp = length(basis.kpoints)    
    ispin = 1 #TODO nspins = 2 case
    for ikp = 1:nkp
        SIRIUS.set_band_occupancies(basis.sirius_kps, ik_global(ikp, basis), ispin, occupation[ikp])
    end
end

function compute_occupation(basis::SiriusBasis, eigenvalues::AbstractVector,
                            fermialg::AbstractFermiAlgorithm=default_fermialg(basis.model);
                            kwargs ...)
    occupation, εF = compute_occupation(basis.pw_basis, eigenvalues, fermialg; kwargs ...)
    #Make sure that SIRIUS is up to date
    sirius_set_occupation(basis, occupation)
    SIRIUS.set_energy_fermi(basis.sirius_kps, εF)
    return occupation, εF
end

function guess_density(basis::SiriusBasis)
    ρ = Array{Cdouble, 4}(undef, basis.fft_size[1], basis.fft_size[2], basis.fft_size[3], 1)
    SIRIUS.generate_initial_density(basis.sirius_gs)
    SIRIUS.get_periodic_function!(basis.sirius_gs, "rho"; f_rg=ρ, size_x=basis.fft_size[1], 
                                  size_y=basis.fft_size[2], size_z=basis.fft_size[3], offset_z=-1)
    return ρ
end

function ik_global(ik, basis)
    return basis.krange_thisproc[1][ik]
end

default_diagtolalg(basis::SiriusBasis; tol, kwargs...) = AdaptiveDiagtol()

function mix_density(mixing, basis::SiriusBasis, Δρ; kwargs...)
    if mixing isa χ0Mixing
        error("x0Mixing is not supported with Sirius")
    end
    mix_density(mixing, basis.pw_basis, Δρ; kwargs...)
end

function compute_density(basis::SiriusBasis, ψ, occupation; kwargs ...)
    compute_density(basis.pw_basis, ψ, occupation; kwargs ...)
end

function diagonalize_all_kblocks(eigensolver, ham::SiriusHamiltonian, nev_per_kpoint::Int; kwargs ...)
    #Note: eigensolver and kwargs are only there to match the function signature for multiple dispatch

    SIRIUS.set_num_bands(ham.basis.sirius_ctx, nev_per_kpoint) 
    #tol is passed by kwargs as determine_diagtol, which is way too big and variable
    #TODO: figure out why and fix it. 
    sirius_diagonalize(ham, nev_per_kpoint)#; kwargs...) 
end