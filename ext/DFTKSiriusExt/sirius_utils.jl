### This file contains a collection of utility functions for SIRIUS in DFTK

# Copy an input DFTK density to SIRIUS
@DFTK.timing function set_sirius_density(basis::SiriusBasis, ρ)

    SIRIUS.set_periodic_function(basis.sirius_gs, "rho"; f_rg=total_density(ρ), size_x=size(ρ)[1], 
                                 size_y=size(ρ)[2], size_z=size(ρ)[3], offset_z=-1)

    # make PW density available in SIRIUS (from real to reciprocal space)
    SIRIUS.fft_transform(basis.sirius_gs, "rho", -1)

    # Note: SIRIUS works with total and spin-density, and not spin-resolved densities as DFTK
    if basis.model.n_spin_components == 2
        SIRIUS.set_periodic_function(basis.sirius_gs, "magz", f_rg=spin_density(ρ),size_x=size(ρ)[1],
                                     size_y=size(ρ)[2], size_z=size(ρ)[3], offset_z=-1)
        SIRIUS.fft_transform(basis.sirius_gs, "magz", -1)
    end
end

# Set SIRIUS band occupation
function sirius_set_occupation(basis::SiriusBasis, occupation)
    nkp = length(basis.kpoints)    
    for ikp = 1:nkp
        ik_global, ispin = ik_global_and_spin(ikp, basis)
        SIRIUS.set_band_occupancies(basis.sirius_kps, ik_global, ispin, occupation[ikp])
    end
end

# Set SIRIUS band energies
function sirius_set_band_energies(basis::SiriusBasis, energies)
    nkp = length(basis.kpoints)    
    for ikp = 1:nkp
        ik_global, ispin = ik_global_and_spin(ikp, basis)
        SIRIUS.set_band_energies(basis.sirius_kps, ik_global, ispin, energies[ikp])
    end
end

# Compute band occupation on the DFTK side, and update SIRIUS accordingly
function compute_occupation(basis::SiriusBasis, eigenvalues::AbstractVector,
                            fermialg::AbstractFermiAlgorithm=default_fermialg(basis.model);
                            kwargs ...)
    occupation, εF = compute_occupation(basis.pw_basis, eigenvalues, fermialg; kwargs ...)
    #Make sure that SIRIUS is up to date
    #TODO: this is necessary to get the energies correctly from SIRIUS, but maybe should compute
    #      them in the DFTK side (at least the OneElectron term, possibly the Entropy too)
    #      Question is: could we still compute the forces?
    SIRIUS.set_num_bands(basis.sirius_ctx, length(eigenvalues[1]))
    sirius_set_band_energies(basis, eigenvalues)
    sirius_set_occupation(basis, occupation)
    SIRIUS.set_energy_fermi(basis.sirius_kps, εF)
    return occupation, εF
end

# Compute band occupation given a fixed Fermi level (forwarded to pw_basis attribute)
function compute_occupation(basis::SiriusBasis, eigenvalues::AbstractVector, εF::Number;
                            temperature=basis.model.temperature,
                            smearing=basis.model.smearing)
    compute_occupation(basis.pw_basis, eigenvalues, εF; 
                       temperature=temperature, smearing=smearing)
end

# Generate a guess density for a SiriusBasis
@DFTK.timing function guess_density(basis::SiriusBasis, magnetic_moments=[],
                       n_electrons=basis.model.n_electrons; use_dftk_guess=true)

    # By default, use the DFTK guess. It insures 100% compatibility with existing DFTK code and results.
    # In particular, note that the magnetic guess of DFTK and SIRIUS are slighlty different.
    # In the long term, we might want to add a test on the PP, because DFTK won't have anything
    # for ulstrasoft pseudos.
    # Note that for 100% compatibility, using the DFTK guess as a default is safer, because SIRIUS
    # crashes when the initial density is missing from the PP file (e.g. with some UPF HGH PPs)
    if !use_dftk_guess
        ρ = Array{Cdouble, 4}(undef, basis.fft_size[1], basis.fft_size[2], basis.fft_size[3],
                              basis.model.n_spin_components)

        if basis.model.n_spin_components == 2
            if length(magnetic_moments) > 0
                for (iat, magmom) in enumerate(magnetic_moments)
                    SIRIUS.set_atom_vector_field(basis.sirius_ctx, iat, [0.0, 0.0, magmom])
                end
            end
        end
        SIRIUS.generate_initial_density(basis.sirius_gs)

        tmp  = @view ρ[:, :, :, 1]
        SIRIUS.get_periodic_function!(basis.sirius_gs, "rho"; f_rg=tmp, size_x=basis.fft_size[1],
                                      size_y=basis.fft_size[2], size_z=basis.fft_size[3], offset_z=-1)

        if basis.model.n_spin_components == 2
            mag = Array{Cdouble, 3}(undef, basis.fft_size[1], basis.fft_size[2], basis.fft_size[3])
            SIRIUS.get_periodic_function!(basis.sirius_gs, "magz"; f_rg=mag, size_x=basis.fft_size[1],
                                          size_y=basis.fft_size[2], size_z=basis.fft_size[3], offset_z=-1)
            ρ = ρ_from_total_and_spin(tmp, mag)
        end
    else
        ρ = guess_density(basis.pw_basis, magnetic_moments, n_electrons)
    end
    return ρ
end

# Given a DFTK local k-point index, return the global k-point index and spin.
# In SIRIUS, k-points are referred to with a global index and spin
function ik_global_and_spin(ik, basis)
    ik_global_dftk = basis.krange_thisproc_allspin[ik]
    nkpt_spin = length(basis.kcoords_global)
    if ik_global_dftk > nkpt_spin
        return ik_global_dftk-nkpt_spin, 2
    end
    return ik_global_dftk, 1
end

### A series of usefull functions where to computation is forwarded to the pw_basis attribute
function krange_spin(basis::SiriusBasis, spin::Integer)
    krange_spin(basis.pw_basis, spin)
end

default_diagtolalg(basis::SiriusBasis; tol, kwargs...) = AdaptiveDiagtol()

function mix_density(mixing, basis::SiriusBasis, Δρ; kwargs...)
    mix_density(mixing, basis.pw_basis, Δρ; kwargs...)
end

compute_density(basis::SiriusBasis, ψ, occupation; kwargs ...) = compute_density(basis.pw_basis, ψ, occupation; kwargs ...)
#function compute_density(basis::SiriusBasis, ψ, occupation; kwargs ...)
#    #TODO: for now, compute the density in SIRIUS, as FFTs are more efficient
#    #compute_density(basis.pw_basis, ψ, occupation; kwargs ...)
#    ρ = Array{Cdouble, 4}(undef, basis.fft_size[1], basis.fft_size[2], basis.fft_size[3],
#                          basis.model.n_spin_components)
#    SIRIUS.generate_density(basis.sirius_gs; add_core=true, transform_to_rg=true)
#    tmp  = @view ρ[:, :, :, 1]
#    SIRIUS.get_periodic_function!(basis.sirius_gs, "rho"; f_rg=tmp, size_x=basis.fft_size[1],
#                                  size_y=basis.fft_size[2], size_z=basis.fft_size[3], offset_z=-1)
#
#    if basis.model.n_spin_components == 2
#        mag = Array{Cdouble, 3}(undef, basis.fft_size[1], basis.fft_size[2], basis.fft_size[3])
#        SIRIUS.get_periodic_function!(basis.sirius_gs, "magz"; f_rg=mag, size_x=basis.fft_size[1],
#                                      size_y=basis.fft_size[2], size_z=basis.fft_size[3], offset_z=-1)
#        ρ = ρ_from_total_and_spin(tmp, mag)
#    end 
#    ρ
#end

# Querry the energy from SIRIUS. Note that the terms do not have a 1 to 1 correspondance with DFTK:
# The kinetic, local, non-local and psp_correction terms are gather into a single OneElectron term
function get_sirius_energies(basis::SiriusBasis)

    #TODO: would need to add terms in case of PAW or full potential
    dftk_to_sirius = Dict()
    dftk_to_sirius["OneElectron"] = [Dict("name" => "one-el", "fac" => -1.0),
                                     Dict("name" => "evalsum", "fac" => 1.0)]
    dftk_to_sirius["Hartree"] = [Dict("name" => "vha", "fac" => 0.5)]
    dftk_to_sirius["Xc"] = [Dict("name" => "exc", "fac" => 1.0)]
    dftk_to_sirius["Ewald"] = [Dict("name" => "ewald", "fac" => 1.0)]
    dftk_to_sirius["Entropy"] = [Dict("name" => "demet", "fac" => 1.0)]

    term_names = Vector{String}()
    energy_values = Vector{Real}()
    for (tname, sinfo) in dftk_to_sirius
        energy = 0.0
        for info in sinfo
            energy += info["fac"]*get_sirius_energy(basis, info["name"])
        end
        push!(term_names, tname)
        push!(energy_values, energy)
    end

    Energies(term_names, energy_values)
end

function get_sirius_energy(basis::SiriusBasis, label)
    return SIRIUS.get_energy(basis.sirius_gs, label)
end

# Compute and retrieve the Cartesian forces from SIRIUS
@DFTK.timing function compute_forces_cart(basis::SiriusBasis, ψ, occupation; kwargs...)
    sirius_forces = Matrix{Cdouble}(undef, 3, length(basis.model.atoms))
    SIRIUS.get_forces!(basis.sirius_gs, "total", sirius_forces)

    forces = [zero(Vec3{Float64}) for _ = 1:length(basis.model.atoms)]
    for iat in 1:length(basis.model.atoms)
        forces[iat] = sirius_forces[:, iat]
    end
    return forces
end

# Compute and retrieve the reduced forces from SIRIUS
function compute_forces(basis::SiriusBasis, ψ, occupation; kwargs...)
    forces_cart = compute_forces_cart(basis, ψ, occupation; kwargs...)
    covector_cart_to_red.(basis.model, forces_cart)
end

# Compute and retrive the stress tensor from SIRIUS
@DFTK.timing function compute_stresses_cart(basis::SiriusBasis, ψ, occupation, eigenvalues, εF)
    sirius_stress = Matrix{Cdouble}(undef, 3, 3)
    SIRIUS.get_stress_tensor!(basis.sirius_gs, "total", sirius_stress)

    stress = Mat3{Float64}(sirius_stress)
    return stress
end

