using SIRIUS

mutable struct SiriusHamiltonian <: AbstractHamiltonian
    basis::SiriusBasis
    ham::SIRIUS.HamiltonianHandler

    function SiriusHamiltonian(basis)
        ham = SIRIUS.create_hamiltonian(basis.sirius_gs)
        H0 = new(basis, ham)
        finalizer(H0) do x
            SIRIUS.free_hamiltonian_handler!(x.ham)
        end
    end
end

function SiriusHamiltonian(basis::SiriusBasis, ρ)
    set_sirius_density(basis, ρ)
    SiriusHamiltonian(basis)
end

function energy_hamiltonian(basis::SiriusBasis, ψ, occupation; ρ, kwargs...)
    #returns the energies and the Hamiltonian calculated by SIRIUS

    ham = SiriusHamiltonian(basis, ρ)
    energies = get_sirius_energies(basis)

    (; energies, ham)
end

function get_sirius_energies(basis::SiriusBasis)

    #TODO: would need to add terms in case of PAW or full potential
    #TODO: need to add terms in case of nspins = 2 (magnetism)
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

function sirius_diagonalize(H0::SiriusHamiltonian, nev_per_kpoint::Int; tol=1.0e-8, maxiter=100, kwargs...)
    converged, niter = SIRIUS.diagonalize_hamiltonian(H0.basis.sirius_gs, H0.ham, tol, maxiter)

    kpoints = H0.basis.kpoints

    #return eigenvalues, eigenvectors, niter, converged
    ispin = 1 #TODO: deal with npsins = 2 case
    λ = Vector{Vector{Float64}}(undef, length(kpoints))
    X = Vector{Matrix{ComplexF64}}(undef, length(kpoints))
    for (ik, kpt) in enumerate(kpoints)
        energies = Vector{Float64}(undef, nev_per_kpoint)
        SIRIUS.get_band_energies!(H0.basis.sirius_kps, ik_global(ik, H0.basis), ispin, energies)
        λ[ik] = energies

        n_Gk = length(G_vectors(H0.basis.pw_basis, kpt)) 
        psi = Matrix{ComplexF64}(undef, n_Gk, nev_per_kpoint)
        SIRIUS.get_psi!(H0.basis.sirius_kps, ik_global(ik, H0.basis), ispin, psi)
        
        for iel = 1:nev_per_kpoint
            psi[:, iel] = psi[H0.basis.d2s_mapping[ik], iel]
        end
        X[ik] = psi
    end
    (; λ=λ, X=X, n_iter=niter, converged=converged)
end