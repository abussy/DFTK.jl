### The SIRIUS Hamiltonian type
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

# Construction for the SiriusHamiltonian, built based on an input density
function SiriusHamiltonian(basis::SiriusBasis, ρ)
    set_sirius_density(basis, ρ)
    SiriusHamiltonian(basis)
end

# Returns an Hamiltonian and corresponding energies, as necessary for the SCF
function energy_hamiltonian(basis::SiriusBasis, ψ, occupation; ρ, kwargs...)
    #returns the energies and the Hamiltonian calculated by SIRIUS

    ham = SiriusHamiltonian(basis, ρ)
    energies = get_sirius_energies(basis)

    (; energies, ham)
end

function Hamiltonian(basis::SiriusBasis; ψ=nothing, occupation=nothing, kwargs...)
    energy_hamiltonian(basis, ψ, occupation; kwargs...).ham
end

function energy(basis::SiriusBasis, ψ, occupation; kwargs...)
    energies = energy_hamiltonian(basis, ψ, occupation; kwargs...).energies
    (; energies=energies)
end

# Diagonalizes the Hamiltonian in SIRIUS and returns the eigenvalues and eigenvectors
function sirius_diagonalize(eigensolver, H0::SiriusHamiltonian, nev_per_kpoint::Int; tol=1.0e-8, maxiter=100, kwargs...)

    if nev_per_kpoint > H0.basis.max_num_bands
        error("Not enough bands available in SIRIUS. Increase 'max_num_bands_factor' when creating the SIRIUS basis.")
    end
    SIRIUS.set_num_bands(H0.basis.sirius_ctx, nev_per_kpoint)

    exact_diag = false
    if nameof(eigensolver) == :diag_full
        exact_diag = true
    end

    #Note: converge_by_energy=0 means we test convergence with the residual L2 norm, like in DFTK
    converged, niter = SIRIUS.diagonalize_hamiltonian(H0.basis.sirius_ctx, H0.basis.sirius_gs, H0.ham,
                                                      H0.basis.iter_tol_factor*tol, maxiter;
                                                      converge_by_energy=0, 
                                                      exact_diagonalization=exact_diag)

    kpoints = H0.basis.kpoints

    # return eigenvalues, eigenvectors, niter, converged
    λ = Vector{Vector{Float64}}(undef, length(kpoints))
    X = Vector{Matrix{ComplexF64}}(undef, length(kpoints))
    res = Vector{Vector{Any}}(undef, length(kpoints))
    for (ik, kpt) in enumerate(kpoints)
        ik_global, ispin = ik_global_and_spin(ik, H0.basis)
        energies = Vector{Float64}(undef, nev_per_kpoint)
        SIRIUS.get_band_energies!(H0.basis.sirius_kps, ik_global, ispin, energies)
        λ[ik] = energies

        n_Gk = length(G_vectors(H0.basis.pw_basis, kpt)) 
        psi = Matrix{ComplexF64}(undef, n_Gk, nev_per_kpoint)
        SIRIUS.get_psi!(H0.basis.sirius_kps, ik_global, ispin, psi)
        
        for iel = 1:nev_per_kpoint
            psi[:, iel] = psi[H0.basis.d2s_mapping[ik], iel]
        end
        X[ik] = psi

        # residuals are not available via SIRIUS, we fill the vector with nothings
        res[ik] = Vector{Any}(nothing, nev_per_kpoint)

    end
    # Number of matvec not available via SIRIUS, we set it to -1 to make it clear
    (; λ=λ, X=X, residual_norms=res, n_iter=niter, converged=converged, n_matvec=-1)
end

function diagonalize_all_kblocks(eigensolver, ham::SiriusHamiltonian, nev_per_kpoint::Int; kwargs ...)
    sirius_diagonalize(eigensolver, ham, nev_per_kpoint; kwargs...) 
end
