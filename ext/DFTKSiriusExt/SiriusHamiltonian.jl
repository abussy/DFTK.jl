using LinearAlgebra

### The Sirius HamiltonianBlock type
mutable struct SiriusHamiltonianBlock <: HamiltonianBlock
    basis::SiriusBasis
    ik::Integer
    kpoint::Kpoint
    H0::SIRIUS.HamiltonianHandler

    # TODO: add some buffers for storage
end

### The SIRIUS Hamiltonian type
mutable struct SiriusHamiltonian <: AbstractHamiltonian
    basis::SiriusBasis
    H0::SIRIUS.HamiltonianHandler
    blocks::Vector{SiriusHamiltonianBlock}

    function SiriusHamiltonian(basis)
        H0 = SIRIUS.create_hamiltonian(basis.sirius_gs)
        blocks = Vector{SiriusHamiltonianBlock}(undef, length(basis.kpoints))
        for (ik, kpoint) in enumerate(basis.kpoints) 
            blocks[ik] = SiriusHamiltonianBlock(basis, ik, kpoint, H0)
        end
        ham = new(basis, H0, blocks)
        finalizer(ham) do x
            SIRIUS.free_hamiltonian_handler!(x.H0)
        end
    end
end

Base.getindex(ham::SiriusHamiltonian, index) = ham.blocks[index]

# Construction for the SiriusHamiltonian, built based on an input density
@DFTK.timing function SiriusHamiltonian(basis::SiriusBasis, ρ)
    set_sirius_density(basis, ρ)
    SiriusHamiltonian(basis)
end

# Returns an Hamiltonian and corresponding energies, as necessary for the SCF
@DFTK.timing function energy_hamiltonian(basis::SiriusBasis, ψ, occupation; ρ, kwargs...)
    #returns the energies and the Hamiltonian calculated by SIRIUS

    ham = SiriusHamiltonian(basis, ρ)
    energies = get_sirius_energies(basis)

    (; energies, ham)
end

@DFTK.timing function Hamiltonian(basis::SiriusBasis; ψ=nothing, occupation=nothing, kwargs...)
    energy_hamiltonian(basis, ψ, occupation; kwargs...).ham
end

#TODO: maybe not necessary to allocate the k-point blocks for just the energy. Is it expensive?
@DFTK.timing function energy(basis::SiriusBasis, ψ, occupation; kwargs...)
    energies = energy_hamiltonian(basis, ψ, occupation; kwargs...).energies
    (; energies=energies)
end

# Diagonalizes the Hamiltonian in SIRIUS and returns the eigenvalues and eigenvectors
@DFTK.timing function sirius_diagonalize(eigensolver, ham::SiriusHamiltonian, nev_per_kpoint::Int; tol=1.0e-8, maxiter=100, kwargs...)

    if nev_per_kpoint > ham.basis.max_num_bands
        error("Not enough bands available in SIRIUS. Increase 'max_num_bands_factor' when creating the SIRIUS basis.")
    end
    SIRIUS.set_num_bands(ham.basis.sirius_ctx, nev_per_kpoint)

    exact_diag = false
    if nameof(eigensolver) == :diag_full
        exact_diag = true
    end

    #Note: converge_by_energy=0 means we test convergence with the residual L2 norm, like in DFTK
    converged, niter = SIRIUS.diagonalize_hamiltonian(ham.basis.sirius_ctx, ham.basis.sirius_gs, ham.H0,
                                                      ham.basis.iter_tol_factor*tol, maxiter;
                                                      converge_by_energy=0, 
                                                      exact_diagonalization=exact_diag)

    kpoints = ham.basis.kpoints

    # return eigenvalues, eigenvectors, niter, converged
    λ = Vector{Vector{Float64}}(undef, length(kpoints))
    X = Vector{Matrix{ComplexF64}}(undef, length(kpoints))
    res = Vector{Vector{Any}}(undef, length(kpoints))
    for (ik, kpt) in enumerate(kpoints)
        ik_global, ispin = ik_global_and_spin(ik, ham.basis)
        energies = Vector{Float64}(undef, nev_per_kpoint)
        SIRIUS.get_band_energies!(ham.basis.sirius_kps, ik_global, ispin, energies)
        λ[ik] = energies

        n_Gk = length(G_vectors(ham.basis.pw_basis, kpt)) 
        psi = Matrix{ComplexF64}(undef, n_Gk, nev_per_kpoint)
        SIRIUS.get_psi!(ham.basis.sirius_kps, ik_global, ispin, psi)
        
        for iel = 1:nev_per_kpoint
            psi[:, iel] = psi[ham.basis.d2s_mapping[ik], iel]
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

# Offload H x ψ product to SIRIUS
function LinearAlgebra.mul!(Hψ, H::SiriusHamiltonian, ψ)
    for ik = 1:length(H.basis.kpoints)
        mul!(Hψ[ik], H[ik], ψ[ik])
    end
    Hψ
end

@views @DFTK.timing "SiriusHamiltonian multiplication" function LinearAlgebra.mul!(
    Hψ::AbstractArray, H::SiriusHamiltonianBlock, ψ::AbstractArray)
    
    n_gkvecs = size(ψ, 1)
    n_bands = size(ψ, 2)
    if n_bands > H.basis.max_num_bands
        error("Not enough bands available in SIRIUS. Increase 'max_num_bands_factor' when creating the SIRIUS basis.")
    end
    SIRIUS.set_num_bands(H.basis.sirius_ctx, n_bands)

    #TODO:
    #For now, allocate buffers that are contiguous. In the future, might want to preallocate
    #them in the HamiltionanBlock, as kinda done already in DFTK. But maybe not, because then
    #that's a lot of memory allocated in buffers... Need to measure
    #It seems that the buffer might be unnecessary, and that the views are interpreted as 
    #contiguous, but this needs to be tests thouroughly, also when the n_bands are not next
    #to each others (they seem to always be next to each other though, but that should be tested
    #before the apply_h call, e.g. test that slice is without hole)

    #ψ_buff = zeros(ComplexF64, n_gkvecs, n_bands)
    #Hψ_buff = zeros(ComplexF64, n_gkvecs, n_bands)

    # reordering to SIRIUS requirements
    # TODO: with GPU, probably will need a buffer to use map!
    for ib = 1:n_bands
        #ψ_buff[:, ib] = ψ[H.basis.s2d_mapping[H.ik], ib]
        ψ[:, ib] = ψ[H.basis.s2d_mapping[H.ik], ib]
    end

    #TODO: should have a ! since changes Hψ
    ik_global, ispin = ik_global_and_spin(H.ik, H.basis)
    #SIRIUS.apply_h(H.basis.sirius_kps, H.H0, ik_global, n_bands, ψ_buff, Hψ_buff)
    # TODO: with GPU, might want to have separate set_psi and get_hpsi functions,
    #       so that a single buffer can be used on DFTK side. Even better if we
    #       can do that band by band
    SIRIUS.apply_h(H.basis.sirius_kps, H.H0, ik_global, n_bands, ψ, Hψ)

    # reordering back
    for ib = 1:n_bands
        #Hψ[:, ib] = Hψ_buff[H.basis.d2s_mapping[H.ik], ib]
        Hψ[:, ib] = Hψ[H.basis.d2s_mapping[H.ik], ib]
        ψ[:, ib] = ψ[H.basis.d2s_mapping[H.ik], ib]
    end
    Hψ
end
