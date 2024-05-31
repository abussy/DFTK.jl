using MPI
using JSON
using SIRIUS

#TODO: next objective is to pass the initial guess density calculated in DFTK to SIRIUS, as
#      the starting point of the SCF => check that it works, that the order is correct, etc.
#      by doing a single SCF step and comparing the energies

mutable struct SiriusBasis{T} <: AbstractBasis{T}

    # Underlying DFTK PW basis which corresponds exactly to the SIRIUS one
    PWBasis::PlaneWaveBasis{T}

    # Sirius handlers
    SiriusCtx::SIRIUS.ContextHandler
    SiriusKps::SIRIUS.KpointSetHandler
    SiriusGs::SIRIUS.GroundStateHandler     

    # Sirius parameters
    SiriusParams::Dict{Any}

    # Mapping of of the G+k vector indices between DFTK and SIRIUS, in both directions, for each KP
    D2S_mapping::Vector{Vector{Int}}
    S2D_mapping::Vector{Vector{Int}}

    function SiriusBasis(PWBasis, SiriusCtx, SiriusKps, SiriusGs, SiriusParams, D2S_mapping, S2D_mapping)
        x = new{typeof(PWBasis.Ecut)}(PWBasis, SiriusCtx, SiriusKps, SiriusGs, SiriusParams,
                                      D2S_mapping, S2D_mapping)
        finalizer(FinalizeBasis, x)
    end

end

function FinalizeBasis(basis::SiriusBasis)
    SIRIUS.free_ground_state_handler(basis.SiriusGs)
    SIRIUS.free_kpoint_set_handler(basis.SiriusKps)
    SIRIUS.free_context_handler(basis.SiriusCtx)
end

function FinalizeSirius()
    if SIRIUS.is_initialized()
        SIRIUS.finalize(false)
    end
end

# Allow direct access to SiriusBasis.PWBasis attribute
function Base.getproperty(basis::SiriusBasis, symbol::Symbol)
    if symbol in fieldnames(PlaneWaveBasis)
        return getfield(basis.PWBasis, symbol)
    else
        return getfield(basis, symbol)
    end
end

function SiriusBasis(model::Model{T};
                     Ecut::Number,
                     kgrid=nothing,
                     kshift=[0, 0, 0],
                     variational=true, fft_size=nothing,
                     symmetries_respect_rgrid=isnothing(fft_size),
                     use_symmetries_for_kpoint_reduction=true,
                     comm_kpts=MPI.COMM_WORLD, architecture=CPU()) where {T <: Real} 
   
    # Create the PW basis on the DFTK side
    PWBasis = PlaneWaveBasis(model; Ecut, kgrid, kshift, variational, fft_size, 
                             symmetries_respect_rgrid, use_symmetries_for_kpoint_reduction,
                             comm_kpts, architecture, instantiate_terms=false)

    # TODO: tmp: we set the SIRIUS library path to local build, so that no need to rebuild JLL
    #       eventually, need to set that in Project.toml or some equivalent file
    SIRIUS.libpath = ENV["LD_LIBRARY_PATH"]*"/libsirius.so" 

    #  Initialize the SIRIUS library
    if !SIRIUS.is_initialized()
        SIRIUS.initialize(false)
    end

    # Parse Model and Basis parameters into SIRIUS parameters JSON format 
    SiriusParams = CreateSiriusParams(model, Ecut)

    ParamsJson = JSON.json(SiriusParams)
    SiriusCtx = SIRIUS.create_context_from_json(PWBasis.comm_kpts, ParamsJson)
    SIRIUS.initialize_context(SiriusCtx)

    #TODO: insure that the k-point distribution is compatibale between SIRIUS and DFTK
    SiriusKps = SIRIUS.create_kset(SiriusCtx; num_kp=length(PWBasis.kweights_global), 
                                   k_coords=PWBasis.kcoords_global, 
                                   k_weights=PWBasis.kweights_global)

    SiriusGs = SIRIUS.create_ground_state(SiriusKps) 

    D2S_mapping, S2D_mapping = GetDftkSiriusMapping(PWBasis, SiriusKps)

    SB = SiriusBasis(PWBasis, SiriusCtx, SiriusKps, SiriusGs, SiriusParams, D2S_mapping, S2D_mapping)

    # Make sure finlalizer is called before MPI.Finalize()
    MPI.add_finalize_hook!(() -> FinalizeBasis(SB))

    # Only finalize SIRIUS library at program exit
    atexit(FinalizeSirius)

    return SB
end

function UpdateSiriusParams(SiriusParams::Dict{Any}, section::String, keyword::String, value::Any)

    # Note: we assume only one level of nesting in Sirius input
    if !haskey(SiriusParams, section) SiriusParams[section] = Dict() end
    if haskey(SiriusParams[section], keyword) 
        old_val = SiriusParams[section][keyword]
        @warn("Overwriting SIRIUS parameter $section/$keyword. "*
              "Old value: $old_val, New value: $value")
    end
    SiriusParams[section][keyword] = value     
end

function UpdateSiriusParams(Basis::SiriusBasis, section::String, keyword::String, value::Any)
    UpdateSiriusParams(Basis.SiriusParams, section, keyword, value)
end

function CreateSiriusParams(model::Model{T}, Ecut::Real) where {T <: Real}

    SiriusParams = Dict()
    #TODO: take that value from architecture input
    UpdateSiriusParams(SiriusParams, "control", "processing_unit", "cpu")
    UpdateSiriusParams(SiriusParams, "control", "verbosity", 1) #TODO: probably want zero default

    #TODO: at some point, might want to allow full-potential as well
    UpdateSiriusParams(SiriusParams, "parameters", "electronic_structure_method", "pseudopotential")

    # Go over the terms of the model, and check that it is compatible with SIRIUS + extract XC info
    # TODO. For now, we simlpy assume it is compatible, and we do a PBE model
    UpdateSiriusParams(SiriusParams, "parameters", "xc_functionals", ["XC_GGA_X_PBE", "XC_GGA_C_PBE"])
    #UpdateSiriusParams(SiriusParams, "iterative_solver", "type", "exact") #TODO: this is tmp

    #Smearing. Note: not 100% match with SIRIUS options
    smearing_type = typeof(model.smearing)
    if smearing_type == Smearing.None
        #Actually not implemented, we just use the tiniest smearing width
        UpdateSiriusParams(SiriusParams, "parameters", "smearing_width", 1.0e-16)
    elseif smearing_type == Smearing.FermiDirac
        UpdateSiriusParams(SiriusParams, "parameters", "smearing", "fermi_dirac")
        UpdateSiriusParams(SiriusParams, "parameters", "smearing_width", model.temperature) 
    elseif smearing_type == Smearing.Gaussian
        UpdateSiriusParams(SiriusParams, "parameters", "smearing", "gaussian")
        UpdateSiriusParams(SiriusParams, "parameters", "smearing_width", model.temperature) 
    elseif smearing_type == Smearing.MarzariVanderbilt
        UpdateSiriusParams(SiriusParams, "parameters", "smearing", "cold")
        UpdateSiriusParams(SiriusParams, "parameters", "smearing_width", model.temperature) 
    else
        @error("Smearing type $smearing_type not implemented in SIRIUS")
    end    

    #Cutoffs work as follow: cutoff_dftk = 0.5*cutoff_qe = 0.5*cutoff_sirius^2
    #TODO: we need to make sure that the grids are 100% compatible. It appears to not always
    #      be the case. Why?
    sirius_cutoff = sqrt(2*Ecut)
    UpdateSiriusParams(SiriusParams, "parameters", "gk_cutoff", sirius_cutoff)
    #TODO: for now, we use the default x2 factor of NC PPs. This will need adaptation
    UpdateSiriusParams(SiriusParams, "parameters", "pw_cutoff", 2*sirius_cutoff)

    UpdateSiriusParams(SiriusParams, "unit_cell", "lattice_vectors", model.lattice)
    UpdateSiriusParams(SiriusParams, "unit_cell", "atom_files", 
                       Dict(String(el.symbol) => el.fname for el in model.atoms))

    atom_types = unique([String(el.symbol) for el in model.atoms])
    UpdateSiriusParams(SiriusParams, "unit_cell", "atom_types", atom_types)

    atoms = Dict(atom_type => [] for atom_type in atom_types)
    for (iel, el) in enumerate(model.atoms)
        append!(atoms[String(el.symbol)], [model.positions[iel]])
    end
    UpdateSiriusParams(SiriusParams, "unit_cell", "atoms", atoms)

    return SiriusParams
end

function SetSiriusDensity(basis::SiriusBasis{T}, ρ::Array{T, 4}) where {T <: Real}

    #TODO: it seems that z_offset = -1 works.  What would happen
    #      in the case where nprocs >> nkpoints? Not allowed by DFTK, but maybe in the future,
    #      this could be useful to over-parallelize sirius
    z_offset = -1 #SIRIUS.get_fft_local_z_offset(basis.SiriusCtx)
    #z_offset = 0 #SIRIUS.get_fft_local_z_offset(basis.SiriusCtx)

    SIRIUS.set_rg_density(basis.SiriusGs, ρ, size(ρ)[1], size(ρ)[2], size(ρ)[3], z_offset)
    FFTSiriusDensity(basis)
end

function GetSiriusDensity(basis::SiriusBasis{T}) where {T <: Real}
    z_offset = -1
    SIRIUS.get_rg_density(basis.SiriusGs, basis.fft_size[1], basis.fft_size[2],
                          basis.fft_size[3], z_offset)
end

function FFTSiriusDensity(basis::SiriusBasis)
    SIRIUS.fft_transform(basis.SiriusGs, "rho", -1)
end

mutable struct SiriusHamiltonian 
    basis::SiriusBasis
    ham::SIRIUS.HamiltonianHandler

    function SiriusHamiltonian(basis)
        ham = SIRIUS.create_hamiltonian(basis.SiriusGs)
        x = new(basis, ham)
        finalizer(FreeSiriusHamiltonian, x)
    end
end

function FreeSiriusHamiltonian(H0::SiriusHamiltonian)
    SIRIUS.free_hamiltonian_handler(H0.ham)
end

function SiriusHamiltonian(basis::SiriusBasis, ρ::Array{<:Real, 4})
    SetSiriusDensity(basis, ρ)
    SiriusHamiltonian(basis)
end

function SiriusEnergies(basis::SiriusBasis)

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
            energy += info["fac"]*GetSiriusEnergy(basis, info["name"])
        end
        push!(term_names, tname)
        push!(energy_values, energy)
    end

    Energies(term_names, energy_values)
end

function energy_hamiltonian(basis::SiriusBasis, ψ, occupation; ρ, kwargs...)
    #returns the energies and the Hamiltonian calculated by SIRIUS

    ham = SiriusHamiltonian(basis, ρ)
    energies = SiriusEnergies(basis)

    (; energies, ham)
end

function SiriusDiagonalize(H0::SiriusHamiltonian, nev_per_kpoint::Int; tol=1.0e-6, maxiter=100, kwargs...)
    #TODO: currently uses SIRIUS' internal guess for Ψ, should we enable passing the guess as well?
    converged, niter = SIRIUS.diagonalize_hamiltonian(H0.basis.SiriusGs, H0.ham, tol, maxiter)

    kpoints = H0.basis.kpoints

    #return eigenvalues, eigenvectors, niter, converged
    ispin = 1 #TODO: deal with npsins = 2 case
    λ = []
    X = []
    for (ik, kpt) in enumerate(kpoints)
        push!(λ, SIRIUS.get_band_energies(H0.basis.SiriusKps, ik, ispin, nev_per_kpoint))
        n_Gk = length(G_vectors(H0.basis.PWBasis, kpt)) #TODO: add warnings as in diag.jl?
        #TODO: could we have a problem because internally in SIRIUS, there are more bands? (contiguity)
        Spsi = SIRIUS.get_psi(H0.basis.SiriusKps, ik, ispin, n_Gk, nev_per_kpoint)
        Dpsi = Matrix{ComplexF64}(undef, n_Gk, nev_per_kpoint)
        for iel = 1:nev_per_kpoint
            Dpsi[:, iel] = RemapArray(Spsi[:, iel], H0.basis.D2S_mapping[ik])[:]
        end
        push!(X, Dpsi)
    end
    (; λ=λ, X=X, n_iter=niter, converged=converged)
end

function SiriusSetOccupation(basis::SiriusBasis, occupation::AbstractVector)
    nkp = length(basis.kpoints)    
    ispin = 1 #TODO nspins = 2 case
    for ikp = 1:nkp
        SIRIUS.set_band_occupancies(basis.SiriusKps, ikp, ispin, occupation[ikp])
    end
end

function SiriusComputeDensity(basis::SiriusBasis)
    SIRIUS.generate_density(basis.SiriusGs)
    GetSiriusDensity(basis)
end

function compute_occupation(H0::SiriusHamiltonian, num_bands::Integer)
    SIRIUS.find_band_occupancies(H0.basis.SiriusKps)
    nkp = length(H0.basis.kpoints)    
    ispin = 1 #TODO nspins = 2 case
    occupation = Vector{Vector{Float64}}()
    for ikp = 1:nkp
        push!(occupation, Vector{Float64}(SIRIUS.get_band_occupancies(H0.basis.SiriusKps, ikp, ispin, num_bands)))
    end
    εF = SIRIUS.get_energy(H0.basis.SiriusGs, "fermi")
    (; occupation, εF)
end

function next_density(ham::SiriusHamiltonian,
                      nbandsalg::NbandsAlgorithm=AdaptiveBands(ham.basis.model),
                      fermialg::AbstractFermiAlgorithm=default_fermialg(ham.basis.model);
                      ψ=nothing, eigenvalues=nothing, occupation=nothing,
                      kwargs...)
    # Digonalizes the SIRIUS Hamiltonian, returns all required data
    # closely follows the original next_density definition
    n_bands_converge, n_bands_compute = determine_n_bands(nbandsalg, occupation,
                                                          eigenvalues, ψ)

    if isnothing(ψ)
        increased_n_bands = true
    else
        @assert length(ψ) == length(ham.basis.kpoints)
        n_bands_compute = max(n_bands_compute, maximum(ψk -> size(ψk, 2), ψ))
        increased_n_bands = n_bands_compute > size(ψ[1], 2)
    end 

    # Inherited from original next_density()
    n_bands_compute = mpi_max(n_bands_compute, ham.basis.comm_kpts)
    #TODO: probably need a safety measure, so that we never go too high
    #      we also need to be 100% sure that changing n_bands in SIRIUS at runtime is ok, if not
    #      we will need to either settle for an upper bound, or fix it in SIRIUS
    SIRIUS.set_num_bands(ham.basis.SiriusCtx, n_bands_compute) 

    #tol is passed by kwargs as determine_diagtol, which is way too big and variable
    #TODO: figure out why and fix it. 
    eigres = SiriusDiagonalize(ham, n_bands_compute)#; kwargs...) 
    eigres.converged || (@warn "Eigensolver not converged" n_iter=eigres.n_iter)

    # Check maximal occupation of the unconverged bands is sensible.
    occupation, εF = compute_occupation(ham, n_bands_compute)
    minocc = maximum(minimum, occupation)

    # Inherited from original next_density()
    if !increased_n_bands && minocc > nbandsalg.occupation_threshold
        @warn("Detected large minimal occupation $minocc. SCF could be unstable. " *
              "Try switching to adaptive band selection (`nbandsalg=AdaptiveBands(model)`) " *
              "or request more converged bands than $n_bands_converge (e.g. " *
              "`nbandsalg=AdaptiveBands(model; n_bands_converge=$(n_bands_converge + 3)`)")
    end

    #TODO: both approach work now that the wave functions are exchanged. Need to choose one.
    #      When we move away from NC, then we might not have a choice anymore
    #ρout = SiriusComputeDensity(ham.basis)
    ρout = compute_density(ham.basis.PWBasis, eigres.X, occupation; nbandsalg.occupation_threshold)

    (; ψ=eigres.X, eigenvalues=eigres.λ, occupation, εF, ρout, diagonalization=eigres,
     n_bands_converge, nbandsalg.occupation_threshold)

end

function guess_density(basis::SiriusBasis)
    SIRIUS.generate_initial_density(basis.SiriusGs)
    GetSiriusDensity(basis)
end

function GetDftkSiriusMapping(PWBasis::PlaneWaveBasis, SiriusKps::SIRIUS.KpointSetHandler)
    #Loop over k-points, then loop over integer coordinates of G+k vectors of DFTK and SIRIUS
    #to match them to each other. Necessary to exchange wave functions

    #TODO: might want to rename these, or at least document them. It's kinda confusing now 
    #      (see remapping in SiriusDiagonalize)
    D2S_mapping = []
    S2D_mapping = []

    kpoints = PWBasis.kpoints

    for (ik, kpt) in enumerate(kpoints)
        n_Gk = length(G_vectors(PWBasis, kpt))
        Sgkvec = SIRIUS.get_gkvec(SiriusKps, ik, n_Gk)
        Dgkvec = kpt.G_vectors
        D2S, S2D = GetMapping(Sgkvec, Dgkvec, kpt.coordinate)
        push!(D2S_mapping, D2S)
        push!(S2D_mapping, S2D)
    end

    return D2S_mapping, S2D_mapping
end

#TODO: is it where we should have explicit types for Julia performance?
function GetMapping(Sgkvec, Dgkvec, kp_coord)
    D2S = Vector{Int}(undef, length(Sgkvec))
    S2D = Vector{Int}(undef, length(Dgkvec))

    for (is, sg) in enumerate(Sgkvec)
        for (id, dg) in enumerate(Dgkvec)
            #In DFTK, test is on (G+k)**2 <= Ecut, but only G is stored
            #In SIRIUS (G+k) is stored, need to remove it for check
            sg_int = [Int(sg[i]-kp_coord[i]) for i in 1:3]
            if sg_int == dg
                D2S[id] = is
                S2D[is] = id
                break
            end
            if id == length(Dgkvec)
                @error("Missmatch in G+k vectors between DFTK and SIRIUS")
            end
        end
    end

    return D2S, S2D
end

#TODO: this would also probably benefits from having explicit types, and not using push!
#TODO: there might also be a Julia intrinsics for this, the same as in Python
function RemapArray(array, mapping)
    new = []
    for i = 1:length(array)
        push!(new, array[mapping[i]])
    end
    return new
end


default_diagtolalg(basis::SiriusBasis; tol, kwargs...) = AdaptiveDiagtol()

mix_density(mixing::Mixing, basis::SiriusBasis, Δρ; kwargs...) = 
    mix_density(mixing, basis.PWBasis, Δρ; kwargs...)

#TEST
function self_consistent_field_test(basis::SiriusBasis; ρ=nothing, maxiter=100, tol=1.0e-6)
    occupation = nothing
    eigenvalues = nothing
    converged = nothing
    niter = nothing
    εF = nothing
    energies = nothing
    ham = nothing

    ρout = ρ

    for i = 1:maxiter
        energies, ham = energy_hamiltonian(basis, nothing, occupation; ρ=ρout, eigenvalues, εF)

        #Note: having variable amounts of states to diagonlize seems to lead to instabilities
        eigres = SiriusDiagonalize(ham, 9; tol=tol)#rand(6:10))

        ρout = SiriusComputeDensity(ham.basis)
        @show energies
        if i == maxiter @show eigres.λ end
    end
    return (; energies=energies)
end