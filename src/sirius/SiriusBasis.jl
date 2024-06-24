using MPI
using JSON
using SIRIUS

#Important node: to get full equivalence (i.e. same total energies) between DFTK and SIRIUS/QE, there
#                are 2 things to do: use the simpson method on all radial grid integrations in DFTK.
#                Disable 10 a.u. arbitrary cutoff of the radial grid in SIRIUS, radial_integrals.cpp 

#TODO: we only allow for double precision with  SIRIUS, should we relect that in T down there?
mutable struct SiriusBasis{T} <: AbstractBasis{T}

    # Underlying DFTK PW basis which corresponds exactly to the SIRIUS one
    pw_basis::PlaneWaveBasis{T}

    # Sirius handlers
    sirius_ctx::SIRIUS.ContextHandler
    sirius_kps::SIRIUS.KpointSetHandler
    sirius_gs::SIRIUS.GroundStateHandler     

    # Mapping of of the G+k vector indices between DFTK and SIRIUS, in both directions, for each KP
    d2s_mapping::Vector{Vector{Int}}
    s2d_mapping::Vector{Vector{Int}}

    function SiriusBasis(pw_basis, sirius_ctx, sirius_kps, sirius_gs, d2s_mapping, s2d_mapping)
        x = new{typeof(pw_basis.Ecut)}(pw_basis, sirius_ctx, sirius_kps, sirius_gs,
                                      d2s_mapping, s2d_mapping)
        finalizer(FinalizeBasis, x)
    end

end

function FinalizeBasis(basis::SiriusBasis)
    SIRIUS.free_ground_state_handler!(basis.sirius_gs)
    SIRIUS.free_kpoint_set_handler!(basis.sirius_kps)
    SIRIUS.free_context_handler!(basis.sirius_ctx)
end

function FinalizeSirius()
    if SIRIUS.is_initialized()
        SIRIUS.finalize(; call_mpi_fin=false)
    end
end

# Allow direct access to SiriusBasis.pw_basis attribute
function Base.getproperty(basis::SiriusBasis, symbol::Symbol)
    if symbol in fieldnames(PlaneWaveBasis)
        return getfield(basis.pw_basis, symbol)
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
    pw_basis = PlaneWaveBasis(model; Ecut, kgrid, kshift, variational, fft_size, 
                              symmetries_respect_rgrid, use_symmetries_for_kpoint_reduction,
                              comm_kpts, architecture, instantiate_terms=false)

    #  Initialize the SIRIUS library
    if !SIRIUS.is_initialized()
        SIRIUS.initialize(false)
    end

    # Parse Model and Basis parameters into SIRIUS parameters JSON format 
    sirius_params = create_sirius_params(model, Ecut, pw_basis.fft_size)

    #TODO: use JSON3 like the rest of DFTK
    json_params = JSON.json(sirius_params)
    sirius_ctx = SIRIUS.create_context_from_json(pw_basis.comm_kpts, json_params)
    SIRIUS.initialize_context(sirius_ctx)

    sirius_kps = SIRIUS.create_kset(sirius_ctx; num_kp=length(pw_basis.kweights_global), 
                                    k_coords=pw_basis.kcoords_global, 
                                    k_weights=pw_basis.kweights_global,
                                    init_kset=false)

    # Insure that the k-point distribution is compatibale between SIRIUS and DFTK
    count = Vector{Int32}(undef, mpi_nprocs(comm_kpts))
    for ip = 1:mpi_nprocs(comm_kpts)
        count[ip] = length(pw_basis.krange_allprocs[ip][1]) #note: second index is spin
    end 
    SIRIUS.initialize_kset(sirius_kps; count=count)

    sirius_gs = SIRIUS.create_ground_state(sirius_kps) 

    d2s_mapping, s2d_mapping = get_gkvec_mapping(pw_basis, sirius_kps)

    SB = SiriusBasis(pw_basis, sirius_ctx, sirius_kps, sirius_gs, d2s_mapping, s2d_mapping)

    # Make sure finlalizer is called before MPI.Finalize()
    MPI.add_finalize_hook!(() -> FinalizeBasis(SB))

    # Only finalize SIRIUS library at program exit
    atexit(FinalizeSirius)

    return SB
end

#TODO: there might be Julia native constrcts for this
function set_sirius_param(sirius_params::Dict{Any}, section::String, keyword::String, value::Any)

    # Note: we assume only one level of nesting in Sirius input
    if !haskey(sirius_params, section) sirius_params[section] = Dict() end
    if haskey(sirius_params[section], keyword) 
        old_val = sirius_params[section][keyword]
        @warn("Overwriting SIRIUS parameter $section/$keyword. "*
              "Old value: $old_val, New value: $value")
    end
    sirius_params[section][keyword] = value     
end

function set_sirius_param(basis::SiriusBasis, section::String, keyword::String, value::Any)
    set_sirius_param(basis.sirius_params, section, keyword, value)
end

function create_sirius_params(model, Ecut, fft_size )

    sirius_params = Dict()
    #TODO: take that value from architecture input
    set_sirius_param(sirius_params, "control", "processing_unit", "cpu")
    set_sirius_param(sirius_params, "control", "verbosity", 0) #TODO: probably want zero default

    #TODO: at some point, might want to allow full-potential as well
    set_sirius_param(sirius_params, "parameters", "electronic_structure_method", "pseudopotential")

    # Go over the terms of the model, and check that it is compatible with SIRIUS + extract XC info
    # TODO. For now, we simlpy assume it is compatible, and we do a PBE model
    set_sirius_param(sirius_params, "parameters", "xc_functionals", ["XC_GGA_X_PBE", "XC_GGA_C_PBE"])
    #TODO: this is tmp, need to figure out adaptive tolerance
    #set_sirius_param(sirius_params, "iterative_solver", "type", "exact") 

    #Impose DFTK FFT grid dimensions to SIRIUS for 100% compatibility
    set_sirius_param(sirius_params, "settings", "fft_grid_size", fft_size)

    #Smearing. Note: not 100% match with SIRIUS options
    smearing_type = typeof(model.smearing)
    if smearing_type == Smearing.None
        #Actually not implemented, we just use the tiniest smearing width
        set_sirius_param(sirius_params, "parameters", "smearing_width", 1.0e-16)
    elseif smearing_type == Smearing.FermiDirac
        set_sirius_param(sirius_params, "parameters", "smearing", "fermi_dirac")
        set_sirius_param(sirius_params, "parameters", "smearing_width", model.temperature) 
    elseif smearing_type == Smearing.Gaussian
        set_sirius_param(sirius_params, "parameters", "smearing", "gaussian")
        set_sirius_param(sirius_params, "parameters", "smearing_width", model.temperature) 
    elseif smearing_type == Smearing.MarzariVanderbilt
        set_sirius_param(sirius_params, "parameters", "smearing", "cold")
        set_sirius_param(sirius_params, "parameters", "smearing_width", model.temperature) 
    else
        error("Smearing type $smearing_type not implemented in SIRIUS")
    end    

    #Cutoffs work as follow: cutoff_dftk = 0.5*cutoff_qe = 0.5*cutoff_sirius^2
    sirius_cutoff = sqrt(2*Ecut)
    set_sirius_param(sirius_params, "parameters", "gk_cutoff", sirius_cutoff)
    #TODO: for now, we use the default x2 factor of NC PPs. This will need adaptation
    set_sirius_param(sirius_params, "parameters", "pw_cutoff", 2*sirius_cutoff)

    set_sirius_param(sirius_params, "unit_cell", "lattice_vectors", model.lattice)
    set_sirius_param(sirius_params, "unit_cell", "atom_files", 
                       Dict(String(el.symbol) => el.fname for el in model.atoms))

    atom_types = unique([String(el.symbol) for el in model.atoms])
    set_sirius_param(sirius_params, "unit_cell", "atom_types", atom_types)

    atoms = Dict(atom_type => [] for atom_type in atom_types)
    for (iel, el) in enumerate(model.atoms)
        append!(atoms[String(el.symbol)], [model.positions[iel]])
    end
    set_sirius_param(sirius_params, "unit_cell", "atoms", atoms)

    return sirius_params
end

function set_sirius_density(basis::SiriusBasis{T}, ρ::Array{T, 4}) where {T <: Real}

    SIRIUS.set_periodic_function(basis.sirius_gs, "rho"; f_rg=ρ, size_x=size(ρ)[1], size_y=size(ρ)[2],
                                 size_z=size(ρ)[3], offset_z=-1)
    #make PW density available in SIRIUS
    SIRIUS.fft_transform(basis.sirius_gs, "rho", -1)
end

function get_sirius_density(basis::SiriusBasis{T}) where {T <: Real}
    #TODO: this should not do a hidden allocation,we should provide the array
    get_rg_density(basis.sirius_gs, basis.fft_size[1], basis.fft_size[2], basis.fft_size[3])
end

#TODO: this is temporary and should absolutely be replaced
function get_rg_density(sirius_gs, size_x, size_y, size_z)
    #TODO: this is the thing to avoid: an allocation at each data exchange
    f_rg = Array{Cdouble, 4}(undef, size_x, size_y, size_z, 1)
    SIRIUS.get_periodic_function!(sirius_gs, "rho"; f_rg=f_rg, size_x=size_x, size_y=size_y,
                                  size_z=size_z, offset_z=-1)
    return f_rg
end

mutable struct SiriusHamiltonian 
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

function energy_hamiltonian(basis::SiriusBasis, ψ, occupation; ρ, kwargs...)
    #returns the energies and the Hamiltonian calculated by SIRIUS

    ham = SiriusHamiltonian(basis, ρ)
    energies = get_sirius_energies(basis)

    (; energies, ham)
end

function sirius_diagonalize(H0::SiriusHamiltonian, nev_per_kpoint::Int; tol=1.0e-8, maxiter=100, kwargs...)
    #TODO: currently uses SIRIUS' internal guess for Ψ, should we enable passing the guess as well?
    converged, niter = SIRIUS.diagonalize_hamiltonian(H0.basis.sirius_gs, H0.ham, tol, maxiter)

    kpoints = H0.basis.kpoints

    #TODO: there should be no implicit allocation in there either

    #return eigenvalues, eigenvectors, niter, converged
    ispin = 1 #TODO: deal with npsins = 2 case
    λ = Vector{Vector{Float64}}(undef, length(kpoints))
    X = Vector{Matrix{ComplexF64}}(undef, length(kpoints))
    for (ik, kpt) in enumerate(kpoints)
        energies = Vector{Float64}(undef, nev_per_kpoint)
        SIRIUS.get_band_energies!(H0.basis.sirius_kps, ik_global(ik, H0.basis), ispin, energies)
        λ[ik] = energies

        n_Gk = length(G_vectors(H0.basis.pw_basis, kpt)) #TODO: add warnings as in diag.jl?
        psi = Matrix{ComplexF64}(undef, n_Gk, nev_per_kpoint)
        #TODO: could we have a problem because internally in SIRIUS, there are more bands? (contiguity)
        SIRIUS.get_psi!(H0.basis.sirius_kps, ik_global(ik, H0.basis), ispin, psi)
        
        for iel = 1:nev_per_kpoint
            psi[:, iel] = psi[H0.basis.d2s_mapping[ik], iel]
        end
        X[ik] = psi
    end
    (; λ=λ, X=X, n_iter=niter, converged=converged)
end

function sirius_compute_density(basis::SiriusBasis)
    SIRIUS.generate_density(basis.sirius_gs)
    get_sirius_density(basis)
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
    return occupation, εF
end

#TODO: a lot of duplicated code, can we make it less so?, maybe by overloading the individual
#      functions called within next_density
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
    SIRIUS.set_num_bands(ham.basis.sirius_ctx, n_bands_compute) 

    #tol is passed by kwargs as determine_diagtol, which is way too big and variable
    #TODO: figure out why and fix it. 
    eigres = sirius_diagonalize(ham, n_bands_compute)#; kwargs...) 
    eigres.converged || (@warn "Eigensolver not converged" n_iter=eigres.n_iter)

    # Check maximal occupation of the unconverged bands is sensible.
    occupation, εF = compute_occupation(ham.basis, eigres.λ, fermialg;
                                        tol_n_elec=nbandsalg.occupation_threshold)
    minocc = maximum(minimum, occupation)

    # Inherited from original next_density()
    if !increased_n_bands && minocc > nbandsalg.occupation_threshold
        @warn("Detected large minimal occupation $minocc. SCF could be unstable. " *
              "Try switching to adaptive band selection (`nbandsalg=AdaptiveBands(model)`) " *
              "or request more converged bands than $n_bands_converge (e.g. " *
              "`nbandsalg=AdaptiveBands(model; n_bands_converge=$(n_bands_converge + 3)`)")
    end

    ρout = compute_density(ham.basis.pw_basis, eigres.X, occupation; nbandsalg.occupation_threshold)

    (; ψ=eigres.X, eigenvalues=eigres.λ, occupation, εF, ρout, diagonalization=eigres,
     n_bands_converge, nbandsalg.occupation_threshold)

end

function guess_density(basis::SiriusBasis)
    SIRIUS.generate_initial_density(basis.sirius_gs)
    get_sirius_density(basis)
end

function get_gkvec_mapping(pw_basis::PlaneWaveBasis, sirius_kps::SIRIUS.KpointSetHandler)
    #Loop over k-points, then loop over integer coordinates of G+k vectors of DFTK and SIRIUS
    #to match them to each other. Necessary to exchange wave functions

    kpoints = pw_basis.kpoints

    # mapping from DFTK idx to SIRIUS idx (d2s_mapping[i_dftk] = i_sirius)
    d2s_mapping = Vector{Vector{Int}}(undef, length(kpoints))
    # mapping from SIRIUS idx to DFTK idx (s2d_mapping[i_sirius] = i_dftk)
    s2d_mapping = Vector{Vector{Int}}(undef, length(kpoints))

    #note: kpoints are local in DFTK, but need to pass global idx to SIRIUS
    for (ik, kpt) in enumerate(kpoints)
        n_Gk = length(G_vectors(pw_basis, kpt))
        sgkvec = get_gkvec(sirius_kps, ik_global(ik, pw_basis), n_Gk)
        dgkvec = kpt.G_vectors
        d2s, s2d = get_mapping(sgkvec, dgkvec, kpt.coordinate)
        d2s_mapping[ik] = d2s
        s2d_mapping[ik] = s2d
    end

    return d2s_mapping, s2d_mapping
end

function get_gkvec(sirius_kps, ik_glob, ngpts)
    gkvec__ = Matrix{Cdouble}(undef, 3, ngpts)
    SIRIUS.get_gkvec!(sirius_kps, ik_glob, gkvec__)

    gkvec = Vector{Vec3{Int}}(undef, ngpts)
    for i = 1:ngpts
       gkvec[i] = gkvec__[:, i]
    end
    return gkvec
end

function ik_global(ik, basis)
    return basis.krange_thisproc[1][ik]
end

function get_mapping(sgkvec, dgkvec, kp_coord)
    d2s = Vector{Int}(undef, length(sgkvec))
    s2d = Vector{Int}(undef, length(dgkvec))

    for (is, sg) in enumerate(sgkvec)
        for (id, dg) in enumerate(dgkvec)
            #In DFTK, test is on (G+k)**2 <= Ecut, but only G is stored
            #In SIRIUS (G+k) is stored, need to remove it for check
            sg_int = [Int(round(sg[i]-kp_coord[i])) for i in 1:3]
            if sg_int == dg
                d2s[id] = is
                s2d[is] = id
                break
            end
            if id == length(dgkvec)
                error("Missmatch in G+k vectors between DFTK and SIRIUS")
            end
        end
    end

    return d2s, s2d
end

function get_sirius_energy(basis::SiriusBasis{T}, label::String) where {T}
    return SIRIUS.get_energy(basis.sirius_gs, label)
end

default_diagtolalg(basis::SiriusBasis; tol, kwargs...) = AdaptiveDiagtol()

function mix_density(mixing, basis::SiriusBasis, Δρ; kwargs...)
    if mixing isa χ0Mixing
        error("x0Mixing is not supported with Sirius")
    end
    mix_density(mixing, basis.pw_basis, Δρ; kwargs...)
end

