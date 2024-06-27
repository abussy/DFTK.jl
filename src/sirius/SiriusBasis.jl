using MPI
using SIRIUS

#Important node: to get full equivalence (i.e. same total energies) between DFTK and SIRIUS/QE, there
#                are 2 things to do: use the simpson method on all radial grid integrations in DFTK.
#                Disable 10 a.u. arbitrary cutoff of the radial grid in SIRIUS, radial_integrals.cpp 

mutable struct SiriusBasis <: AbstractBasis{Float64}

    # Underlying DFTK PW basis which corresponds exactly to the SIRIUS one
    pw_basis::PlaneWaveBasis{Float64}

    # Sirius handlers
    sirius_ctx::SIRIUS.ContextHandler
    sirius_kps::SIRIUS.KpointSetHandler
    sirius_gs::SIRIUS.GroundStateHandler     

    # Mapping of of the G+k vector indices between DFTK and SIRIUS, in both directions, for each KP
    d2s_mapping::Vector{Vector{Int}}
    s2d_mapping::Vector{Vector{Int}}

    function SiriusBasis(pw_basis, sirius_ctx, sirius_kps, sirius_gs, d2s_mapping, s2d_mapping)
        x = new(pw_basis, sirius_ctx, sirius_kps, sirius_gs, d2s_mapping, s2d_mapping)
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

function SiriusBasis(model::Model;
                     Ecut::Number,
                     kgrid=nothing,
                     kshift=[0, 0, 0],
                     variational=true, fft_size=nothing,
                     symmetries_respect_rgrid=isnothing(fft_size),
                     use_symmetries_for_kpoint_reduction=true,
                     comm_kpts=MPI.COMM_WORLD, architecture=CPU())
   
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

    sirius_ctx = SIRIUS.create_context_from_dict(pw_basis.comm_kpts, sirius_params)
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

function set_sirius_param(sirius_params, section, keyword, value)

    # Note: we assume only one level of nesting in Sirius input
    if !haskey(sirius_params, section) sirius_params[section] = Dict() end
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
    set_sirius_param(sirius_params, "parameters", "use_scf_correction", "false")

    # Go over the terms of the model, and check that it is compatible with SIRIUS + extract XC info
    # TODO. For now, we simlpy assume it is compatible, and we do a PBE model
    set_sirius_param(sirius_params, "parameters", "xc_functionals", ["XC_GGA_X_PBE", "XC_GGA_C_PBE"])
    #TODO: this is tmp, need to figure out adaptive tolerance
    set_sirius_param(sirius_params, "iterative_solver", "type", "exact") 

    #Impose DFTK FFT grid dimensions to SIRIUS for 100% compatibility
    set_sirius_param(sirius_params, "settings", "fft_grid_size", fft_size)
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

    #Note: DFTK has lattice vectors as columns, SIRIUS as rows
    set_sirius_param(sirius_params, "unit_cell", "lattice_vectors", [model.lattice[:, i] for i in 1:3])
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

    gkvec = Vector{Vec3{Float64}}(undef, ngpts)
    for i = 1:ngpts
       gkvec[i] = gkvec__[:, i]
    end
    return gkvec
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
