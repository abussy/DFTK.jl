using MPI
using SIRIUS

#TODO: go around the code, and make sure there are no hidden tmp arrays when using slices. Do the
#      same for SIRIUS.jl

"""
A standard plane-wave basis set relying on the SIRIUS library for compuationally demanding calculations.
Note that the SiriusBasis has a PlaneWaveBasis as attribute. Both bases have the exact same parameters
(PW cutoff, fft_size, kgrid, etc.), which allows switching to the PlaneWaveBasis basis for DFTK specific
operations that are not availalbe in SIRIUS, while keeping 1 to 1 correspondance.
The SIRIUS library is available via the SIRIUS_jll.jl package. The provided binary is the most basic CPU build.
For optimal performance on the SIRIUS side, it is recommanded to build the library optimally for the local
architecture (e.g. with spack) and to change the `libsirius_path` in LocalPreferences.toml to the properly
shared object (libsirius.so).
Leveraging multiple dispatch, the SiriusBasis is implemented in such a way that most DFTK functions 
implemented for a PlaneWaveBasis work straight out of the box. This is either done via direct calls to
the SIRIUS library (builiding the Hamiltonian, diagonalization, computing forces, etc.) or by passing the
PlaneWaveBasis attribute to the function (calculating occupations, building the density from Ψ, 
density mixing, etc.). 
All in all, the following calculations can be performed with the SiriusBasis:
- SCF calculations with NC UPF pseudopotentials
- Band structure calculations
- Geometry optimizations
The above calculations can be done with the following options:
- Restricted KS and collinear magnetism
- All possible LDA and GGA XC functionals from LibXC (no meta-GGA) 
- Fermi-Dirac, Gaussian or Marzari-Vanderblit smearing
- MPI parallelization with k-points distributed over MPI ranks
- Iterative or exact diagonalization
The following is not yet implemented, but on the TODO list:
- Enabling GPU accelerated calculations with SIRIUS, with minimum data movement
- Enabling ulstrasoft pseudopotentials
"""
mutable struct SiriusBasis <: AbstractBasis{Float64}

    # Underlying DFTK PW basis which corresponds exactly to the SIRIUS one
    pw_basis::PlaneWaveBasis{Float64}

    # Sirius handlers (pointers to SIRIUS objects)
    sirius_ctx::SIRIUS.ContextHandler
    sirius_kps::SIRIUS.KpointSetHandler
    sirius_gs::SIRIUS.GroundStateHandler     

    # Mapping of of the G+k vector indices between DFTK and SIRIUS, in both directions, for each KP
    d2s_mapping::Vector{Vector{Int}}
    s2d_mapping::Vector{Vector{Int}}

    # The maximum number of bands SIRIUS can deal with (set at initialization)
    max_num_bands::Integer
    max_num_bands_factor::Real

    # A factor by which the tolerance is multiplied upon iterative eigensolver call
    iter_tol_factor::Float64

    # Whether the stdout out of SIRIUS should be silenced to avoid output noise
    is_silent::Bool

    function SiriusBasis(pw_basis, sirius_ctx, sirius_kps, sirius_gs, d2s_mapping, s2d_mapping, 
                         max_num_bands, max_num_bands_factor, iter_tol_factor, is_silent)
        x = new(pw_basis, sirius_ctx, sirius_kps, sirius_gs, d2s_mapping, s2d_mapping, max_num_bands, 
                max_num_bands_factor, iter_tol_factor, is_silent)
        finalizer(FinalizeBasis, x)
    end

end

# Finalizer for the SiriusBasis, releasing SIRIUS objects
function FinalizeBasis(basis::SiriusBasis)
    SIRIUS.free_ground_state_handler!(basis.sirius_gs)
    SIRIUS.free_kpoint_set_handler!(basis.sirius_kps)
    SIRIUS.free_context_handler!(basis.sirius_ctx)
end

# Finalizer for the SIRIUS library
function FinalizeSirius()
    if SIRIUS.is_initialized()
        SIRIUS.finalize(; call_mpi_fin=false)
    end
end

# Allow direct access to SiriusBasis.pw_basis attributes:
# e.g. SiriusBasis.Ecut --> SiriusBasis.pw_basis.Ecut
function Base.getproperty(basis::SiriusBasis, symbol::Symbol)
    if symbol in fieldnames(PlaneWaveBasis)
        return getfield(basis.pw_basis, symbol)
    else
        return getfield(basis, symbol)
    end
end

"""
The SiriusBasis constructor takes the same arguments as the PlaneWaveBasis, and feeds them to its pw_basis
attribute. All required parameters are then querried from it, and passed over to the various SIRIUS objects.
There are a few SiriusBasis specific arguments:
- max_num_bands_factor: Contrary to DFTK, SIRIUS is not flexible with its number of bands: it is set once and
                        for all at initialization. To allow for the variable bands algorithm of DFTK, enough
                        bands must be allocated from the start. This number is defined by 
                        max_num_bands_factor x n_electrons. The default value should be safe, while keeping
                        memory usage low. If not enough bands are available, an error message is issued.
- iter_tol_factor: SIRIUS energies are more sensitive to numerical noise than DFTK energies. One way
                   to increase numerical stability during the SCF is to reduce the input tolerance of
                   the iterative solver when diagonalizing the Hamiltonian. In case of slow converging
                   SCF cycle, reduce this number at the creation of the basis.
- sirius_silent: SIRIUS tends to print messages to stdout, even on its lowest verbosity setting. This can
                 generate undesiered noise. Setting sirius_silent = true will completely silence the library.
                 Note that in case of hard calculations, some insights might be gained via the SIRIUS output
"""
@DFTK.timing function DFTK.SiriusBasis(model::Model;
                          Ecut::Number,
                          kgrid=nothing,
                          kshift=nothing,
                          variational=true, fft_size=nothing,
                          symmetries_respect_rgrid=isnothing(fft_size),
                          use_symmetries_for_kpoint_reduction=true,
                          comm_kpts=MPI.COMM_WORLD, architecture=CPU(),
                          max_num_bands_factor=2.0, iter_tol_factor=0.25,
                          sirius_silent=true)
   
    # Create the PW basis on the DFTK side
    pw_basis = PlaneWaveBasis(model; Ecut, kgrid, kshift, variational, fft_size,
                              symmetries_respect_rgrid, use_symmetries_for_kpoint_reduction,
                              comm_kpts, architecture)

    # By default, make SIRIUS output to stdout silent (mostly noise). Un-silencing can be
    # useful for debugging difficult calcualations
    SIRIUS.output_mode(;make_silent=sirius_silent)

    #  Initialize the SIRIUS library
    if !SIRIUS.is_initialized()
        SIRIUS.initialize(false)
    end

    # Parse Model and Basis parameters into SIRIUS parameters JSON format 
    # TODO: is that robust for magnetic systems?
    max_num_bands = min(ceil(DFTK.default_n_bands(model)*max_num_bands_factor), 
                        get_num_bands_ub(pw_basis))
    sirius_params = create_sirius_params(model, pw_basis.Ecut, pw_basis.fft_size, max_num_bands)

    # Create SIRIUS simulation context
    sirius_ctx = SIRIUS.create_context_from_dict(pw_basis.comm_kpts, sirius_params)
    SIRIUS.initialize_context(sirius_ctx)

    # Create SIRIUS kpoint_set object
    sirius_kps = SIRIUS.create_kset(sirius_ctx; num_kp=length(pw_basis.kweights_global), 
                                    k_coords=pw_basis.kcoords_global, 
                                    k_weights=pw_basis.kweights_global,
                                    init_kset=false)

    # Insure that the k-point distribution is compatibale between SIRIUS and DFTK
    count = Vector{Int32}(undef, mpi_nprocs(comm_kpts))
    for ip = 1:mpi_nprocs(comm_kpts) 
        # note: second index is spin, spin up and spin down are distributed the same way
        count[ip] = length(pw_basis.krange_allprocs[ip][1])
    end
    SIRIUS.initialize_kset(sirius_kps; count=count)

    # Create SIRIUS ground_state object
    sirius_gs = SIRIUS.create_ground_state(sirius_kps)
 
    # Initialize SIRIUS internals for diagonalization
    SIRIUS.initialize_subspace(sirius_gs, sirius_kps)

    # Get mapping between G + k points from SIRIUS to DFTK and back
    d2s_mapping, s2d_mapping = get_gkvec_mapping(pw_basis, sirius_kps)

    SB = SiriusBasis(pw_basis, sirius_ctx, sirius_kps, sirius_gs, d2s_mapping, s2d_mapping,
                     max_num_bands, max_num_bands_factor, iter_tol_factor, sirius_silent)

    # Make sure finlalizer is called before MPI.Finalize(), because the release 
    # of some SIRIUS objects rely on MPI
    MPI.add_finalize_hook!(() -> FinalizeBasis(SB))

    # Only finalize SIRIUS library at program exit
    atexit(FinalizeSirius)

    return SB
end

# Create a new basis that is the same as the input, except for the kgrid (for band structures)
function SiriusBasis(basis::SiriusBasis, kgrid::AbstractKgrid)
    pw_basis = basis.pw_basis
    DFTK.SiriusBasis(pw_basis.model; Ecut=pw_basis.Ecut, kgrid=kgrid, variational=pw_basis.variational,
                     fft_size=pw_basis.fft_size, symmetries_respect_rgrid=pw_basis.symmetries_respect_rgrid,
                     use_symmetries_for_kpoint_reduction=pw_basis.use_symmetries_for_kpoint_reduction,
                     comm_kpts=pw_basis.comm_kpts, architecture=pw_basis.architecture,
                     max_num_bands_factor=basis.max_num_bands_factor, iter_tol_factor=iter_tol_factor,
                     sirius_silent=basis.is_silent)
end


# The SIRIUS simulation context is built based on the parameters stored in a JSON dictionary.
# This function sets the value of a given parameter, in a given input section.
function set_sirius_param(sirius_params, section, keyword, value)

    # Note: we assume only one level of nesting in Sirius input
    if !haskey(sirius_params, section) sirius_params[section] = Dict() end
    sirius_params[section][keyword] = value     
end

function set_sirius_param(basis::SiriusBasis, section::String, keyword::String, value::Any)
    set_sirius_param(basis.sirius_params, section, keyword, value)
end

# This function create the whole SIRIUS input JSON structure, based on the parameters of the
# model the PlaneWaveBasis.
function create_sirius_params(model, Ecut, fft_size, num_bands)

    sirius_params = Dict()
    #TODO: take that value from architecture input, once GPU support is implemeted and tested
    set_sirius_param(sirius_params, "control", "processing_unit", "cpu")
    set_sirius_param(sirius_params, "control", "verbosity", 1)

    set_sirius_param(sirius_params, "parameters", "electronic_structure_method", "pseudopotential")
    set_sirius_param(sirius_params, "parameters", "use_scf_correction", "false")
 
    set_sirius_param(sirius_params, "parameters", "num_bands", num_bands)
    set_sirius_param(sirius_params, "parameters", "xc_functionals", get_functionals(model))

    # Note: impose exact diagonalization at initialization, because SIRIUS cannot switch from
    #       iterative to exact at runtime (but switching from exat to iterative is ok). Solver
    #       choice is done at diagonalization call (default is iterative).
    set_sirius_param(sirius_params, "iterative_solver", "type", "exact")

    # Magnetisation
    if model.n_spin_components == 1
        set_sirius_param(sirius_params, "parameters", "num_mag_dims", 0)
    else
        set_sirius_param(sirius_params, "parameters", "num_mag_dims", 1)
    end
    set_sirius_param(sirius_params, "settings", "smooth_initial_mag", true)

    #Impose DFTK FFT grid dimensions to SIRIUS for 100% compatibility
    set_sirius_param(sirius_params, "settings", "fft_grid_size", fft_size)

    #Impose DFTK grid cutoff for the PP grid for 100% compatibility
    rcut = get_rcut(model.atoms[1])
    max_rcut = rcut
    warn = false
    for el in model.atoms[1:end]
        if abs(rcut - get_rcut(el)) > 1.0e-6
            warn = true
        end
        max_rcut = max(rcut, get_rcut(el))
        rcut = get_rcut(el)
    end
    set_sirius_param(sirius_params, "settings", "pseudo_grid_cutoff", max_rcut)
    if warn
        @warn("SIRIUS expects a single rcut value for all pseudopotentials. " *
              "The maximum value of $max_rcut was taken.")
    end

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
    #TODO: for now, we use the default x2 factor of NC PPs. This will need adaptation for ultra-soft
    set_sirius_param(sirius_params, "parameters", "pw_cutoff", 2*sirius_cutoff)

    #Note: DFTK has lattice vectors as columns, SIRIUS as rows
    set_sirius_param(sirius_params, "unit_cell", "lattice_vectors", [model.lattice[:, i] for i in 1:3])
    set_sirius_param(sirius_params, "unit_cell", "atom_files", 
                        Dict(String(element_symbol(el)) => get_path(el) for el in model.atoms))

    # Atom types in the format exepected by SIRIUS
    atom_types = unique([String(element_symbol(el)) for el in model.atoms])
    set_sirius_param(sirius_params, "unit_cell", "atom_types", atom_types)

    # Atomic position in the format expected by SIRIUS. Note: initial magnitisation set at SCF start
    atoms = Dict(atom_type => [] for atom_type in atom_types)
    for (iel, el) in enumerate(model.atoms)
        pos_and_mag = zeros(Float64, 6)
        pos_and_mag[1:3] = model.positions[iel]
        append!(atoms[String(element_symbol(el))], [pos_and_mag])
    end
    set_sirius_param(sirius_params, "unit_cell", "atoms", atoms)

    return sirius_params
end

# G+k vectors are ordered differently in SIRIUS and DFTK. This function returns the mapping
# from SIRIUS to DFTK and back. This is necessary to exchange Ψ between programs
@DFTK.timing function get_gkvec_mapping(pw_basis::PlaneWaveBasis, sirius_kps::SIRIUS.KpointSetHandler)

    kpoints = pw_basis.kpoints

    # mapping from DFTK idx to SIRIUS idx (d2s_mapping[i_dftk] = i_sirius)
    d2s_mapping = Vector{Vector{Int}}(undef, length(kpoints))
    # mapping from SIRIUS idx to DFTK idx (s2d_mapping[i_sirius] = i_dftk)
    s2d_mapping = Vector{Vector{Int}}(undef, length(kpoints)) 

    #note: kpoints are local in DFTK, but need to pass global idx to SIRIUS (and spin resolved)
    for (ik, kpt) in enumerate(kpoints)
        ik_global, ispin = ik_global_and_spin(ik, pw_basis)
        n_Gk = length(G_vectors(pw_basis, kpt))
        sgkvec = get_gkvec(sirius_kps, ik_global, n_Gk)
        dgkvec = kpt.G_vectors
        d2s, s2d = get_mapping(sgkvec, dgkvec, kpt.coordinate)
        d2s_mapping[ik] = d2s
        s2d_mapping[ik] = s2d
    end

    return d2s_mapping, s2d_mapping
end

# Returns the G+k integer coordinates for a given K-point
function get_gkvec(sirius_kps, ik_glob, ngpts)
    gkvec__ = Matrix{Cdouble}(undef, 3, ngpts)
    SIRIUS.get_gkvec!(sirius_kps, ik_glob, gkvec__)

    gkvec = Vector{Vec3{Float64}}(undef, ngpts)
    for i = 1:ngpts
       gkvec[i] = gkvec__[:, i]
    end
    return gkvec
end

# Loop over G+k wave vectors of DFTK and SIRIUS, and compare integer coordinates
function get_mapping(sgkvec, dgkvec, kp_coord)
    d2s = Vector{Int}(undef, length(sgkvec))
    s2d = Vector{Int}(undef, length(dgkvec))

    if length(sgkvec) != length(dgkvec)
        error("Missmatch in G+k vectors between DFTK and SIRIUS")
    end

    # Use O(n) logic for mapping using dictionaries
    # TODO: is this really better
    #index_map = Dict([Int(round(sg[i]-kp_coord[i])) for i in 1:3] => is 
    #                 for (is, sg) in enumerate(sgkvec))
    #d2s = [index_map[dg] for dg in dgkvec]

    #index_map = Dict(dg => id for (id, dg) in enumerate(dgkvec))
    #s2d = [index_map[[Int(round(sg[i]-kp_coord[i])) for i in 1:3]] 
    #       for sg in sgkvec]


    for (is, sg) in enumerate(sgkvec)
        #In DFTK, test is on (G+k)**2 <= Ecut, but only G is stored
        #In SIRIUS (G+k) is stored, need to remove it for check
        sg_int = [Int(round(sg[i]-kp_coord[i])) for i in 1:3]
        for (id, dg) in enumerate(dgkvec)
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

# Extract the XC functionals from a model in SIRIUS libxc format
function get_functionals(model::Model)
    functionals = []
    for term in model.term_types
        if typeof(term) == Xc
            for func in term.functionals
                fstring = "XC_"*uppercase(String(func.inner.identifier))
                if startswith(fstring, "XC_MGGA_")
                    error("SIRIUS does dot support meta-GGA functionals")
                end
                push!(functionals, fstring)
            end
        end
    end
    return functionals
end

# Get the maximum possible number of bands in SIRIUS
function get_num_bands_ub(pw_basis)

    kpoints = pw_basis.kpoints

    n_Gk = typemax(Int64)
    for (ik, kpt) in enumerate(kpoints)
        n_Gk = min(n_Gk, length(G_vectors(pw_basis, kpt)))
    end

    mpi_min(n_Gk, pw_basis.comm_kpts)
end

# TODO: for now, explicitly offload everything we need to use with SiriusBasis
#       Ideally, in the future, define all generic function with AbstraBasis arguments
G_vectors(basis::SiriusBasis) = basis.fft_grid.G_vectors
G_vectors(::SiriusBasis, kpt::Kpoint) = kpt.G_vectors

random_orbitals(basis::SiriusBasis, kpt::Kpoint, howmany::Integer) = 
    random_orbitals(basis.pw_basis, kpt, howmany)

PreconditionerTPA(basis::SiriusBasis, kpt::Kpoint; default_shift=1) = 
    PreconditionerTPA(basis.pw_basis, kpt; default_shift=default_shift)


PreconditionerNone(::SiriusBasis, ::Kpoint) = I

interpolate_kpoint(data_in::AbstractVecOrMat,
                   basis_in::SiriusBasis,  kpoint_in::Kpoint,
                   basis_out::SiriusBasis, kpoint_out::Kpoint) =
    interpolate_kpoint(data_in, basis_in.pw_basis, kpoint_in, basis_out.pw_basis, kpoint_out)