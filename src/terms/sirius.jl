"""
Sirius term: for a potential coming from the SIRIUS library
Notes: - this is work in progress
       - for now, we get the full potential (Hartree + Vxc + etc) from Sirius
       - the goal is to pass the density from DFTK to SIRIUS, and get the potential back
       - this way we can use DFTK to deal with the SCF, preconditioning, etc, but get new features 
         from sirius (ultrsoft PPs, full potential, etc.)
TODO:  - We start by assuming serial calculations. Once this works, we parallelize over MPI. We want
         to make sure that there is no communication there, i.e. that the k-point dist matches
       - We start by doing CPU only. One this works, we can thing on how to exchange data that
         stays on the GPU
"""

include("../../../SIRIUS/julia_module/Sirius.jl") #TODO: cannot stay as an absolute path
import .Sirius

using JSON

struct SIRIUS
    functionals::AbstractVector
    params::AbstractDict
end

function SIRIUS(functionals::AbstractVector)
    params = Dict()
    params["mixer"] = Dict()
    params["settings"] = Dict()
    params["unit_cell"] = Dict()
    params["control"] = Dict()
    params["parameters"] = Dict()
    params["nlcg"] = Dict()
    params["vcsqnm"] = Dict()
    params["hubbard"] = Dict()
    SIRIUS(functionals, params)
end

function (term::SIRIUS)(basis::PlaneWaveBasis{T}) where {T}
    #tmp, set SIRIUS library to local so that no need to rebuild package 
    Sirius.libpath = ENV["LD_LIBRARY_PATH"]*"/libsirius.so" 

    if !Sirius.is_initialized()
        Sirius.initialize(false)
    end

    #TODO: need to figure out a way to exchange data between SIRIUS and DFTK, such as density,
    #      PW coeffs, etc. Ideally without MPI communication nor GPU to device intermediate
    #      Maybe we can try this by passing an initial density to SIRIUS and retrieve the 
    #      converged one? Or maybe by retrieving the wavfunction at the end of the SCF

    #TODO: need to understand the parameter correspondance bwtween the 2 codes, such that the
    #      same DFTK input using SIRIUS or not yields the same result

    #TODO: need to print detailed output to json + return energy/forces/stress

    #TODO: maybe we can translate temperature to smearing_width directly. And also make a check
    #      on the DFTK smearing scheme, and pass the corresponding one to SIRIUS
    #warn user in case some parameters passed to the model need to be actively passed to Sirius
    if basis.model.temperature > 0.0
        @warn("Temperature > 0 is not passed automatically to SIRIUS. Use the UpdateSiriusParams "*
              "function to define the corresponding smearing parameters directly in SIRIUS.")
    end

    #create a dictionary that we later dump into a JSON string 
    UpdateSiriusParams(term, "control", "processing_unit", "cpu") #TODO: get from basis.architecture
    UpdateSiriusParams(term, "control", "verbosity", 1) #TODO: set to zero, and only show cool stuff

    #TODO: probably need to pass the method to the model when constructing it
    UpdateSiriusParams(term, "parameters", "electronic_structure_method", "pseudopotential")
    UpdateSiriusParams(term, "parameters", "xc_functionals", [String(func) for func in term.functionals])
    #TODO: need to exactly figure out the correspondance of these wrt to DFTK
    UpdateSiriusParams(term, "parameters", "gk_cutoff", basis.Ecut)
    UpdateSiriusParams(term, "parameters", "pw_cutoff", 2*basis.Ecut)

    UpdateSiriusParams(term, "unit_cell", "lattice_vectors", basis.model.lattice)
    UpdateSiriusParams(term, "unit_cell", "atom_files", 
                       Dict(String(el.symbol) => el.fname for el in basis.model.atoms))

    atom_types = unique([String(el.symbol) for el in basis.model.atoms])
    UpdateSiriusParams(term, "unit_cell", "atom_types", atom_types)

    atoms = Dict(atom_type => [] for atom_type in atom_types)
    for (iel, el) in enumerate(basis.model.atoms)
        append!(atoms[String(el.symbol)], [basis.model.positions[iel]])
    end
    UpdateSiriusParams(term, "unit_cell", "atoms", atoms)

    param_json = JSON.json(term.params)
    ctx = Sirius.create_context_from_json(basis.comm_kpts, param_json)
    Sirius.initialize_context(ctx)

    # Create the Kpoint set from the basis. TODO: We assume MonkhorstPack grid for now
    k_grid = Vector{Int32}(basis.kgrid.kgrid_size)
    k_shift = Vector{Int32}([0, 0, 0]) #TODO: figure out how shift works in both codes
    use_symmetry = basis.use_symmetries_for_kpoint_reduction
    kps = Sirius.create_kset_from_grid(ctx, k_grid, k_shift, use_symmetry)

    gs = Sirius.create_ground_state(kps)

    TermSirius(ctx, kps, gs)
end

mutable struct TermSirius <: TermNonlinear

    # Sirius handlersgg
    ctx::Sirius.ContextHandler
    kps::Sirius.KpointSetHandler
    gs::Sirius.GroundStateHandler
end

function FinalizeSirius(term::TermSirius)
    Sirius.free_ground_state_handler(term.gs)
    Sirius.free_kpoint_set_handler(term.kps)
    Sirius.free_context_handler(term.ctx)
    if Sirius.is_initialized()
        Sirius.finalize(false)
    end
end

function FinalizeSirius(term::Term)
    #does nothing
end

function FinalizeSirius(basis::PlaneWaveBasis{T}) where {T}
    for term in basis.terms
        FinalizeSirius(term)
    end
end

function UpdateSiriusParams(term::SIRIUS, section::String, keyword::String, value::Any)

    #Note: we assume only one level of nesting in Sirius input
    if !haskey(term.params, section) term.params[section] = Dict() end
    term.params[section][keyword] = value
end

function UpdateSiriusParams(model::Model, section::String, keyword::String, value::Any)
    #update the ctx parameters
    for term_type in model.term_types
        if typeof(term_type) != SIRIUS continue end

        UpdateSiriusParams(term_type, section, keyword, value)
    end
end

function SiriusSCF(basis::PlaneWaveBasis{T}; density_tol=1.0e-6, energy_tol=1.0e-6, 
                   iter_solver_tol=1.0e-2, max_niter=100) where {T}
    for term in basis.terms
        if typeof(term) == TermSirius
            Sirius.find_ground_state(term.gs, true, true; density_tol, energy_tol, 
                                     iter_solver_tol, max_niter)
        end
    end
end

function GetSiriusForces(basis::PlaneWaveBasis{T}, label::String) where {T}
    for term in basis.terms
        if typeof(term) == TermSirius
            return Sirius.get_forces(term.gs, label)
        end
    end
end

function GetSiriusStress(basis::PlaneWaveBasis{T}, label::String) where {T}
    for term in basis.terms
        if typeof(term) == TermSirius
            return Sirius.get_stress_tensor(term.gs, label)
        end
    end
end