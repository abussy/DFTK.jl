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

    #create a dictionary that we later dump into a JSON string 
    UpdateSiriusParams(term, "control", "processing_unit", "cpu") #TODO: get from basis.architecture

    #TODO: probably need to pass the method to the model when constructing it (PAW, full-potential)
    #TODO: also need to pass magnetism information there
    UpdateSiriusParams(term, "parameters", "electronic_structure_method", "pseudopotential")
    UpdateSiriusParams(term, "parameters", "xc_functionals", [String(func) for func in term.functionals])

    #Smearing. Only some of it available in SIRIUS
    smearing_type = typeof(basis.model.smearing)
    UpdateSiriusParams(term, "parameters", "smearing_width", basis.model.temperature) 
    if smearing_type == Smearing.None 
        #Actually not implemented, we just use the tiniest smearing width
        UpdateSiriusParams(term, "parameters", "smearing_width", 1.0e-16)
    elseif smearing_type == Smearing.FermiDirac
        UpdateSiriusParams(term, "parameters", "smearing", "fermi_dirac")
    elseif smearing_type == Smearing.Gaussian
        UpdateSiriusParams(term, "parameters", "smearing", "gaussian")
    elseif smearing_type == Smearing.MarzariVanderbilt
        UpdateSiriusParams(term, "parameters", "smearing", "cold")
    else
        @error("Smearing type $smearing_type not implemented in SIRIUS")
    end


    #Cutoffs work as follow: cutoff_dftk = 0.5*cutoff_qe = 0.5*cutoff_sirius^2
    sirius_cutoff = sqrt(2*basis.Ecut)
    UpdateSiriusParams(term, "parameters", "gk_cutoff", sirius_cutoff)
    UpdateSiriusParams(term, "parameters", "pw_cutoff", 2*sirius_cutoff)

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

    kps = Sirius.create_kset(ctx, length(basis.kweights_global), basis.kcoords_global, 
                             basis.kweights_global)

    if typeof(basis.kgrid) == MonkhorstPack
        k_grid = Vector{Int32}(basis.kgrid.kgrid_size)
        k_shift = Vector{Int32}([0, 0, 0]) #TODO: deal with the shifted case
        use_symmetry = basis.use_symmetries_for_kpoint_reduction

        UpdateSiriusParams(term, "parameters", "ngridk", k_grid)
        UpdateSiriusParams(term, "parameters", "shiftk", k_shift)
        UpdateSiriusParams(term, "parameters", "use_symmetry", use_symmetry)
    end

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

function PrintSiriusInput(model::Model; fname::String="dftk_sirius_input.json")
    for term_type in model.term_types
        if typeof(term_type) != SIRIUS continue end

        open(fname,"w") do f
            JSON.print(f, term_type.params, 4)
        end
    end
end

function SiriusSCF(basis::PlaneWaveBasis{T}; density_tol=1.0e-6, energy_tol=1.0e-6, 
                   iter_solver_tol=1.0e-2, max_niter=100, print_sirius_input=false) where {T}
    for term in basis.terms
        if typeof(term) == TermSirius
            UpdateSiriusParams(basis.model, "parameters", "density_tol", density_tol)
            UpdateSiriusParams(basis.model, "parameters", "energy_tol", energy_tol)
            UpdateSiriusParams(basis.model, "parameters", "num_dft_iter", max_niter)

            if print_sirius_input
                if typeof(basis.kgrid) != MonkhorstPack
                    @warn("Only Monkhorst-Pack k-meshes are possible in SIRIUS JSON input files."*
                          "Please double check the k-mesh specifications in the printed SIRIUS input.")
                end
                PrintSiriusInput(basis.model)
            end

            Sirius.find_ground_state(term.gs, true, true; density_tol, energy_tol, 
                                     iter_solver_tol, max_niter) 
        end
    end
end

function GetSiriusEnergy(basis::PlaneWaveBasis{T}, label::String) where {T}
    for term in basis.terms
        if typeof(term) == TermSirius
            return Sirius.get_energy(term.gs, label)
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