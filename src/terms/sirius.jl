"""
Sirius term: for a potential coming from the SIRIUS library
Notes: this is work in progress. For now, this allows to create a model very much like
       the rest of DFTK, but with SIRIUS as the driver for the SCF. For example:

        using DFTK                                                                                           
        using MKL                                                                                            

        a = 3.37                                                                                             
        lattice = a * [[0 1 1.];                                                                             
                       [1 0 1.];                                                                             
                       [1 1 0.]]                                                                             
        positions = [ones(3)/8, -ones(3)/8]                                                                  

        #Potential file coming from SIRIUS' upf_to_json tool
        C = ElementSirius(:C; fname=("./C_molsim.upf.json"))                                                 
        atoms     = [C, C]                                                                                   
        model = model_SIRIUS(lattice, atoms, positions, ["XC_GGA_X_PBE", "XC_GGA_C_PBE"];                    
                             temperature=0.1, smearing=DFTK.Smearing.FermiDirac(),                           
                             spin_polarization=:collinear)                                                   

        UpdateSiriusParams(model, "control", "verbosity", 1)                                                 
        basis = PlaneWaveBasis(model; Ecut=30, kgrid=[2, 2, 2])                                              

        #initial SCF step for good NLCG guess
        SiriusSCF(basis; density_tol=1.0e-8, energy_tol=1.0e-7, max_niter=4)                                 

        #direct minimzation with NLCG
        SiriusNlcg(basis)                                                                                    

        @show energy = GetSiriusEnergy(basis, "total")
        @show forces = GetSiriusForces(basis, "total")
        @show stress = GetSiriusStress(basis, "total")
"""

using JSON
using SIRIUS

struct Sirius
    functionals::AbstractVector
    params::AbstractDict
end

function Sirius(functionals::AbstractVector)
    params = Dict()
    params["mixer"] = Dict()
    params["settings"] = Dict()
    params["unit_cell"] = Dict()
    params["control"] = Dict()
    params["parameters"] = Dict()
    params["nlcg"] = Dict()
    params["vcsqnm"] = Dict()
    params["hubbard"] = Dict()
    Sirius(functionals, params)
end

function (term::Sirius)(basis::PlaneWaveBasis{T}) where {T}
    #tmp, set SIRIUS library to local so that no need to rebuild package 
    SIRIUS.libpath = ENV["LD_LIBRARY_PATH"]*"/libsirius.so" 

    if !SIRIUS.is_initialized()
        SIRIUS.initialize(false)
    end

    #TODO: need to figure out a way to exchange data between SIRIUS and DFTK, such as density,
    #      PW coeffs, etc. Ideally without MPI communication nor GPU to device intermediate

    #create a dictionary that we later dump into a JSON string 
    UpdateSiriusParams(term, "control", "processing_unit", "cpu") #TODO: get from basis.architecture

    #TODO: might want a stand-alone function that sets SIRIUS verbosity
    #TODO: probably need to pass the method to the model when constructing it (PAW, full-potential)
    UpdateSiriusParams(term, "parameters", "electronic_structure_method", "pseudopotential")
    UpdateSiriusParams(term, "parameters", "xc_functionals", [String(func) for func in term.functionals])

    #Smearing. Note 100% match with SIRIUS options
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
    ctx = SIRIUS.create_context_from_json(basis.comm_kpts, param_json)
    SIRIUS.initialize_context(ctx)

    kps = SIRIUS.create_kset(ctx; num_kp=length(basis.kweights_global), k_coords=basis.kcoords_global, 
                             k_weights=basis.kweights_global)

    if typeof(basis.kgrid) == MonkhorstPack
        k_grid = Vector{Int32}(basis.kgrid.kgrid_size)
        k_shift = Vector{Int32}([0, 0, 0]) #TODO: deal with the shifted case
        use_symmetry = basis.use_symmetries_for_kpoint_reduction

        UpdateSiriusParams(term, "parameters", "ngridk", k_grid)
        UpdateSiriusParams(term, "parameters", "shiftk", k_shift)
        UpdateSiriusParams(term, "parameters", "use_symmetry", use_symmetry)
    end

    gs = SIRIUS.create_ground_state(kps)

    TermSirius(ctx, kps, gs)
end

mutable struct TermSirius <: TermNonlinear

    # Sirius handlersgg
    ctx::SIRIUS.ContextHandler
    kps::SIRIUS.KpointSetHandler
    gs::SIRIUS.GroundStateHandler
end

function FinalizeSirius(term::TermSirius)
    SIRIUS.free_ground_state_handler(term.gs)
    SIRIUS.free_kpoint_set_handler(term.kps)
    SIRIUS.free_context_handler(term.ctx)
    if SIRIUS.is_initialized()
        SIRIUS.finalize(false)
    end
end

function FinalizeSirius(basis::PlaneWaveBasis{T}) where {T}
    for term in basis.terms
        if typeof(term) == TermSirius FinalizeSirius(term) end
    end
end

function GetSiriusCtx(basis::PlaneWaveBasis{T}) where {T}
    for term in basis.terms
        if typeof(term) == TermSirius return term.ctx end
    end
end

function GetSiriusKps(basis::PlaneWaveBasis{T}) where {T}
    for term in basis.terms
        if typeof(term) == TermSirius return term.kps end
    end
end

function GetSiriusGs(basis::PlaneWaveBasis{T}) where {T}
    for term in basis.terms
        if typeof(term) == TermSirius return term.gs end
    end
end

function UpdateSiriusParams(term_type::Sirius, section::String, keyword::String, value::Any)
    #Note: we assume only one level of nesting in Sirius input
    if !haskey(term_type.params, section) term_type.params[section] = Dict() end
    if haskey(term_type.params[section], keyword) 
        old_val = term_type.params[section][keyword]
        @warn("Overwriting SIRIUS parameter $section/$keyword. "*
              "Old value: $old_val, New value: $value")
    end
    term_type.params[section][keyword] = value
end

function UpdateSiriusParams(model::Model, section::String, keyword::String, value::Any)
    #update the ctx parameters
    for term_type in model.term_types
        if typeof(term_type) == Sirius UpdateSiriusParams(term_type, section, keyword, value) end
    end
end


function UpdateSiriusParams(basis::PlaneWaveBasis{T}, section::String, keyword::String, 
                            value::Any) where {T}
    UpdateSiriusParams(basis.model, section, keyword, value)
end

function PrintSiriusParams(basis::PlaneWaveBasis{T}; fname::String="dftk_sirius_input.json") where {T}
    for term_type in basis.model.term_types
        if typeof(term_type) != Sirius continue end

        if typeof(basis.kgrid) != MonkhorstPack
            @warn("Only Monkhorst-Pack k-meshes are available in SIRIUS JSON parameters files."*
                  "Please double check the k-mesh specifications in the printed SIRIUS file.")
        end

        open(fname,"w") do f
            JSON.print(f, term_type.params, 4)
        end
    end
end

function GetSiriusParams(basis::PlaneWaveBasis{T}) where {T}
    for term_type in basis.model.term_types
        if typeof(term_type) == Sirius return term_type.params end
    end
end

function SiriusSCF(basis::PlaneWaveBasis{T}; density_tol=1.0e-6, energy_tol=1.0e-6, 
                   iter_solver_tol=1.0e-2, max_niter=100) where {T}

    UpdateSiriusParams(basis.model, "parameters", "density_tol", density_tol)
    UpdateSiriusParams(basis.model, "parameters", "energy_tol", energy_tol)
    UpdateSiriusParams(basis.model, "parameters", "num_dft_iter", max_niter)

    gs = GetSiriusGs(basis)
    SIRIUS.find_ground_state(gs, true, true; density_tol, energy_tol, iter_solver_tol, max_niter) 
end

function SiriusNlcg(basis::PlaneWaveBasis{T}; kappa=0.3, tau=0.1, tol=1.0e-9, 
                    maxiter=300, restart=10) where {T}

    #get params that are already set
    params = GetSiriusParams(basis)
    temp = 315775.326864009*params["parameters"]["smearing_width"] #convert from Ha to K
    smearing = params["parameters"]["smearing"]
    processing_unit = params["control"]["processing_unit"]

    #save new nlcg params
    UpdateSiriusParams(basis.model, "nlcg", "kappa", kappa)
    UpdateSiriusParams(basis.model, "nlcg", "tau", tau)
    UpdateSiriusParams(basis.model, "nlcg", "tol", tol)
    UpdateSiriusParams(basis.model, "nlcg", "maxiter", maxiter)
    UpdateSiriusParams(basis.model, "nlcg", "restart", restart)
    UpdateSiriusParams(basis.model, "nlcg", "processing_unit", processing_unit)
    UpdateSiriusParams(basis.model, "nlcg", "T", temp)

    gs = GetSiriusGs(basis)
    kps = GetSiriusKps(basis)
    SIRIUS.nlcg(gs, kps; temp, smearing, kappa, tau, tol, maxiter, restart, processing_unit)
end

function GetSiriusEnergy(basis::PlaneWaveBasis{T}, label::String) where {T}
    return SIRIUS.get_energy(GetSiriusGs(basis), label)
end

function GetSiriusForces(basis::PlaneWaveBasis{T}, label::String) where {T}
    return SIRIUS.get_forces(GetSiriusGs(basis), label)
end

function GetSiriusStress(basis::PlaneWaveBasis{T}, label::String) where {T}
    return SIRIUS.get_stress_tensor(GetSiriusGs(basis), label)
end