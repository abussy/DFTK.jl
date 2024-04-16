using MPI
using JSON
using SIRIUS

struct SiriusBasis{T} <: AbstractBasis{T}

    # Underlying DFTK PW basis which corresponds exactly to the SIRIUS one
    PWBasis::PlaneWaveBasis{T}

    # Sirius handlers
    SiriusCtx::SIRIUS.ContextHandler
    SiriusKps::SIRIUS.KpointSetHandler
    SiriusGs::SIRIUS.GroundStateHandler     

    # Sirius parameters
    SiriusParams::Dict{Any}
                     
end

function FinalizeBasis(basis::SiriusBasis)
    SIRIUS.free_ground_state_handler(basis.SiriusGs)
    SIRIUS.free_kpoint_set_handler(basis.SiriusKps)
    SIRIUS.free_context_handler(basis.SiriusCtx)
    if SIRIUS.is_initialized()
        SIRIUS.finalize(false)
    end
end

@doc raw"""

"""
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

    #  Initialize the SIRIUS library
    if !SIRIUS.is_initialized()
        SIRIUS.initialize(false)
    end

    # Parse Model and Basis parameters into SIRIUS parameters JSON format 
    SiriusParams = CreateSiriusParams(model, Ecut)

    ParamsJson = JSON.json(SiriusParams)
    SiriusCtx = SIRIUS.create_context_from_json(PWBasis.comm_kpts, ParamsJson)
    SIRIUS.initialize_context(SiriusCtx)

    SiriusKps = SIRIUS.create_kset(SiriusCtx; num_kp=length(PWBasis.kweights_global), 
                                   k_coords=PWBasis.kcoords_global, 
                                   k_weights=PWBasis.kweights_global)

    SiriusGs = SIRIUS.create_ground_state(SiriusKps) 

    SiriusBasis(PWBasis, SiriusCtx, SiriusKps, SiriusGs, SiriusParams)
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

    #Smearing. Note: not 100% match with SIRIUS options
    smearing_type = typeof(model.smearing)
    UpdateSiriusParams(SiriusParams, "parameters", "smearing_width", model.temperature) 
    if smearing_type == Smearing.None 
        #Actually not implemented, we just use the tiniest smearing width
        UpdateSiriusParams(SiriusParams, "parameters", "smearing_width", 1.0e-16)
    elseif smearing_type == Smearing.FermiDirac
        UpdateSiriusParams(SiriusParams, "parameters", "smearing", "fermi_dirac")
    elseif smearing_type == Smearing.Gaussian
        UpdateSiriusParams(SiriusParams, "parameters", "smearing", "gaussian")
    elseif smearing_type == Smearing.MarzariVanderbilt
        UpdateSiriusParams(SiriusParams, "parameters", "smearing", "cold")
    else
        @error("Smearing type $smearing_type not implemented in SIRIUS")
    end    

    #Cutoffs work as follow: cutoff_dftk = 0.5*cutoff_qe = 0.5*cutoff_sirius^2
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
