"""
Contains a collection of functions to run black-box SCF or NLCG SIRIUS calculations
and query the results (energy, forces, stress)
"""
function SiriusSCF(basis::SiriusBasis{T}; density_tol=1.0e-6, energy_tol=1.0e-6, 
                   iter_solver_tol=1.0e-2, max_niter=100) where {T}

    SIRIUS.find_ground_state(basis.SiriusGs, true, true; density_tol, energy_tol, 
                             iter_solver_tol, max_niter) 
end


function SiriusNlcg(basis::SiriusBasis{T}; kappa=0.3, tau=0.1, tol=1.0e-9, 
                    maxiter=300, restart=10) where {T}

    #get params that are already set
    temp = 315775.326864009*basis.SiriusParams["parameters"]["smearing_width"] #convert from Ha to K
    smearing = basis.SiriusParams["parameters"]["smearing"]
    processing_unit = basis.SiriusParams["control"]["processing_unit"]

    SIRIUS.nlcg(basis.SiriusGs, basis.SiriusKps; temp, smearing, kappa, tau, tol, 
                maxiter, restart, processing_unit)
end

function GetSiriusEnergy(basis::SiriusBasis{T}, label::String) where {T}
    return SIRIUS.get_energy(basis.SiriusGs, label)
end

function GetSiriusForces(basis::SiriusBasis{T}, label::String) where {T}
    return SIRIUS.get_forces(basis.SiriusGs, label)
end

function GetSiriusStress(basis::SiriusBasis{T}, label::String) where {T}
    return SIRIUS.get_stress_tensor(basis.SiriusGs, label)
end