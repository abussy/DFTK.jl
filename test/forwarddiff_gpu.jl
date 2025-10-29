### ForwardDiff tests on the GPU, closely following forwarddiff.jl tests

### CUDA tests:
@testitem "Force derivatives using ForwardDiff (CUDA)" #=
    =#    tags=[:gpu] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    using CUDA
    ForwardDiffCases.test_FD_force_derivatives(DFTK.GPU(CuArray))
end

@testitem "Strain sensitivity using ForwardDiff (CUDA)" #=
    =#    tags=[:gpu] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    using CUDA
    ForwardDiffCases.test_FD_strain_sensitivity(DFTK.GPU(CuArray))
end

@testitem "scfres PSP sensitivity using ForwardDiff (CUDA)" #=
    =#    tags=[:gpu] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    using CUDA
    ForwardDiffCases.test_FD_psp_sensitivity(DFTK.GPU(CuArray))
end

@testitem "Functional force sensitivity using ForwardDiff (CUDA)" #=
    =#    tags=[:gpu] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    using CUDA
    ForwardDiffCases.test_FD_force_sensitivity(DFTK.GPU(CuArray))
end

@testitem "LocalNonlinearity sensitivity using ForwardDiff (CUDA)" #=
    =#    tags=[:gpu] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    using CUDA
    ForwardDiffCases.test_FD_local_nonlinearity_sensitivity(DFTK.GPU(CuArray))
end

@testitem "Symmetries broken by perturbation are filtered out (CUDA)" #=
    =# tags=[:gpu] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    using CUDA
    ForwardDiffCases.test_FD_filter_broken_symmetries(DFTK.GPU(CuArray))
end

@testitem "Symmetry-breaking perturbation using ForwardDiff (CUDA)" #=
    =# tags=[:gpu] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    using CUDA
    ForwardDiffCases.test_FD_symmetry_breaking_perturbation(DFTK.GPU(CuArray))
end

@testitem "Test scfres dual has the same params as scfres primal (CUDA)" #=
    =# tags=[:gpu] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    using CUDA
    ForwardDiffCases.test_FD_scfres_parameter_consistency(DFTK.GPU(CuArray))
end

@testitem "ForwardDiff wrt temperature (CUDA)" #=
    =# tags=[:gpu] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    using CUDA
    ForwardDiffCases.test_FD_wrt_temperature(DFTK.GPU(CuArray))
end

### AMDGPU tests:
@testitem "Force derivatives using ForwardDiff (AMDGPU)" #=
    =#    tags=[:gpu] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    using AMDGPU 
    ForwardDiffCases.test_FD_force_derivatives(DFTK.GPU(ROCArray))
end

@testitem "Strain sensitivity using ForwardDiff (AMDGPU)" #=
    =#    tags=[:gpu] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    using AMDGPU
    ForwardDiffCases.test_FD_strain_sensitivity(DFTK.GPU(ROCArray))
end

@testitem "scfres PSP sensitivity using ForwardDiff (AMDGPU)" #=
    =#    tags=[:gpu] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    using AMDGPU
    ForwardDiffCases.test_FD_psp_sensitivity(DFTK.GPU(ROCArray))
end

@testitem "Functional force sensitivity using ForwardDiff (AMDGPU)" #=
    =#    tags=[:gpu] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    using AMDGPU
    ForwardDiffCases.test_FD_force_sensitivity(DFTK.GPU(ROCArray))
end

@testitem "LocalNonlinearity sensitivity using ForwardDiff (AMDGPU)" #=
    =#    tags=[:gpu] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    using AMDGPU
    ForwardDiffCases.test_FD_local_nonlinearity_sensitivity(DFTK.GPU(ROCArray))
end

@testitem "Symmetries broken by perturbation are filtered out (AMDGPU)" #=
    =# tags=[:gpu] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    using AMDGPU
    ForwardDiffCases.test_FD_filter_broken_symmetries(DFTK.GPU(ROCArray))
end

@testitem "Symmetry-breaking perturbation using ForwardDiff (AMDGPU)" #=
    =# tags=[:gpu] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    using AMDGPU
    ForwardDiffCases.test_FD_symmetry_breaking_perturbation(DFTK.GPU(ROCArray))
end

@testitem "Test scfres dual has the same params as scfres primal (AMDGPU)" #=
    =# tags=[:gpu] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    using AMDGPU
    ForwardDiffCases.test_FD_scfres_parameter_consistency(DFTK.GPU(ROCArray))
end

@testitem "ForwardDiff wrt temperature (AMDGPU)" #=
    =# tags=[:gpu] setup=[TestCases, ForwardDiffCases] begin
    using DFTK
    using AMDGPU
    ForwardDiffCases.test_FD_wrt_temperature(DFTK.GPU(ROCArray))
end