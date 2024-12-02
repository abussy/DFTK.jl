module DFTKCUDAExt
using PrecompileTools
using CUDA
import DFTK: GPU, DispatchFunctional
using DftFunctionals
using DFTK
using Libxc

DFTK.synchronize_device(::GPU{<:CUDA.CuArray}) = CUDA.synchronize()

for fun in (:potential_terms, :kernel_terms)
    @eval function DftFunctionals.$fun(fun::DispatchFunctional,
                                       ρ::CUDA.CuMatrix{Float64}, args...)
        @assert Libxc.has_cuda()
        $fun(fun.inner, ρ, args...)
    end
end

### TODO: once there are new versions of the GPU stack, we can push this to precompile
###       code on the GPU, in oder to reduce the start-up time

@show Base.get_extension(Libxc, :LibxcCudaExt) #TODO: we could use that to make sure the
                                               #      the extension is available, so that
                                               #      we keep back compatibility with older
                                               #      versions of Julia
### TODO: this is independent of DFTK, but from Juli 1.11, I can also try to set:
###       GPUCompiler.enable_disk_cache!(), to potentially be even faster

@show Libxc.has_cuda()

# Precompilation block with a basic workflow
@setup_workload begin
    # very artificial silicon ground state example
    a = 10.26
    lattice = a / 2 * [[0 1 1.];
                       [1 0 1.];
                       [1 1 0.]]
    Si = ElementPsp(:Si; psp=load_psp("hgh/lda/Si-q4"))
    atoms     = [Si, Si]
    positions = [ones(3)/8, -ones(3)/8]
    magnetic_moments = [2, -2]

    @compile_workload begin
        model = model_DFT(lattice, atoms, positions;
                          functionals=LDA(), magnetic_moments,
                          temperature=0.1, spin_polarization=:collinear)
        basis = PlaneWaveBasis(model; Ecut=5, kgrid=[2, 2, 2], architecture=GPU(CuArray))
        ρ = guess_density(basis, magnetic_moments)
        scfres = self_consistent_field(basis; ρ=ρ0, tol=1e-2, maxiter=3,
                                       mixing=SimpleMixing(), callback=identity)
    end
end


end
