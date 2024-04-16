module DFTKSiriusExt

using DFTK

# Because SIRIUS in DFTK requires full access to the internals of DFTK, and
# not just the explicitely exported features, we expose everything.
excluded = [:SiriusBasis, :include, :eval]
for name in names(DFTK, all=true)
    if isdefined(DFTK, name) && !(name in excluded) && Base.isidentifier(name)
        @eval import DFTK: $(name)
    end
end

include("SiriusBasis.jl")
include("SiriusHamiltonian.jl")
include("sirius_utils.jl")

end

