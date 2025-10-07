# GPU workarounds for core density integration. Standard implementation
# is far from being GPU portable due to many custim data strctures that
# are not isbits, and therefore cannot be passed to GPU kernels. instead
# we only rewrite the performance critical part in a GPU optimized way.
function atomic_density_inner_loop!(form_factors_cpu, norm_indices, igroup,
                                    element::ElementPsp{<:PspUpf},
                                    method::CoreDensity,
                                    arch::GPU{AT}) where {AT}

    rgrid = to_device(arch, @view element.psp.rgrid[1:element.psp.ircut])
    r2_ρcore = to_device(arch, @view element.psp.r2_ρcore[1:element.psp.ircut])
    ps = to_device(arch, collect(keys(norm_indices)))
    ints = similar(ps)

    ints .= 0.0
    !has_core_density(element) && return ints

    map!(ints, ps) do p
        hankel(rgrid, r2_ρcore, 0, p)
    end

    ints_cpu = to_cpu(ints)
    for (p, I) in zip(keys(norm_indices), ints_cpu)
        form_factors_cpu[norm_indices[p], igroup] = I
    end
end