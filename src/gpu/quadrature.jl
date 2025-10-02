# GPU workarounds for atomic grid integrations
function internal_loop!(form_factors_cpu, norm_indices, igroup,
                        element::ElementPsp{<:PspUpf}, arch::GPU{AT}) where {AT}

    rgrid = to_device(arch, @view element.psp.rgrid[1:element.psp.ircut])
    vloc = to_device(arch, @view element.psp.vloc[1:element.psp.ircut])
    ps = to_device(arch, collect(keys(norm_indices)))
    Zion = element.psp.Zion

    ints = map(ps) do p
        if p == 0
            zero(eltype(p)) #prob want to type this correctly
        else
            #TODO: only works if simpson_uniform (or any other method)
            #      is called explicitly. The branchin in the simpson
            #      dispatcher function fails hard
            internal_integration(rgrid, vloc, Zion, p)
        end
    end

    ints_cpu = to_cpu(ints)
    for (p, I) in zip(keys(norm_indices), ints_cpu)
        form_factors_cpu[norm_indices[p], igroup] = I
    end
end
