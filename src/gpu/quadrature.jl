# GPU workarounds for atomic grid integrations
using GPUArraysCore

function to_device(architecture, psp::PspUpf)
    PspUpf{eltype(psp.rgrid),typeof(psp.vloc_interp)}(
        psp.Zion, psp.lmax,
        to_device(architecture, psp.rgrid),
        to_device(architecture, psp.drgrid),
        to_device(architecture, psp.vloc),
        to_device(architecture, psp.buffer),
        to_device(architecture, psp.indices),
        psp.r2_projs,
        psp.h,
        psp.r2_pswfcs,
        psp.pswfc_occs,
        psp.pswfc_energies,
        psp.pswfc_labels,
        psp.r2_ρion,
        psp.r2_ρcore, 
        psp.vloc_interp,
        psp.r2_projs_interp,
        psp.r2_ρion_interp,
        psp.r2_ρcore_interp,
        psp.rcut,
        psp.ircut,
        psp.identifier,
        psp.description
    )
end

function to_device(architecture, psp::PspHgh)
    psp
end

function to_device(architecture, el::ElementPsp)
    ElementPsp(el.species, to_device(architecture, el.psp), el.family, el.mass)
end

function integrate_local_gpu(rgrid::AbstractGPUArray{T}, vloc::AbstractGPUArray{T},
                         buffer::AbstractGPUArray{T}, indices::AbstractGPUArray{Int},
                         Zion, p) where {T}

    tmp = to_cpu(rgrid[1:2])
    dx = tmp[2] - tmp[1]
    n = length(rgrid)
    n_intervals = n - 1
    istop = isodd(n_intervals) ? n - 3 : n - 1

    #Full buffer with function to integrate, with weights
    #Assume uniform grid for now
    map!(buffer, vloc, rgrid, indices) do v, r, i
        tmp = r * (r * v - -Zion * erf(r)) * sphericalbesselj_fast(0, p * r)
        if i == 1
            tmp *= 1 / 3 * dx
        elseif 1 < i <= istop
            if iseven(i)
                tmp *= 4 / 3 * dx
            else
                tmp *= 2 / 3 * dx
            end
        end

        if isodd(n_intervals)
            if i == n
                tmp *= 5 /12 * dx
            elseif i == n-1
                tmp *= dx
            elseif i == n-2
                tmp *= 15 / 12 * dx
            end
        else
            if i == n
                tmp *= 1 / 3 * dx
            end
        end
        tmp
    end
    sum(buffer)
end