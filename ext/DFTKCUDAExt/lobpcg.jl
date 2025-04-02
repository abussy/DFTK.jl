#TODO: description of file

function compute_λ(X::CUDA.CuArray{T}, AX::CUDA.CuArray{T}, BX::CUDA.CuArray{T}) where {T}
    num = sum(conj(X) .* AX, dims=1)
    den = sum(conj(X) .* BX, dims=1)
    vec(real.(num ./ den))
end

function diag_prod(A::CUDA.CuArray{T}, B::CUDA.CuArray{T}; diag=nothing) where {T}

    #TODO: use a CPU friendly thing here, and put this in CUDA ext
    #TODO: measure is any gains using a special case for ones, only relevant on GPU
    @assert size(A) == size(B)
    if isnothing(diag)
        res = sum(conj(A) .* B; dims=1)
    else
        @assert length(diag) == size(B, 1)
        res = sum(conj(A) .* (diag .* B); dims=1)
    end
    res
end

function ldiv!(Y::CUDA.CuArray{T}, P::PreconditionerTPA, R::CUDA.CuArray{T}) where {T}
    if P.mean_kin === nothing
        ldiv!(Y, Diagonal(P.kin .+ P.default_shift), R)
    else
        Y .= (P.mean_kin' ./ (P.mean_kin' .+ P.kin)) .* R
    end
    Y
end

function mul!(Y::CUDA.CuArray{T}, P::PreconditionerTPA, R::CUDA.CuArray{T}) where {T}
    if P.mean_kin === nothing
        mul!(Y, Diagonal(P.kin .+ P.default_shift), R)
    else
        Y .= ((P.mean_kin' .+ P.kin) ./ P.mean_kin') .* R
    end
    Y
end