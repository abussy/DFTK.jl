include("lobpcg_hyper_impl.jl")

# Note that this function will return λ on the CPU,
# but X and the history on the device (for GPU runs)
function lobpcg_hyper(A, X0; maxiter=100, prec=nothing,
                      tol=20size(A, 2)*eps(real(eltype(A))),
                      largest=false, n_conv_check=nothing, kwargs...)
    prec === nothing && (prec = I)
    @assert !largest "Only seeking the smallest eigenpairs is implemented."

    n_bands = size(X0, 2)
    n_batches = min(3, n_bands) #TODO: pass that through arch    
    batch_size = ceil(Int, n_bands / n_batches)
    results = []

    Q = nothing
    for i_batch in 1:n_batches
        first_band_idx = (i_batch-1)*batch_size + 1
        last_band_idx = min(i_batch*batch_size, n_bands)
        batch_range = first_band_idx : last_band_idx
        length(batch_range) == 0 && continue
        batch_n_conv_check = nothing
        if !isnothing(n_conv_check) && last_band_idx ≥ n_conv_check
            batch_n_conv_check = max(1, length(first_band_idx:n_conv_check))
        end
        res = LOBPCG(A, @view(X0[:, batch_range]), I, prec, tol, maxiter;
                     n_conv_check=batch_n_conv_check, Q, kwargs...)
        Q = isnothing(Q) ? res.X : hcat(Q, res.X)
        push!(results, (; λ = res.λ, residual_norms = res.residual_norms,  # save memory by not storing X
                       n_matvec = res.n_matvec, residual_history = res.residual_history))
    end

    λ = vcat((res.λ for res in results)...)
    p = sortperm(λ)
    result = (; λ = λ[p],
                X = Q[:, p],
                residual_norms = vcat((res.residual_norms for res in results)...)[p],
                n_matvec = sum(res.n_matvec for res in results))

    n_conv_check === nothing && (n_conv_check = size(X0, 2))
    converged = maximum(result.residual_norms[1:n_conv_check]) < tol
    n_iter = sum(size(res.residual_history, 2) for res in results) - length(results)

    merge(result, (; n_iter, converged))
end
