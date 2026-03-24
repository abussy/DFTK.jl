include("lobpcg_hyper_impl.jl")

# Note that this function will return λ on the CPU,
# but X and the history on the device (for GPU runs)
function lobpcg_hyper(A, X0; maxiter=100, prec=nothing,
                      tol=20size(A, 2)*eps(real(eltype(A))),
                      largest=false, n_conv_check=nothing, kwargs...)
    prec === nothing && (prec = I)
    @assert !largest "Only seeking the smallest eigenpairs is implemented."

    #TODO: should probably pass n_batches rather than batch_size (better load balancing)
    n_bands = size(X0, 2)
    batch_size = min(n_bands, 4)
    n_batches = ceil(Int, n_bands / batch_size)
    results = []

    Q = nothing
    for i_batch in 1:n_batches
        batch_range = (i_batch-1)*batch_size + 1 : min(i_batch*batch_size, n_bands)
        push!(results, LOBPCG(A, @view(X0[:, batch_range]), I, prec, tol, maxiter;
                              n_conv_check=length(batch_range), Q, kwargs...))
                              #TODO: make sure n_conv_check is correct (over tight right now)
        Q = results[end].X
    end

    λ = vcat((res.λ for res in results)...)
    p = sortperm(λ)
    result = (; λ = λ[p],
                X = hcat((res.X for res in results)...)[:, p],
                residual_norms = vcat((res.residual_norms for res in results)...)[p],
                n_matvec = sum(res.n_matvec for res in results))

    n_conv_check === nothing && (n_conv_check = size(X0, 2))
    converged = maximum(result.residual_norms[1:n_conv_check]) < tol
    n_iter = sum(size(res.residual_history, 2) for res in results) - length(results)

    merge(result, (; n_iter, converged))
end
