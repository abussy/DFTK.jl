"""
Returns the unique norms of input vector 'Gs' and a mapping such that\
norm(Gs[i]) = unique_ps[iG2ifnorm[i]]. Runs on CPU and GPU.
"""
function unique_norms_and_mapping(Gs::AbstractVector{Vec3{T}}) where {T}
    # Sort the norms and remember where each original element were
    ps = map(norm, Gs)
    perm = sortperm(ps)
    sorted_ps = p[perm]

    # Mark the first occurrence of each distinct value in the sorted list
    diffs = diff(sorted_ps)
    isnew = similar(sorted_ps, Bool, length(sorted_ps))
    isnew[1:1] .= true
    isnew[2:end] .= diffs .!= zero(T)

    # Use cumulative sum to assign a unique group id to each distinct value
    group_id_sorted = cumsum(isnew)
    iG2ifnorm = similar(group_id_sorted)
    iG2ifnorm[perm] = group_id_sorted

    # Keep only the unique norms
    unique_ps = sorted_ps[isnew]

    (; unique_ps, iG2ifnorm)
end