# Return the unique values of a norm vector and a mapping from every input index
# to the corresponding unique entry.
#
# The implementation is array-level and works for both CPU and GPU arrays:
# it uses sortperm/cumsum/indexed assignment instead of a dictionary loop, so the
# whole operation can stay on the device for GPU architectures.
function _unique_norms_and_mapping(p::AbstractVector{T}) where {T <: Real}
    # Sort the norms and remember where each original element landed.
    perm = sortperm(p)
    sorted_p = p[perm]

    # Mark the first occurrence of each distinct value in the sorted list.
    diffs = diff(sorted_p)
    isnew = similar(sorted_p, Bool, length(sorted_p))
    isnew[1:1] .= true
    isnew[2:end] .= diffs .!= zero(T)

    # Cumulative sum turns the true/false flags into consecutive unique-group IDs
    # in the sorted order; scatter them back to the original G-vector order.
    group_id_sorted = cumsum(isnew)
    iG2ifnorm = similar(group_id_sorted)
    iG2ifnorm[perm] = group_id_sorted

    # Keep only the unique norms and the trivial row indices used by callers.
    unique_p = sorted_p[isnew]
    indices = to_device(architecture(unique_p), collect(1:length(unique_p)))

    ps = unique_p
    (; ps, iG2ifnorm, indices)
end
