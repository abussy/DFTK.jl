"""
Abstract supertype for architectures supported by DFTK.
"""
abstract type AbstractArchitecture end

struct CPU <: AbstractArchitecture 
    nonlocal_batch_size #TODO: document this, explain default
end

function CPU(; nonlocal_batch_size=nothing)
    CPU(nonlocal_batch_size)
end

struct GPU{ArrayType <: AbstractArray} <: AbstractArchitecture
    nonlocal_batch_size
end

"""
Construct a particular GPU architecture by passing the ArrayType
"""
function GPU(::Type{T}; nonlocal_batch_size=nothing) where {T <: AbstractArray}
    GPU{T}(nonlocal_batch_size)
end

"""
Transfer an array from a device (typically a GPU) to the CPU.
"""
to_cpu(x::AbstractArray) = Array(x)
to_cpu(x::Array) = x

"""
Transfer an array to a particular device (typically a GPU)
"""
to_device(::CPU, x) = to_cpu(x)
to_device(::GPU{ArrayType}, x::AbstractArray) where {ArrayType} = ArrayType(x)
to_device(::GPU{ArrayType}, x::ArrayType)     where {ArrayType} = x

"""
Synchronize data and finish all operations on the execution stream of the device.
This needs to be called explicitly before a task finishes (e.g. in an `@spawn` block).
"""
synchronize_device(::AbstractArchitecture) = nothing

"""
Get key memory usage statistics. What is determined depends
on the architecture and values on the calling MPI process are returned.
Explanaition of key entries:
  * `max_rss`: Maximal residual memory size of the julia process up to this point.
    Note that this number may be well above the minimal memory required to run
    the computation for performance reasons (see the '--heap-size-hint' command line flag
    of julia for details).
  * `gc_bytes`: Currently used bytes according to the Julia GC, usually a little lower
    than the currently required memory to run the computation.
  * `gpu`: Currently used memory on the GPU device associated to this MPI process.
"""
function memory_usage(::CPU)
    (; max_rss=Sys.maxrss(), gc_bytes=Base.gc_live_bytes())
end

"""
Returns the architecture of the given array, independent of the element type.
"""
architecture(x::AbstractArray) = x isa AbstractGPUArray ? GPU(typeof(x).name.wrapper) : CPU()
