
const MAX_BLAS_THREADS=Sys.CPU_THREADS

#expand for other types of numbers
function set_precision(a)
    t = typeof(a)
    return t == Float32 ? Float32(1e-8) : convert(t,1e-16) 
end

# Runs a loop with ProgressMeter only when requested.
macro maybe_showprogress(show_progress, loop)
    return esc(:($show_progress ? (@showprogress $loop) : $loop))
end
"""
    use_threads(args...)
    
The macro expects either:
(a) A keyword argument "multithreading" followed by a loop expression, or
b) A lone loop expression (in which case multithreading defaults to true).
NOTE: Already @inbounds 
"""
macro use_threads(args...)
    if length(args)>=2 && args[1] isa Expr && args[1].head==:(=) && args[1].args[1]==:multithreading
        if length(args)!=2
            error("Usage: @use_threads multithreading=[true|false] for ...")
        end
        # Extract the provided multithreading value and the loop expression.
        multithreading_val=args[1].args[2]
        loop_expr=args[2]
        # If the provided value is literally true or false, we branch accordingly at compile time.
        if multithreading_val===true
            return esc(:(@inbounds Threads.@threads $loop_expr))
        elseif multithreading_val===false
            return esc(:(@inbounds $loop_expr))
        else
            # If not a literal (i.e. it's some expression that will be evaluated at runtime),
            # we generate code that conditionally selects the threaded version.
            return esc(quote
                @inbounds begin
                    if $multithreading_val
                        Threads.@threads $loop_expr
                    else
                        $loop_expr
                    end
                end
            end)
        end
    elseif length(args)==1
        # No keyword argument provided. Default behavior is multithreading=true.
        loop_expr=args[1]
        return esc(:(@inbounds Threads.@threads $loop_expr))
    else
        error("Usage: @use_threads [multithreading=[true|false]] for ...")
    end
end

"""
    @blas_threads n expr

Temporarily set BLAS threads to `n` for `expr`, then restore the previous value.
"""
macro blas_multi(n,expr)
    quote
        _old_blas_threads=LinearAlgebra.BLAS.get_num_threads()
        LinearAlgebra.BLAS.set_num_threads($(esc(n)))
        $(esc(expr))
        LinearAlgebra.BLAS.set_num_threads(_old_blas_threads)    
    end
end

"""
    @blas_1 expr

Temporarily set BLAS threads to `1` for `expr`.
"""
macro blas_1(expr)
    quote
        LinearAlgebra.BLAS.set_num_threads(1)
        $(esc(expr))
    end
end

"""
    @blas_multi_then_1 n expr

Set BLAS threads to `n` for `expr`, then set it to 1 afterward.
"""
macro blas_multi_then_1(n,expr)
    quote
        LinearAlgebra.BLAS.set_num_threads($(esc(n)))
        $(esc(expr))
        LinearAlgebra.BLAS.set_num_threads(1)
    end
end

"""
    try_MKL!()

Tries to use the MKL on x86_64 architecture if possible. Otherwise it defaults to the stock BLAS backend :lbt.
macOS currently not supported due to Accelerate.jl causing issues with non-even FFTW. MKL.jl does not have this issue (for now).
"""
function try_MKL!()
    if Sys.ARCH==:x86_64
        try
            @eval using MKL
            println(BLAS.get_config())
        catch e
            println(e)
            @warn "Install Math Kernel Library (MKL) via MKL.jl"
            @info "Defaulting to stock BLAS backend: $(BLAS.vendor())"
        end
    else
        @info "Not on x86_64 architecture ($(Sys.ARCH)), defaulting to stock BLAS backend: $(BLAS.vendor())"
    end
end