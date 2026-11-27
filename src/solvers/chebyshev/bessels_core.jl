# Compute Chebyshev expansion coefficients c (in place) from samples f taken at
# Chebyshev–Lobatto nodes on [-1,1]. This is a direct O(M^2) Discrete Cosine
# Transform of type I with both endpoint half-weights
# c[m+1] = 2/M*(0.5*f[1] + sum(f[j+1]*cospi((j*m)/M) for j in 1:M-1) + 0.5*((-1)^m)*f[M+1])
# then finally the polyeval is p(x) = 1/2*c[1]*T_0(x) + sum(c[k+1]*T_{k}(x) for k in 1:M-1) + 1/2*c[end]*T_M(x)
# where T_k(x) is the k-th Chebyshev polynomial.
@inline function _chebfit_no_cospi_precomp_weights!(c::Vector{ComplexF64},f::Vector{ComplexF64},M::Int)::Vector{ComplexF64}
    # precompute the start/endpoint corrections as s = 0.5*f[1] + 0.5*((-1)^m)*f[M+1]
    final_mul=2/M
    half_f=0.5*f[1] # startpoint correction
    f_end=f[M+1] # endpoint correction wihtout the halfweight since it is m dependant due to the (-1)^m term
    mul_endp_corr= -0.5 # start odd and then mul each iteration with (-1)
    @inbounds for m in 0:M
        mul_endp_corr= -mul_endp_corr # alternate the sign for (-1)^m term
        s=muladd(f_end,mul_endp_corr,half_f) # s = 0.5*f[1] + 0.5*((-1)^m)*f[M+1] - precomputed start/end corrections
        for j in 1:M-1 # sum(f[j+1]*cospi((j*m)/M) for j in 1:M-1) inner sum part
            s=muladd(f[j+1],cospi((j*m)/M),s) # !!! EXTREMELY SLOW, cospi needs to be precomputed 
        end
        c[m+1]=final_mul*s
    end # c[k+1] = 2/M*(0.5*f[1] + sum(f[j+1]*cospi((j*k)/M) for j in 1:M-1) + 0.5*((-1)^k)*f[M+1])
    # for the final polynomial evaluation, we need to halve the first and last coefficients
    # now we need p(x) = {1/2*c[1]}*T_0(x) + sum({c[k+1]}*T_{k}(x) for k in 1:M-1) + {1/2*c[end]}*T_M(x) in the _cheb_clenshaw below
    c[1]*=0.5
    c[end]*=0.5
    # c now corresponds to p(x) = sum(c[k+1]*T_{k}(x) for k in 0:M)
    return c
end

# Supported from M=3:30, you should probably never have to go higher then 30 since the optimalpanelization.jl 
# favors increasing panel count (smaller panels so lower the poly degree needed for a wanted precision)
# 
# Instead of above doing c[m+1] = (2/M)*endpoint(m)*(0.5*f[1] + sum(f[j+1]*cospi((j*m)/M) for j in 1:M-1) + 0.5*((-1)^m)*f[M+1])
# WE DO c[k] = sum(W[j,k]*f[j] for j in 1:N) with precomputed weights W
# where W[j+1,m+1] = (2/M)*endpoint(j)*endpoint(m)*cos(pi*j*m/M), [j,m = 0:M], endpoint(q) = 0.5 at q = 0 or M, and 1.0 otherwise.
#
# C[1]: This makes sure that C[1] = sum(W[l,1]*f[l] for l in 1:N) = sum(W[l+1,1]*f[l+1] for l in 0:N-1)
#       By construction then m=0 and W[l+1,1] = (2/M)*endpoint(l)*endpoint(0)*cos(pi*(l)*0/M) = (2/M)*endpoint(l)*0.5*1 = (1/M)*endpoint(l)
#
# C[M+1]: This makes sure that C[M+1] = sum(W[l,M+1]*f[l] for l in 1:N) = sum(W[l+1,M+1]*f[l+1] for l in 0:N-1)
#         By construction then m=M and W[l+1,M+1] = (2/M)*endpoint(l)*endpoint(M)*cos(pi*(l)*M/M) = (2/M)*endpoint(l)*0.5*(-1)^l = (1/M)*endpoint(l)*(-1)^l
#
# C[k]: For general k, C[k] = sum(W[l,k]*f[l] for l in 1:N) = sum(W[l+1,k]*f[l+1] for l in 0:N-1)
#       = sum((2/M)*endpoint(l)*endpoint(k-1)*cos(pi*(l)*(k-1)/M)*f[l+1] for l in 0:N-1)
#       = (2/M)*endpoint(k-1)*(0.5*f[1] + sum(f[j+1]*cospi((j*(k-1))/M) for j in 1:M-1) + 0.5*((-1)^(k-1))*f[M+1]) as in the _chebfit_no_cospi_precomp_weights!
@inline function _chebfit_with_cospi_precomp_weights!(c::Vector{ComplexF64},f::Vector{ComplexF64},M::Int)::Vector{ComplexF64}
    N=length(f) # total number of sample points, corresponding to M+1 Chebyshev nodes
    @inbounds W=_CHEBFIT_WEIGHTS[M-2] # since hardcoded M=3:30, the first stored matrix corresponds to M=3
    @inbounds for m in 1:N
        sr=0.0
        si=0.0
        @simd for j in 1:N # c[m] = sum(W[j,m]*f[j] for j in 1:N), rest of loop self-explanatory
            z=f[j] 
            w=W[j,m]
            sr=muladd(real(z),w,sr)
            si=muladd(imag(z),w,si)
        end
        c[m]=ComplexF64(sr,si)
    end
    return c
end

@inline function _chebfit!(c::Vector{ComplexF64},f::Vector{ComplexF64})::Vector{ComplexF64}
    M=length(f)-1 # so that the chebyshev polynomial has indexing 0:M
    if 3<=M<=30
        return _chebfit_with_cospi_precomp_weights!(c,f,M)
    else
        return _chebfit_no_cospi_precomp_weights!(c,f,M)
    end
end

# Generate uniformly spaced panel breakpoints between rmin and rmax.
@inline function _breaks_uniform(rmin::Float64,rmax::Float64,np::Int)::Vector{Float64}
    h=(rmax-rmin)/np
    b=Vector{Float64}(undef,np+1)
    @inbounds Threads.@threads for i in 0:np # never called in a threaded loop above so we can thread it
        b[i+1]=muladd(i,h,rmin)
    end
    b[end]=rmax
    return b
end

@inline _cheb_nthreads_buf()::Int=Threads.maxthreadid()

# During package precompilation, generate the unrolled versions of the Clenshaw recurrence for small fixed M
# to avoid loop overhead in the innermost loops of Hankel/Bessel evaluations.
# Recurrences are for clenshaw:
# b_{k} = 2*t*b_{k+1} - b_{k+2} + c[k+1], k=M,...,1, with b_{M+1}=b_{M+2}=0.
# By construction then b_{M}=c[M+1], so we can start from k=M-1.
# Finally p(t) = t*b_{1} - b_{2} + c[1].
function _generate_clenshaw_body(M::Int,matrix::Bool)
    # Access the coefficient either from a vector or from a fixed column.
    coeff(k)=matrix ? :(coeffs[$k,col]) : :(coeffs[$k])
    # Generate all the intermediate recurrence steps explicitly.
    steps=Expr(:block)
    for k in M-1:-1:1
        push!(steps.args,quote
            b0=muladd(u,b1,$(coeff(k+1))-b2)
            b2=b1
            b1=b0
        end)
    end
    return quote
        @inbounds begin
            u=2*t
            b1=$(coeff(M+1))
            b2=zero(ComplexF64)
            $steps
            return muladd(t,b1,$(coeff(1))-b2)
        end
    end
end

# Generate the unrolled Clenshaw recurrence body functions for M=3:30 
# since tipically we will have this situation and unrolling the recurrence avoids loop overhead.
for M in 3:30
    fname=Symbol("_cheb_clenshaw_",M)
    @eval begin
        @inline function $fname(coeffs::AbstractVector{ComplexF64},t::Float64)::ComplexF64 # p(x) = sum(coeffs[k+1]*T_{k}(t) for k in 0:M)
            @boundscheck checkbounds(coeffs,1:$(M+1))
            $(_generate_clenshaw_body(M,false))
        end
        @inline function $fname(coeffs::AbstractMatrix{ComplexF64},col::Int,t::Float64)::ComplexF64 # p(x) = sum(coeffs[k+1,col]*T_{k}(t) for k in 0:M)
            @boundscheck checkbounds(coeffs,1:$(M+1),col)
            $(_generate_clenshaw_body(M,true))
        end
    end
end

# top-level dispatch for Clenshaw recurrence based on M that uses the unrolled functions for small fixed M.
function _generate_clenshaw_dispatch(matrix::Bool)
    branches=Expr(:block)
    for degree in 3:30
        fname=Symbol("_cheb_clenshaw_",degree)
        call=matrix ? :($fname(coeffs,col,t)) : :($fname(c,t))
        push!(branches.args,quote
            if M==$degree
                return $call
            end
        end)
    end
    return branches
end

### HIGH LEVEL API ###

@eval begin
    @inline function _cheb_clenshaw(c::AbstractVector{ComplexF64},t::Float64)::ComplexF64
        M=length(c)-1
        # Insert the calls to the unrolled functions for M=3:30.
        $(_generate_clenshaw_dispatch(false))
        # For degrees outside 3:30, use the ordinary recurrence.
        u=2*t
        b1=c[M+1]
        b2=zero(ComplexF64)
        @inbounds for k in M-1:-1:1
            b0=muladd(u,b1,c[k+1]-b2)
            b2=b1
            b1=b0
        end
        return muladd(t,b1,c[1]-b2)
    end

    @inline function _cheb_clenshaw_col(coeffs::AbstractMatrix{ComplexF64},col::Int,M::Int,t::Float64)::ComplexF64
        # Insert the calls to the unrolled functions for M=3:30.
        $(_generate_clenshaw_dispatch(true))
        # For degrees outside 3:30, use the ordinary recurrence.
        # standard clenshaw recurrence for degrees outside 3:30
        u=2*t
        b1=coeffs[M+1,col]
        b2=zero(ComplexF64)
        @inbounds for k in M-1:-1:1
            b0=muladd(u,b1,coeffs[k+1,col]-b2)
            b2=b1
            b1=b0
        end
        return muladd(t,b1,coeffs[1,col]-b2)
    end
end