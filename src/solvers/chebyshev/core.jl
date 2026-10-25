###############################################################################
# Core Chebyshev routines for piecewise spectral approximation.
#
# Low-level building blocks used by the BIM/Beyn/EBIM Chebyshev-accelerated
# Hankel/Bessel-J evaluation (see bessels.jl, dlp.jl, cfie.jl,
# optimalpanelization.jl in this directory) to approximate special functions
# on many geometric panels with high accuracy and stability.
#
# Ported verbatim from QuantumBilliards-develop/src/chebyshev/chebyshev_core.jl
# per the migration plan's numerical-fidelity rule (see
# QuantumBilliardsTests/scratchpad/migration-plan/10-chebyshev-acceleration.md).
###############################################################################

# =============================================================================
# Compute Chebyshev expansion coefficients c (in place) from samples f taken at
# Chebyshev–Lobatto nodes on [-1,1]. This is a direct O(M^2) Discrete Cosine
# Transform of type I with endpoint half-weights.
# =============================================================================
@inline function _chebfit!(c::Vector{ComplexF64}, f::Vector{ComplexF64})::Vector{ComplexF64}
    M = length(f)-1
    @inbounds for m in 0:M
        s = 0.0+0.0im
        s += 0.5*f[1]
        s += 0.5*((isodd(m) ? -1.0 : 1.0)*f[M+1])
        for j in 1:M-1
            s += f[j+1]*cospi((j*m)/M)
        end
        c[m+1] = (2/M)*s
    end
    c[1] *= 0.5
    c[end] *= 0.5
    return c
end

# =============================================================================
# Generate uniformly spaced panel breakpoints between rmin and rmax.
# =============================================================================
@inline function _breaks_uniform(rmin::Float64, rmax::Float64, np::Int)::Vector{Float64}
    h = (rmax-rmin)/np
    b = Vector{Float64}(undef, np+1)
    @inbounds for i in 0:np
        b[i+1] = muladd(i, h, rmin)
    end
    b[end] = rmax
    return b
end

@inline _cheb_nthreads_buf()::Int = Threads.maxthreadid()

# =============================================================================
# UNROLLED CLENSHAW
# Unrolled versions of the Clenshaw recurrence for small fixed M (4 to 10) to
# avoid loop overhead, used in the innermost loops of Hankel/Bessel evaluations
# where M is typically small with large panelization.
# =============================================================================

@inline function _cheb_clenshaw_4(coeffs::AbstractVector{ComplexF64}, t::Float64)::ComplexF64
    b1 = 0.0+0.0im; b2 = 0.0+0.0im; u = 2*t
    b0 = muladd(u, b1, -b2)+coeffs[5]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[4]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[3]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[2]; b2 = b1; b1 = b0
    return muladd(t, b1, -b2)+coeffs[1]
end
@inline function _cheb_clenshaw_4(coeffs::AbstractMatrix{ComplexF64}, col::Int, t::Float64)::ComplexF64
    b1 = 0.0+0.0im; b2 = 0.0+0.0im; u = 2*t
    b0 = muladd(u, b1, -b2)+coeffs[5,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[4,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[3,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[2,col]; b2 = b1; b1 = b0
    return muladd(t, b1, -b2)+coeffs[1,col]
end

@inline function _cheb_clenshaw_5(coeffs::AbstractVector{ComplexF64}, t::Float64)::ComplexF64
    b1 = 0.0+0.0im; b2 = 0.0+0.0im; u = 2*t
    b0 = muladd(u, b1, -b2)+coeffs[6]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[5]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[4]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[3]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[2]; b2 = b1; b1 = b0
    return muladd(t, b1, -b2)+coeffs[1]
end
@inline function _cheb_clenshaw_5(coeffs::AbstractMatrix{ComplexF64}, col::Int, t::Float64)::ComplexF64
    b1 = 0.0+0.0im; b2 = 0.0+0.0im; u = 2*t
    b0 = muladd(u, b1, -b2)+coeffs[6,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[5,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[4,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[3,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[2,col]; b2 = b1; b1 = b0
    return muladd(t, b1, -b2)+coeffs[1,col]
end

@inline function _cheb_clenshaw_6(coeffs::AbstractVector{ComplexF64}, t::Float64)::ComplexF64
    b1 = 0.0+0.0im; b2 = 0.0+0.0im; u = 2*t
    b0 = muladd(u, b1, -b2)+coeffs[7]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[6]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[5]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[4]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[3]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[2]; b2 = b1; b1 = b0
    return muladd(t, b1, -b2)+coeffs[1]
end
@inline function _cheb_clenshaw_6(coeffs::AbstractMatrix{ComplexF64}, col::Int, t::Float64)::ComplexF64
    b1 = 0.0+0.0im; b2 = 0.0+0.0im; u = 2*t
    b0 = muladd(u, b1, -b2)+coeffs[7,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[6,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[5,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[4,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[3,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[2,col]; b2 = b1; b1 = b0
    return muladd(t, b1, -b2)+coeffs[1,col]
end

@inline function _cheb_clenshaw_7(coeffs::AbstractVector{ComplexF64}, t::Float64)::ComplexF64
    b1 = 0.0+0.0im; b2 = 0.0+0.0im; u = 2*t
    b0 = muladd(u, b1, -b2)+coeffs[8]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[7]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[6]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[5]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[4]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[3]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[2]; b2 = b1; b1 = b0
    return muladd(t, b1, -b2)+coeffs[1]
end
@inline function _cheb_clenshaw_7(coeffs::AbstractMatrix{ComplexF64}, col::Int, t::Float64)::ComplexF64
    b1 = 0.0+0.0im; b2 = 0.0+0.0im; u = 2*t
    b0 = muladd(u, b1, -b2)+coeffs[8,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[7,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[6,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[5,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[4,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[3,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[2,col]; b2 = b1; b1 = b0
    return muladd(t, b1, -b2)+coeffs[1,col]
end

@inline function _cheb_clenshaw_8(coeffs::AbstractVector{ComplexF64}, t::Float64)::ComplexF64
    b1 = 0.0+0.0im; b2 = 0.0+0.0im; u = 2*t
    b0 = muladd(u, b1, -b2)+coeffs[9]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[8]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[7]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[6]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[5]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[4]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[3]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[2]; b2 = b1; b1 = b0
    return muladd(t, b1, -b2)+coeffs[1]
end
@inline function _cheb_clenshaw_8(coeffs::AbstractMatrix{ComplexF64}, col::Int, t::Float64)::ComplexF64
    b1 = 0.0+0.0im; b2 = 0.0+0.0im; u = 2*t
    b0 = muladd(u, b1, -b2)+coeffs[9,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[8,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[7,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[6,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[5,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[4,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[3,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[2,col]; b2 = b1; b1 = b0
    return muladd(t, b1, -b2)+coeffs[1,col]
end

@inline function _cheb_clenshaw_9(coeffs::AbstractVector{ComplexF64}, t::Float64)::ComplexF64
    b1 = 0.0+0.0im; b2 = 0.0+0.0im; u = 2*t
    b0 = muladd(u, b1, -b2)+coeffs[10]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[9]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[8]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[7]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[6]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[5]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[4]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[3]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[2]; b2 = b1; b1 = b0
    return muladd(t, b1, -b2)+coeffs[1]
end
@inline function _cheb_clenshaw_9(coeffs::AbstractMatrix{ComplexF64}, col::Int, t::Float64)::ComplexF64
    b1 = 0.0+0.0im; b2 = 0.0+0.0im; u = 2*t
    b0 = muladd(u, b1, -b2)+coeffs[10,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[9,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[8,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[7,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[6,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[5,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[4,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[3,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[2,col]; b2 = b1; b1 = b0
    return muladd(t, b1, -b2)+coeffs[1,col]
end

@inline function _cheb_clenshaw_10(coeffs::AbstractVector{ComplexF64}, t::Float64)::ComplexF64
    b1 = 0.0+0.0im; b2 = 0.0+0.0im; u = 2*t
    b0 = muladd(u, b1, -b2)+coeffs[11]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[10]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[9]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[8]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[7]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[6]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[5]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[4]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[3]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[2]; b2 = b1; b1 = b0
    return muladd(t, b1, -b2)+coeffs[1]
end
@inline function _cheb_clenshaw_10(coeffs::AbstractMatrix{ComplexF64}, col::Int, t::Float64)::ComplexF64
    b1 = 0.0+0.0im; b2 = 0.0+0.0im; u = 2*t
    b0 = muladd(u, b1, -b2)+coeffs[11,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[10,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[9,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[8,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[7,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[6,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[5,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[4,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[3,col]; b2 = b1; b1 = b0
    b0 = muladd(u, b1, -b2)+coeffs[2,col]; b2 = b1; b1 = b0
    return muladd(t, b1, -b2)+coeffs[1,col]
end

# =============================================================================
# TOP CALLING FUNCTIONS FOR CHEBYSHEV
# =============================================================================

@inline function _cheb_clenshaw(c::AbstractVector{ComplexF64}, t::Float64)::ComplexF64
    M = length(c)-1
    if M==4
        return _cheb_clenshaw_4(c, t)
    elseif M==5
        return _cheb_clenshaw_5(c, t)
    elseif M==6
        return _cheb_clenshaw_6(c, t)
    elseif M==7
        return _cheb_clenshaw_7(c, t)
    elseif M==8
        return _cheb_clenshaw_8(c, t)
    elseif M==9
        return _cheb_clenshaw_9(c, t)
    elseif M==10
        return _cheb_clenshaw_10(c, t)
    else
        b1 = 0.0+0.0im
        b2 = 0.0+0.0im
        u = 2*t
        @inbounds for k in M:-1:1
            b = muladd(u, b1, c[k+1]-b2)
            b2 = b1
            b1 = b
        end
        return muladd(t, b1, c[1])-b2
    end
end

@inline function _cheb_clenshaw_col(coeffs::AbstractMatrix{ComplexF64}, col::Int, M::Int, t::Float64)::ComplexF64
    if M==4
        return _cheb_clenshaw_4(coeffs, col, t)
    elseif M==5
        return _cheb_clenshaw_5(coeffs, col, t)
    elseif M==6
        return _cheb_clenshaw_6(coeffs, col, t)
    elseif M==7
        return _cheb_clenshaw_7(coeffs, col, t)
    elseif M==8
        return _cheb_clenshaw_8(coeffs, col, t)
    elseif M==9
        return _cheb_clenshaw_9(coeffs, col, t)
    elseif M==10
        return _cheb_clenshaw_10(coeffs, col, t)
    else
        b1 = 0.0+0.0im
        b2 = 0.0+0.0im
        u = 2*t
        @inbounds for k in M:-1:1
            b0 = muladd(u, b1, coeffs[k+1,col]-b2)
            b2 = b1
            b1 = b0
        end
        return muladd(t, b1, coeffs[1,col])-b2
    end
end
