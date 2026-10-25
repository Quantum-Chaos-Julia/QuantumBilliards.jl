# Shared reference generators and tolerance constants for the integration tier.
# Contains no tests. All generators return sorted wavenumbers k (Dirichlet
# Laplacian eigenvalues are k^2) in the closed window [k1, k2], with multiplicity.

using SpecialFunctions: besselj, bessely

# Level and tension tolerances, calibrated against the observed values logged in
# scratchpad/test-suite-redesign-plan.md ("Calibration log"). Each is at least 10x the
# worst observed value.
const SPECTRUM_ATOL_BASIS = 1e-5      # DM, PSM (worst observed 8e-6 against a BIM reference)
const SPECTRUM_ATOL_VS = 1e-3         # Vergini-Saraceno (worst observed 1.8e-4)
const SPECTRUM_ATOL_BIM = 1e-6        # DLP, CFIE, Composite, Beyn, CORK (worst observed 3e-8)
const SPECTRUM_ATOL_EBIM = 1e-3       # EBIM (worst observed 7e-5 on genuine levels)
const TENSION_TOL_BASIS = 1e-3        # DM, PSM (worst observed 3e-5)
const TENSION_TOL_VS = 0.2            # Vergini-Saraceno (worst observed 0.016)
const TENSION_TOL_BIM = 1e-5          # DLP, CFIE, Composite sweeps (worst observed 1.2e-7)
const TENSION_TOL_ACCEL = 1e-10       # Beyn, CORK (worst observed 1.3e-12)
const TENSION_TOL_EBIM = 0.5          # EBIM genuine levels <= 0.04, spurious >= 2
const IMAG_TOL = 1e-7                 # |Im k| of Beyn levels (worst observed 7e-10)

# High-wavenumber tier.
const HIGH_K = 100.0
const HIGH_K_LEVELS = 5
const HIGH_K_ATOL = 1e-5              # analytic and ppw-convergence agreement (worst observed 8.5e-7)
const HIGH_K_TENSION_TOL = 1e-9       # worst observed 1.8e-11

"""
    rectangle_spectrum(a, b, k1, k2; parity_x=nothing, parity_y=nothing)

Levels `pi*sqrt(m^2/a^2 + n^2/b^2)`, `m,n >= 1`, in `[k1, k2]`. `parity_x`/`parity_y`
(`+1` or `-1`) keep only modes with that parity under reflection about the rectangle centre:
`sin(m*pi*x/a)` is even about `x = a/2` for odd `m` and odd for even `m`.
"""
function rectangle_spectrum(a::Real, b::Real, k1::Real, k2::Real; parity_x=nothing, parity_y=nothing)
    ks = Float64[]
    mmax = ceil(Int, k2 * a / pi)
    nmax = ceil(Int, k2 * b / pi)
    for m in 1:mmax, n in 1:nmax
        parity_x === nothing || (isodd(m) ? 1 : -1) == parity_x || continue
        parity_y === nothing || (isodd(n) ? 1 : -1) == parity_y || continue
        k = pi * sqrt((m / a)^2 + (n / b)^2)
        k1 <= k <= k2 && push!(ks, k)
    end
    return sort!(ks)
end

"""
    right_triangle_spectrum(a, k1, k2)

Levels `pi*sqrt(m^2 + n^2)/a`, `m > n >= 1`, of the right isosceles triangle with legs `a`.
"""
function right_triangle_spectrum(a::Real, k1::Real, k2::Real)
    ks = Float64[]
    mmax = ceil(Int, k2 * a / pi)
    for m in 2:mmax, n in 1:(m-1)
        k = pi * sqrt(m^2 + n^2) / a
        k1 <= k <= k2 && push!(ks, k)
    end
    return sort!(ks)
end

function _bisect_root(f, x1, x2, f1; xtol=1e-14)
    lo, hi, flo = x1, x2, f1
    while hi - lo > xtol * max(1.0, abs(hi))
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if sign(fm) == sign(flo)
            lo, flo = mid, fm
        else
            hi = mid
        end
    end
    return 0.5 * (lo + hi)
end

function _scan_roots(f, x1, x2; step=0.02)
    xs = range(x1, x2; length=max(2, ceil(Int, (x2 - x1) / step) + 1))
    roots = Float64[]
    xprev, fprev = xs[1], f(xs[1])
    for x in Iterators.drop(xs, 1)
        fx = f(x)
        if isfinite(fprev) && isfinite(fx) && sign(fprev) != sign(fx) && fprev != 0
            push!(roots, _bisect_root(f, xprev, x, fprev))
        end
        xprev, fprev = x, fx
    end
    return roots
end

"""
    circle_spectrum(R, k1, k2)

Levels `j_{m,n}/R` of the Dirichlet disc of radius `R`, with multiplicity 1 for `m = 0` and 2 for `m > 0`.
"""
function circle_spectrum(R::Real, k1::Real, k2::Real)
    ks = Float64[]
    m = 0
    while m < k2 * R
        xlo = max(k1 * R, m + 1e-6)
        if xlo < k2 * R
            for x in _scan_roots(t -> besselj(m, t), xlo, k2 * R)
                k = x / R
                append!(ks, m == 0 ? (k,) : (k, k))
            end
        end
        m += 1
    end
    return sort!(ks)
end

"""
    annulus_spectrum(R1, R2, k1, k2)

Levels of the concentric Dirichlet annulus `R1 < r < R2`: roots in `k` of
`J_m(k R2) Y_m(k R1) - Y_m(k R2) J_m(k R1)`, with multiplicity 1 for `m = 0` and 2 for `m > 0`.
"""
function annulus_spectrum(R1::Real, R2::Real, k1::Real, k2::Real)
    ks = Float64[]
    m = 0
    while m < k2 * R2
        f(k) = besselj(m, k * R2) * bessely(m, k * R1) - bessely(m, k * R2) * besselj(m, k * R1)
        klo = max(k1, m / R2 * 0.5)
        for k in _scan_roots(f, klo, k2; step=0.01)
            append!(ks, m == 0 ? (k,) : (k, k))
        end
        m += 1
    end
    return sort!(ks)
end

"""
    high_k_window(A; k=HIGH_K, nlevels=HIGH_K_LEVELS)

Window `(k1, k2)` starting at `k` that holds about `nlevels` levels on average for area `A`
(Weyl density `A*k/(2*pi)`). For geometries with analytic references the window must be
chosen explicitly between levels instead (the disc has no level in `[100, 100.19]`).
"""
function high_k_window(A::Real; k::Real=HIGH_K, nlevels::Real=HIGH_K_LEVELS)
    return (Float64(k), Float64(k + 2 * pi * nlevels / (A * k)))
end
