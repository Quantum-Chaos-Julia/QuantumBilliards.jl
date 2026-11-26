############################
# DLP and CFIE kernel Taylor expansions
# Ref: Kress, R. (1991). “Boundary Integral Equations in Time-Harmonic Acoustic Scattering.” Mathematical and Computer Modelling, 15(3–5), 229–243.
############################

# Implements the analytic Taylor expansions of the DLP and CFIE kernels using the radial recurrences for Bessel and Hankel functions via _kernel_taylor!. The ones with the Rmat overload handle the Kress logarithmic quadrature correction (these are closed curves that need singular quadrature). The ones that dont have Rmat ahve regular trapezoidal quadrature since they are classified as cross-kernel and are for holes etc. where source-target separation has no self-interaction.

# NOTE: J0 and J1 are now obtained from real(H0) and real(H1) recurrsion for performance. If in future we will want complex k0 support, this part will need to be modified.

"""
    _kernel_taylor!(::DoubleLayerPotentialSolver, w::TaylorWorkspace, pts, Rmat::Matrix{Float64}, G, k::Float64, p::Int, i::Int, j::Int) -> Vector{ComplexF64}

Compute the normalized Taylor coefficients of the DLP
entry `Dᵢⱼ(k)`. 

The returned workspace vector satisfies `w.a[n+1]=Dᵢⱼ⁽ⁿ⁾(k)/n!`. These are
coefficients of the raw operator `D(k)`; the sign and identity in
`A(k)=I-D(k)` are applied later during matrix assembly.

## Arguments
- `::DoubleLayerPotentialSolver`: DLP solver instance used for multiple
  dispatch.
- `w::TaylorWorkspace`: Reusable Taylor workspace.
- `pts`: Boundary discretization data.
- `Rmat::Matrix{Float64}`: Kress logarithmic quadrature matrix.
- `G`: Precomputed geometry cache containing distances, logarithmic factors,
  inner products, inverse distances, and curvature.
- `k::Float64`: Taylor expansion wavenumber.
- `p::Int`: Maximum Taylor degree.
- `i::Int`: Target boundary index.
- `j::Int`: Source boundary index.

## Returns
- `Vector{ComplexF64}`: `w.a`, containing the normalized Taylor coefficients
  `Dᵢⱼ⁽ⁿ⁾(k)/n!` for `n=0,...,p`.
"""
@inline function _kernel_taylor!(::DoubleLayerPotentialSolver, w::TaylorWorkspace, pts, Rmat::Matrix{Float64}, G, k::Float64, p::Int, i::Int, j::Int)::Vector{ComplexF64}
    a = w.a; fill!(a, 0)
    if i == j # standard diagonal Kress limit for DLP, the exact value is wᵢκᵢ where wᵢ is the quadrature weight and κᵢ is the curvature at the point. NOTE: 1/(2π) factor is included already in the .kappa field of G (see boundarygeomcache.jl)
        a[1] = ComplexF64(pts.ws[i]*G.kappa[i])
        return a
    end
    invtwopi = inv(2π); r = G.R[i,j]; r2 = r*r; invr = G.invR[i,j]
    lt = G.logterm[i,j]; inn = G.inner[i,j]; z = k*r
    # L1 term corresponds to -(1/2π)(inn/r)kJ₁(kr) in the DLP split.
    h0 = _bim_hankelh1(0,z); h1 = _bim_hankelh1(1,z)
    radial1_taylor!(w.H1,k,r2,p,k*h1,k*r*h0)
    wj=pts.ws[j]
    q=muladd(-wj,lt,Rmat[i,j]) # Rmat[i,j] - wj*lt
    dlp_geom_factor=0.5*inn*invr*wj
    dlp_L1_factor= -invtwopi*inn*invr*q
    @inbounds @simd for l = 1:p+1
        ### For documentation ###
        #l1 = -invtwopi*inn*invr*real(w.H1[l]) # Ref Eq. 2.6 below
        #l2 = im/2*inn*invr*w.H1[l] - l1*lt # Ref Eq. 2.6 below (this is the full kernel L(i,j) - L1 term with the Kress logarithm - the term log(4*sin^2((t_i-t_j)/2)) wihtout the diagonal divergent pointwise limit - early return diag limit))
        #a[l] = Rmat[i,j]*l1 + pts.ws[j]*l2 # Ref Eq 3.1 combine the L1 and L2 contributions with the Kress quadrature integration weights
        #########################
        h1_k=w.H1[l]
        h1_k_re=real(h1_k)
        a[l]=ComplexF64(muladd(-dlp_geom_factor,imag(h1_k),dlp_L1_factor*h1_k_re),dlp_geom_factor*h1_k_re) # dlp_L1_factor*real(H1) + im*dlp_geom_factor*H1.
    end
    return a
end

"""
    _kernel_taylor!(::CombinedFieldIntegralEquationSolver, w::TaylorWorkspace, pts, Rmat::Matrix{Float64}, G, k::Float64, p::Int, i::Int, j::Int) -> Vector{ComplexF64}

Compute normalized Taylor coefficients of the raw combined-field entry
`Kᵢⱼ(k)=Dᵢⱼ(k)+ikSᵢⱼ(k)`. The returned vector
satisfies `w.a[n+1]=Kᵢⱼ⁽ⁿ⁾(k)/n!`; the Fredholm transformation
`A(k)=I-K(k)` is performed later.

## Arguments
- `::CombinedFieldIntegralEquationSolver`: CFIE solver instance used for
  multiple dispatch.
- `w::TaylorWorkspace`: Reusable Taylor workspace.
- `pts`: Boundary discretization data.
- `Rmat::Matrix{Float64}`: Kress logarithmic quadrature matrix.
- `G`: Precomputed geometry cache containing the geometric quantities required
  by the Kress DLP and single-layer splits.
- `k::Float64`: Taylor expansion wavenumber.
- `p::Int`: Maximum Taylor degree.
- `i::Int`: Target boundary index.
- `j::Int`: Source boundary index.

## Returns
- `Vector{ComplexF64}`: `w.a`, containing the normalized Taylor coefficients
  `Kᵢⱼ⁽ⁿ⁾(k)/n!` of `K(k)=D(k)+ikS(k)` for `n=0,...,p`.
"""
@inline function _kernel_taylor!(::CombinedFieldIntegralEquationSolver, w::TaylorWorkspace, pts, Rmat::Matrix{Float64}, G, k::Float64, p::Int, i::Int, j::Int)::Vector{ComplexF64}
    a = w.a; fill!(a, 0); invtwopi = inv(2π)
    if i == j # Ref: Eq. 2.6 Diagonal contribution from the M2[t,t] term (self-interaction)
        # 0th: kernel written as Kᵢᵢ(k) = wᵢκᵢ + ik(cᵢ - aᵢ log k)
        # 1st: derivative with respect to k gives K₁ = i[cᵢ - aᵢ(log k + 1)]
        # n≥2: higher-order derivatives follow Kₙ = iaᵢ(-1)ⁿ⁺¹/[n(n-1)kⁿ⁻¹]
        si = G.speed[i]; wi = pts.ws[i]; dval = ComplexF64(wi*G.kappa[i]) 
        # see QuantumBilliards paper Appendix F. for the derivation of aᵢ and c for 
        # the implementation below. It follows from direct Taylor expansion of the diagonal kernel Kᵢᵢ(k) with respect to k.
        a_log = wi*si/π 
        c = si*(-Rmat[i,i]*invtwopi + wi*(im/2 - Base.MathConstants.eulergamma/π + invtwopi*log(4/si^2)))
        a[1] = dval + im*k*(c - a_log*log(k)) # 0th Taylor coefficient of the diagonal kernel Kᵢᵢ(k) (dval contains the DLP diagonal with the curvature)
        p >= 1 && (a[2] = im*(c - a_log*(log(k) + 1))) # 1st Taylor coefficient of the diagonal kernel Kᵢᵢ(k)
        @inbounds for n = 2:p
            a[n+1] = im*a_log*(-1)^(n+1)/(n*(n-1)*k^(n-1)) # n≥2 Taylor coefficient of the diagonal kernel Kᵢᵢ(k)
        end
        return a
    end
    # Ref: off-diagonal contributions for L1,L2,M1,M2
    r = G.R[i,j]; r2 = r*r; invr = G.invR[i,j]
    lt = G.logterm[i,j]; inn = G.inner[i,j]; sj = G.speed[j]; wj = pts.ws[j]; z = k*r
    # We need all H0, H1, J0, J1 since CFIE combines both single-layer and double-layer contributions
    h0 = _bim_hankelh1(0,z); h1 = _bim_hankelh1(1,z) # Hankel functions of the first kind for orders 0 and 1
    # see QuantumBilliards paper Appendix F. for the definition of the radial Taylor expansions
    radial1_taylor!(w.H1,k,r2,p,k*h1,k*r*h0) # Taylor coeffs for H1 - DLP part
    radial0_k_taylor!(w.H0,w.yH0,k,r2,p,h0,-r*h1) # Taylor coeffs for H0 - SLP part
    q=muladd(-wj,lt,Rmat[i,j]) # Rmat[i,j] - wj*lt
    dlp_geom_factor=0.5*inn*invr*wj
    dlp_L1_factor= -invtwopi*inn*invr*q
    slp_geom_factor= -0.5*sj*wj
    slp_M1_factor= -invtwopi*sj*q
    @inbounds @simd for l = 1:p+1
        ### for documentation ###
        #l1 = -invtwopi*inn*invr*real(w.H1[l]) # Contribution from the J1 term (off-diagonal DLP), see also _kernel_taylor! for the DLP only
        #l2 = im/2*inn*invr*w.H1[l] - l1*lt # Contribution from the H1 term (off-diagonal DLP), see also _kernel_taylor! for the DLP only
        #dval = Rmat[i,j]*l1 + wj*l2 # Off-diagonal DLP contribution with the Kress logarithm 
        #ikm1 = -im*invtwopi*sj*real(w.H0[l]) # Contribution from the J0 term (off-diagonal SLP)
        #ikm2 = -0.5*sj*w.H0[l] + im*invtwopi*sj*lt*real(w.H0[l]) # Contribution from the H0 term (off-diagonal SLP)
        #a[l] = dval + Rmat[i,j]*ikm1 + wj*ikm2 # Total off-diagonal contribution (DLP + SLP) with Kress integration weight Rmat
        ########################
        h0_k=w.H0[l];h1_k=w.H1[l]
        h0_k_re=real(h0_k);h1_k_re=real(h1_k); # kJ₀ and kJ₁ coefficients at real k
        # following 2 lines need some comment:
        # DLP = dlp_L1_factor*real(h1) + im*dlp_geom_factor*h1
        # since im*h1 = -imag(h1) + im*real(h1), the Hankel term
        # contributes -dlp_geom_factor*imag(h1) to the real part
        dlp_real=muladd(-dlp_geom_factor,imag(h1_k),dlp_L1_factor*h1_k_re)
        dlp_imag=dlp_geom_factor*h1_k_re
        # add ikS = slp_geom_factor*h0_k + im*slp_M1_factor*real(h0_k).
        cfie_real=muladd(slp_geom_factor,h0_k_re,dlp_real)
        cfie_imag=muladd(slp_geom_factor,imag(h0_k),muladd(slp_M1_factor,h0_k_re,dlp_imag))
        a[l]=ComplexF64(cfie_real,cfie_imag)
    end
    return a
end

"""
    _kernel_taylor!(::DoubleLayerPotentialSolver, w::TaylorWorkspace, pts, xi::Float64, yi::Float64, k::Float64, p::Int, j::Int) -> Vector{ComplexF64}

Compute normalized Taylor coefficients of a smooth cross-component DLP entry for target `xᵢ=(xi,yi)` and source `xⱼ=pts.xy[j]`,
## Arguments
- `::DoubleLayerPotentialSolver`: DLP source-component solver used for
  multiple dispatch.
- `w::TaylorWorkspace`: Reusable Taylor workspace.
- `pts`: Boundary data for the source component.
- `xi::Float64`: Target x-coordinate.
- `yi::Float64`: Target y-coordinate.
- `k::Float64`: Taylor expansion wavenumber.
- `p::Int`: Maximum Taylor degree.
- `j::Int`: Local source index on the source component.

## Returns
- `Vector{ComplexF64}`: `w.a`, containing normalized Taylor coefficients of
  the smooth cross-component DLP entry through degree `p`.
"""
@inline function _kernel_taylor!(::DoubleLayerPotentialSolver, w::TaylorWorkspace, pts, xi::Float64, yi::Float64, k::Float64, p::Int, j::Int)::Vector{ComplexF64}
    # see _kernel_taylor! for the Kress DLP version above, it is basically the same without the Kress split, so we only need H1 (kr) for the off-diagonal DLP (also H0 because of recurrence)
    a = w.a; fill!(a,0)
    xj,yj = pts.xy[j]; dx = xi-xj; dy = yi-yj; r = hypot(dx,dy); r2 = r*r
    tx,ty = pts.tangent[j]; inn = ty*dx-tx*dy; z = k*r
    h0 = _bim_hankelh1(0,z); h1 = _bim_hankelh1(1,z) # see above function for more details
    radial1_taylor!(w.H1,k,r2,p,k*h1,k*r*h0) # Generate normalized Taylor coefficients hₙ for (k+δ)*H₁⁽¹⁾((k+δ)*r): (k+δ)*H₁⁽¹⁾((k+δ)*r) = Σₙhₙδⁿ,
    c = pts.ws[j]*(im/2)*inn/r # Pre-factor for the off-diagonal DLP entry, since it is not Taylor expanded. See Ref: 2.2 below
    @inbounds @simd for l = 1:p+1
        a[l] = c*w.H1[l] # Multiply the pre-factor with the Taylor coefficient for kH₁⁽¹⁾(kr)
    end # Dᵢⱼ(k+δ) = Σₙ[(i/2)wⱼ(inn/r)hₙ]δⁿ.
    return a
end

"""
    _kernel_taylor!(::CombinedFieldIntegralEquationSolver, w::TaylorWorkspace, pts, xi::Float64, yi::Float64, k::Float64, p::Int, j::Int) -> Vector{ComplexF64}

Compute normalized Taylor coefficients of a smooth cross-component CFIE entry.

## Arguments
- `::CombinedFieldIntegralEquationSolver`: CFIE source-component solver used
  for multiple dispatch.
- `w::TaylorWorkspace`: Reusable Taylor workspace.
- `pts`: Boundary data for the source component.
- `xi::Float64`: Target x-coordinate.
- `yi::Float64`: Target y-coordinate.
- `k::Float64`: Taylor expansion wavenumber.
- `p::Int`: Maximum Taylor degree.
- `j::Int`: Local source index on the source component.

## Returns
- `Vector{ComplexF64}`: `w.a`, containing normalized Taylor coefficients of
  the smooth cross-component CFIE entry through degree `p`.
"""
@inline function _kernel_taylor!(::CombinedFieldIntegralEquationSolver, w::TaylorWorkspace, pts, xi::Float64, yi::Float64, k::Float64, p::Int, j::Int)::Vector{ComplexF64}
    # Ref: Eq: 2.6 below; we want CFIE kernel Kᵢⱼ(k) = Dᵢⱼ(k)+ikSᵢⱼ(k), where Dᵢⱼ(k) = c_D * (kH₁⁽¹⁾(kr)) AND ikSᵢⱼ(k) = c_S * (kH₀⁽¹⁾(kr)), coeffs calculated below
    a = w.a; fill!(a,0)
    xj,yj = pts.xy[j]; dx = xi-xj; dy = yi-yj; r = hypot(dx,dy); r2 = r*r
    tx,ty = pts.tangent[j]; inn = ty*dx-tx*dy; sj = hypot(tx,ty); z = k*r
    h0 = _bim_hankelh1(0,z); h1 = _bim_hankelh1(1,z) # H₀⁽¹⁾(kr) = h0, H₁⁽¹⁾(kr) = h1
    radial1_taylor!(w.H1,k,r2,p,k*h1,k*r*h0) # Generate normalized Taylor coefficients hₙ for kH₁⁽¹⁾(kr)
    radial0_k_taylor!(w.H0,w.yH0,k,r2,p,h0,-r*h1) # Generate normalized Taylor coefficients hₙ for kH₀⁽¹⁾(kr)
    cD = pts.ws[j]*(im/2)*inn/r # c_D = (i/2)wⱼinn/r
    cS = -pts.ws[j]*sj/2 # c_S = -½wⱼsⱼ
    @inbounds @simd for l = 1:p+1
        a[l]=muladd(cS,w.H0[l],cD*w.H1[l]) # Combine Dᵢⱼ'th and ikSᵢⱼ'th l-th Taylor coefficient contribution for the CFIE kernel Kᵢⱼ(k)
    end
    return a
end

# Reduced smooth DLP visits each (fundamental target, full source) pair once.
# Keep only one-dimensional geometry and the full-grid Kress first column.
@experimental struct ReducedDLPTaylorCache
    pts::BoundaryPoints{Float64}
    X::Vector{Float64}
    Y::Vector{Float64}
    dX::Vector{Float64}
    dY::Vector{Float64}
    kappa::Vector{Float64}
    kress::Vector{Float64}
end

@experimental function _reduced_dlp_taylor_cache(pts::BoundaryPoints{Float64})
    # Cache full-boundary node data and the first column of the full-grid Kress matrix.
    # But the storage is reduced from O(N^2) to O(N) by keeping only the first column of the Kress matrix unlike
    # the BilliardGeometry.jl version to keep storage low
    #TODO Move this one to BilliardGeometry.jl to replace the old one!
    N = length(pts)
    X = getindex.(pts.xy, 1); Y = getindex.(pts.xy, 2)
    dX = getindex.(pts.tangent, 1); dY = getindex.(pts.tangent, 2)
    ddX = getindex.(pts.tangent_2, 1); ddY = getindex.(pts.tangent_2, 2)
    kappa = @. -(dX*ddY-dY*ddX)/(2*Float64(pi)*(dX*dX+dY*dY))
    # Use N full-boundary nodes, including the even-grid Nyquist correction.
    a = zeros(ComplexF64, N); n = (N-1)÷2; twopi = Float64(2*pi)
    @inbounds for m in 1:n # See MPSpack from Alex Barnett
        a[m+1] = 1/m; a[N-m+1] = 1/m
    end
    rjn = real(FFTW.ifft(a)); kress = similar(rjn)
    if iseven(N)
        alt = (-1).^(0:N-1) # Nyquist term
        @. kress = -twopi*rjn - (2*twopi/(N^2))*alt
    else
        @. kress = -twopi*rjn
    end
    return ReducedDLPTaylorCache(pts, X, Y, dX, dY, kappa, kress)
end

# Generate the Taylor coefficients for the DLP kernel between points i and j, this is always used for boundary - boundary interactions.
#TODO Make this one from scratch, this is just a temp fix
@experimental @inline function _kernel_taylor!(cache::ReducedDLPTaylorCache, w::TaylorWorkspace, k::Float64, p::Int, i::Int, j::Int)::Vector{ComplexF64}
    a = w.a; fill!(a, 0); pts = cache.pts
    if i == j
        a[1] = ComplexF64(pts.ws[i]*cache.kappa[i])
        return a
    end
    # Replay the existing triangular geometry arithmetic for identical values.
    lo = min(i,j); hi = max(i,j)
    dx = cache.X[lo]-cache.X[hi]; dy = cache.Y[lo]-cache.Y[hi]
    r = hypot(dx,dy); invr = inv(r); r2 = r*r
    inn = i<j ? cache.dY[j]*dx-cache.dX[j]*dy : -cache.dY[j]*dx+cache.dX[j]*dy
    dt = pts.ts[lo]-pts.ts[hi]; lt = log(4*sin(dt/2)^2)
    R = cache.kress[mod(i-j,length(pts))+1]; z = k*r; invtwopi = inv(2π)
    h0 = _bim_hankelh1(0,z); h1 = _bim_hankelh1(1,z)
    radial1_taylor!(w.H1,k,r2,p,k*h1,k*r*h0)
    wj=pts.ws[j]
    q=muladd(-wj,lt,R) # see above function
    dlp_geom_factor=0.5*inn*invr*wj
    dlp_L1_factor= -invtwopi*inn*invr*q
    @inbounds @simd for l = 1:p+1
        ### for documentation ###
        #l1 = -invtwopi*inn*invr*real(w.H1[l]) # L1 = -(inn*invr)/(2π) * j1_k.
        #l2 = im/2*inn*invr*w.H1[l] - l1*lt # L2 = (im/2)*inn*invr*h1_k - lt*L1.
        #a[l] = R*l1 + pts.ws[j]*l2 # DLP = R*L1 + w_j*L2
        #########################
        h1_k=w.H1[l];j1_k=real(h1_k) # h1_k and j1_k already include the factor k.
        a[l] = ComplexF64(muladd(-dlp_geom_factor,imag(h1_k),dlp_L1_factor*j1_k),dlp_geom_factor*j1_k)
    end
    return a
end

"""
    _taylor_cache(solver::Union{DoubleLayerPotentialSolver,CombinedFieldIntegralEquationSolver}, pts)

Construct the geometry-dependent cache for analytic Taylor evaluation of a
single-component BIM Fredholm matrix. The Kress matrix and geometry cache are
formed once and reused for every source-target pair.

## Arguments
- `solver::Union{DoubleLayerPotentialSolver,CombinedFieldIntegralEquationSolver}`:
  Single-component DLP or CFIE solver.
- `pts`: Boundary discretization data used by the solver.

## Returns
- `NamedTuple`: Cache with fields `solver`, `pts`, `Rmat`, `G`, and `N`,
  containing the solver, boundary data, Kress matrix, geometry cache, and
  boundary matrix dimension.
"""
function _taylor_cache(solver::Union{DoubleLayerPotentialSolver,CombinedFieldIntegralEquationSolver}, pts)
    N = length(pts.xy)
    G = boundary_geom_cache(pts,_is_nontrivial_dlp_grading(pts))
    Rmat = zeros(Float64,N,N); kress_R!(Rmat)
    return (;solver,pts,Rmat,G,N)
end

"""
    _taylor_cache(solver::CompositeBIMSolver, pts)

Construct component-local geometry data for analytic Taylor evaluation of a
composite BIM Fredholm matrix.

For each physical component `a`, the cache contains its local boundary points,
Kress matrix, and geometry cache. `g2c[i]` and `g2l[i]` map a global boundary
index to its component and local index, allowing same-component and smooth
cross-component interactions to be selected without rebuilding geometry.

## Arguments
- `solver::CompositeBIMSolver`: Composite solver containing one DLP or CFIE
  component solver for each physical boundary component.
- `pts`: Globally merged composite boundary discretization.

## Returns
- `NamedTuple`: Composite Taylor cache containing `solver`, `pts`,
  component-local boundary data `comp_pts`, Kress matrices `Rmats`, geometry
  caches `Gs`, global-to-component maps `g2c` and `g2l`, and global matrix
  dimension `N`.
"""
function _taylor_cache(solver::CompositeBIMSolver, pts)
    nc = length(solver.loop_solver); offs = _composite_offsets(pts,nc)
    comp_pts = ntuple(a -> _composite_component_slice(pts,offs[a]:offs[a+1]-1,a),nc)
    Gs = ntuple(a -> boundary_geom_cache(comp_pts[a],_is_nontrivial_dlp_grading(comp_pts[a])),nc)
    Rmats = ntuple(nc) do a
        Na = length(comp_pts[a].xy)
        Rmat = zeros(Float64,Na,Na); kress_R!(Rmat); Rmat
    end
    g2c,g2l = _composite_global_to_local(offs); N = length(pts.xy)
    return (;solver,pts,comp_pts,Rmats,Gs,g2c,g2l,N)
end

"""
    _kernel_taylor!(cache, w::TaylorWorkspace, k::Float64, p::Int, i::Int, j::Int) -> Vector{ComplexF64}

Dispatch the evaluation of one raw BIM kernel Taylor series through the solver
stored in `cache`. The cache contains all geometry-dependent quantities needed
by the corresponding single-component or composite implementation.

For the raw BIM operator

Kᵢⱼ(k+δ) = Σₙ₌₀ᵖ Kᵢⱼ,ₙδⁿ + O(δᵖ⁺¹),

the returned workspace vector satisfies

w.a[n+1] = Kᵢⱼ,ₙ = Kᵢⱼ⁽ⁿ⁾(k)/n!.

For a single-component DLP or CFIE solver, evaluation is forwarded to the
Kress-corrected same-component kernel using the cached `Rmat` and geometry
data `G`. For a [`CompositeBIMSolver`](@ref), the global indices are first
mapped to their physical components and local indices, after which either the
same-component Kress kernel or the smooth cross-component kernel is selected.

This method returns coefficients of the raw operator `K(k)` rather than the
Fredholm operator `A(k)=I-K(k)`.

## Arguments
- `cache`: Single-component or composite Taylor cache returned by
  [`_taylor_cache`](@ref).
- `w::TaylorWorkspace`: Reusable Taylor workspace.
- `k::Float64`: Taylor expansion wavenumber.
- `p::Int`: Maximum Taylor degree.
- `i::Int`: Global target index.
- `j::Int`: Global source index.

## Returns
- `Vector{ComplexF64}`: `w.a`, containing normalized Taylor coefficients of
  the selected raw BIM kernel entry through degree `p`.
"""
@inline function _kernel_taylor!(cache, w::TaylorWorkspace, k::Float64, p::Int, i::Int, j::Int)::Vector{ComplexF64}
    return _kernel_taylor!(cache.solver,w,cache,cache.pts,k,p,i,j)
end

"""
    _kernel_taylor!(solver::Union{DoubleLayerPotentialSolver,CombinedFieldIntegralEquationSolver}, w::TaylorWorkspace, cache, pts, k::Float64, p::Int, i::Int, j::Int) -> Vector{ComplexF64}

Evaluate one raw single-component BIM kernel Taylor series using precomputed
geometry data.

## Arguments
- `solver::Union{DoubleLayerPotentialSolver,CombinedFieldIntegralEquationSolver}`:
  Single-component DLP or CFIE solver selecting the analytic kernel by
  multiple dispatch.
- `w::TaylorWorkspace`: Reusable Taylor workspace.
- `cache`: Taylor cache containing `Rmat` and `G`.
- `pts`: Boundary discretization data.
- `k::Float64`: Taylor expansion wavenumber.
- `p::Int`: Maximum Taylor degree.
- `i::Int`: Target boundary index.
- `j::Int`: Source boundary index.

## Returns
- `Vector{ComplexF64}`: `w.a`, containing normalized Taylor coefficients of
  the raw DLP or CFIE kernel entry through degree `p`.
"""
@inline function _kernel_taylor!(solver::Union{DoubleLayerPotentialSolver,CombinedFieldIntegralEquationSolver}, w::TaylorWorkspace, cache, pts, k::Float64, p::Int, i::Int, j::Int)::Vector{ComplexF64}
    return _kernel_taylor!(solver,w,pts,cache.Rmat,cache.G,k,p,i,j)
end

"""
    _kernel_taylor!(solver::CompositeBIMSolver, w::TaylorWorkspace, cache, pts, k::Float64, p::Int, gi::Int, gj::Int) -> Vector{ComplexF64}

Evaluate normalized Taylor coefficients of one global composite BIM kernel
entry. The global target and source indices are mapped to component-local
indices using `g2c` and `g2l`.

For a same-component interaction, the component solver evaluates its
Kress-corrected DLP or CFIE kernel. For a cross-component interaction, the
interaction is smooth and the solver belonging to the source component
selects the corresponding direct DLP or CFIE Taylor kernel.

## Arguments
- `solver::CompositeBIMSolver`: Composite BIM solver containing the
  component-specific DLP/CFIE solvers.
- `w::TaylorWorkspace`: Reusable Taylor workspace.
- `cache`: Composite Taylor cache containing component-local boundary data,
  Kress matrices, geometry caches, and global-to-local maps.
- `pts`: Globally merged composite boundary discretization.
- `k::Float64`: Taylor expansion wavenumber.
- `p::Int`: Maximum Taylor degree.
- `gi::Int`: Global target boundary index.
- `gj::Int`: Global source boundary index.

## Returns
- `Vector{ComplexF64}`: `w.a`, containing normalized Taylor coefficients of
  the selected global raw composite BIM entry through degree `p`.
"""
@inline function _kernel_taylor!(solver::CompositeBIMSolver, w::TaylorWorkspace, cache, pts, k::Float64, p::Int, gi::Int, gj::Int)::Vector{ComplexF64}
    ca = cache.g2c[gi]; cb = cache.g2c[gj]; ia = cache.g2l[gi]; jb = cache.g2l[gj]
    ca == cb && return _kernel_taylor!(_loop_solver(solver,ca),w,cache.comp_pts[ca],cache.Rmats[ca],cache.Gs[ca],k,p,ia,jb)
    xi,yi = cache.comp_pts[ca].xy[ia]
    return _kernel_taylor!(_loop_solver(solver,cb),w,cache.comp_pts[cb],xi,yi,k,p,jb)
end

################################################################################
# FREDHOLM TAYLOR COEFFICIENTS
# The methods above return the raw BIM operator
#
#   K(k)=D(k)          for DLP,
#   K(k)=D(k)+ikS(k)   for CFIE.
#
# All supported eigenproblems have Fredholm form
#
# A(k)=I-K(k).
#
# Hence, if
#
# Kᵢⱼ(k+δ)=Σₙ₌₀ᵖKᵢⱼ,ₙδⁿ,
#
# then
#
#   Aᵢⱼ,₀ = δᵢⱼ-Kᵢⱼ,₀,
#   Aᵢⱼ,ₙ = -Kᵢⱼ,ₙ,    n≥1.
#
# `_fredholm_taylor!` is therefore the final analytic BIM interface.
################################################################################

"""
    _fredholm_taylor!(cache, w::TaylorWorkspace, k::Float64, p::Int, i::Int, j::Int) -> Vector{ComplexF64}

# Compute normalized Taylor coefficients of one Fredholm-matrix entry.
## Arguments
- `cache`: Single-component or composite Taylor cache.
- `w::TaylorWorkspace`: Reusable Taylor workspace.
- `k::Float64`: Taylor expansion wavenumber.
- `p::Int`: Maximum Taylor degree.
- `i::Int`: Target matrix index.
- `j::Int`: Source matrix index.

## Returns
- `Vector{ComplexF64}`: `w.a`, containing the normalized Fredholm Taylor
  coefficients `Aᵢⱼ⁽ⁿ⁾(k)/n!` for `n=0,...,p` for `_horner` eval.
"""
@inline function _fredholm_taylor!(cache, w::TaylorWorkspace, k::Float64, p::Int, i::Int, j::Int)::Vector{ComplexF64}
    # Aᵢⱼ(k+δ) = Σₙ₌₀ᵖ Aᵢⱼ,ₙδⁿ + O(δᵖ⁺¹),
    # Aᵢⱼ,ₙ = Aᵢⱼ⁽ⁿ⁾(k)/n!.
    # Since `A(k)=I-K(k)`, these are transformed in place as
    # Aᵢⱼ,₀ = δᵢⱼ-Kᵢⱼ,₀,
    # Aᵢⱼ,ₙ = -Kᵢⱼ,ₙ,    n≥1.
    a = _kernel_taylor!(cache,w,k,p,i,j)
    @inbounds @simd for l = 1:p+1
        a[l] = -a[l]
    end
    i == j && (a[1] += 1)
    return a
end

######## BEYN HALF - CONTOUR (DLP ONLY) #########

# cast the user supplied beta into a Float64 (Taylor recurr precision, no exception!). Tipically should be
# handled already by the high level solver API, but let's just be paranoid about it.
@inline _real_beta_coupling(beta::Number)=Float64(beta)

"""
     _fredholm_real_beta_taylor!(cache, w::TaylorWorkspace, k::Float64, p::Int, i::Int, j::Int, beta::Real) -> Vector{ComplexF64}

Compute normalized Taylor coefficients of one Fredholm-matrix entry,

    Aᵢⱼ(k+δ) = Σₙ₌₀ᵖ Aᵢⱼ,ₙδⁿ + O(δᵖ⁺¹),
    Aᵢⱼ,ₙ = Aᵢⱼ⁽ⁿ⁾(k)/n!.

This works only when `cache.solver isa DoubleLayerPotentialSolver`, because for default CFIE the im * k 
contrivution breaks the complex conjugation symmetry.
First [`_kernel_taylor!`](@ref) evaluates the corresponding raw BIM kernel
coefficients (DLP in this case) via the reccurence below (since `A(k)=I-K(k)`)

    Aᵢⱼ,₀ = δᵢⱼ-Kᵢⱼ,₀,
    Aᵢⱼ,ₙ = -Kᵢⱼ,ₙ,    n≥1.

Thus the returned workspace vector satisfies

    w.a[n+1] = Aᵢⱼ⁽ⁿ⁾(k)/n!.

This approach efficiently works because taking the normal derivative of the 
Green function is linear (and integration is linear) and therefore the Taylor 
expansion commutes with the normal derivative - so basically same eval complexity as the OG kernel.

## Arguments
- `cache`: Single-component or composite Taylor cache (DLP only).
- `w::TaylorWorkspace`: Reusable Taylor workspace.
- `k::Float64`: Taylor expansion wavenumber.
- `p::Int`: Maximum Taylor degree.
- `i::Int`: Target matrix index.
- `j::Int`: Source matrix index.

## Returns
- `Vector{ComplexF64}`: `w.a`, containing the normalized Fredholm Taylor
  coefficients `Aᵢⱼ⁽ⁿ⁾(k)/n!` for `n=0,...,p` for `_horner` eval.
"""
@inline function _fredholm_real_beta_taylor!(cache, w::TaylorWorkspace, k::Float64, p::Int, i::Int, j::Int, beta::Real)
    cache.solver isa DoubleLayerPotentialSolver || throw(ArgumentError("Real-beta coefficients require Euclidean DLP"))
    coupling = _real_beta_coupling(beta)
    a = _fredholm_taylor!(cache, w, k, p, i, j)
    @inbounds @simd for ell = 1:p+1
        a[ell] = ComplexF64(real(a[ell])-coupling*imag(a[ell]),0.0)
    end
    return a
end

#####################################################
################## HIGH LEVEL API ###################
#####################################################

######################################
# MULTI-WAVENUMBER TAYLOR CONSTRUCTION
######################################

"""
    _construct_matrices_multi_k_taylor_full(solver, pts, k0, ks, p;
        multithreaded::Bool=true) -> Vector{Matrix{ComplexF64}}

Construct full-boundary Fredholm matrices from one degree-p Taylor expansion,

    A(k0+δ) ≈ Σ_{l=0}^p A_l*δ^l,
    A_l = A^(l)(k0)/l!.

## Arguments
- `solver::SweepBIMSolver`: BIM solver with `symmetry===nothing`.
- `pts::BoundaryPoints{Float64}`: Full-boundary discretization.
- `k0::Float64`: Positive real Taylor center.
- `ks::AbstractVector{<:Number}`: Requested real or complex wavenumbers.
- `p::Int`: Maximum Taylor degree.

## Keywords
- `multithreaded::Bool=true`: Parallelize over source columns.

## Returns
- `Vector{Matrix{ComplexF64}}`: One N×N Fredholm matrix per wavenumber,
  in the order of `ks`, where `N=length(pts)`.
"""
function _construct_matrices_multi_k_taylor_full(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, ks::AbstractVector{<:Number}, p::Int; multithreaded::Bool = true)::Vector{Matrix{ComplexF64}}
    solver.symmetry === nothing || throw(ArgumentError("full Taylor construction requires solver.symmetry === nothing"))
    cache = _taylor_cache(solver, pts) # all the neccesery geometricial data for Taylor coefficient consutrction
    N::Int = cache.N # number of boundary points (full boundary is the fundamental domain)
    nk::Int = length(ks) 
    δ::Vector{ComplexF64} = ComplexF64.(ks .- k0) # all the offsets from the central wavenumber k0 to be computed for
    Tbufs::Vector{Matrix{ComplexF64}} = [Matrix{ComplexF64}(undef, N, N) for _ = 1:nk] # one output matrix per k where threads fill disjoint source columns.
    work::Vector{TaylorWorkspace} = [TaylorWorkspace(p) for _ = 1:Threads.maxthreadid()] # same for all, just in case of race conditions in the future each gets its own same copy
    @use_threads multithreading=multithreaded for j = 1:N
        # Work on column j of the kernel matrix 
        #  [ pt1-pt1 pt1-pt2 ... pt1-ptj ... pt1-pm;
        #    pt2-pt1 pt2-pt2 ... pt2-ptj ... pt2-pm;
        #    pt3-pt1 pt3-pt2 ... pt3-ptj ... pt3-pm;
        #     ...
        #    ptm-pt1 ptm-pt2 ... ptm-ptj ... ptm-pm; ]
        w::TaylorWorkspace = work[Threads.threadid()]
        @inbounds for i = 1:N # for each row i of column j above
            a::Vector{ComplexF64} = _fredholm_taylor!(cache, w, k0, p, i, j) # get its Taylor coefficients for the (i,j) entry of the kernel matrix
            for q = 1:nk
                Tbufs[q][i, j] = _horner(a, δ[q])
            end
        end
        # _fredholm_taylor! already includes I-K; no extra identity is added here.
        # The reduced builder folds raw kernel coefficients, then adds I once.
    end
    return Tbufs
end

"""
    _construct_matrices_multi_k_taylor_reduced(solver, pts, k0, ks, p;
        multithreaded::Bool=true) -> Vector{Matrix{ComplexF64}}

Construct symmetry-reduced Fredholm matrices by summing method of images's
kernels's Taylor coefficients,

    A_red,ab,l = δ_l0*δ_ab - Σ_{j∈O_b} phase[j]*K_fund[a],j,l.

## Arguments
- `solver::SweepBIMSolver`: BIM solver with active symmetry reduction.
- `pts::BoundaryPoints{Float64}`: Full-boundary discretization.
- `k0::Float64`: Positive real Taylor center.
- `ks::AbstractVector{<:Number}`: Requested real or complex wavenumbers.
- `p::Int`: Maximum Taylor degree.

## Keywords
- `multithreaded::Bool=true`: Parallelize over reduced source columns.

## Returns
- `Vector{Matrix{ComplexF64}}`: One m×m reduced Fredholm matrix per
  wavenumber, where m is the number of fundamental boundary representatives.
"""
function _construct_matrices_multi_k_taylor_reduced(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, ks::AbstractVector{<:Number}, p::Int; multithreaded::Bool = true)::Vector{Matrix{ComplexF64}}
    solver.symmetry === nothing && throw(ArgumentError("reduced Taylor construction requires an active symmetry"))
    compact = solver isa DoubleLayerPotentialSolver && solver.grading isa SmoothPeriodicGrading
    cache = compact ? _reduced_dlp_taylor_cache(pts) : _taylor_cache(solver, pts)
    orbits = solver isa CompositeBIMSolver ? _composite_symmetry_orbits(Float64, solver, pts) : _fold_boundary(Float64, solver.billiard, length(pts.xy), solver.symmetry, solver.sym_characters) # compute the symmetry orbits of the boundary points - this is the symmetry reduction via modular arithmetic. See BilliardGeometry.jl for more details on implementation. orbits contains the fundamental indices, orbit membership, and phase factors for each boundary point such that irrep factors and full boundary can be constructed.
    m::Int = fundamental_size(orbits) # size of the fundamental domain of the boundary points
    fund = orbits.fundamental_indices # fundamental indices of the boundary points
    orbit_of = orbits.orbit_of # orbit membership of each boundary point
    phase = orbits.phase # phase factors for each boundary point
    images::Vector{Vector{Int}} = [Int[] for _ = 1:m] # preallocate for each fundamental orbit its immage index array
    @inbounds for j = eachindex(orbit_of) # populate the image index array for each fundamental orbit
        push!(images[orbit_of[j]], j)
    end
    # example of a C_5 = 2pi/5 rotation with m = 3 fundamental points (boundary nodes on the fundamental domain)
    # each one is rotated by the corresponding symmetry operation (identity, R, R², R³, R⁴) which tesselate the entire full domain
    #                    identity   R    R²   R³   R⁴
    # images[1] = [         1,      4,    7,  10,  13 ]
    # images[2] = [         2,      5,    8,  11,  14 ]
    # images[3] = [         3,      6,    9,  12,  15 ]
    nk::Int = length(ks) 
    δ::Vector{ComplexF64} = ComplexF64.(ks .- k0) # for each k find the distance from center k0
    #TODO Make a non-allocating version of this
    Tbufs::Vector{Matrix{ComplexF64}} = [Matrix{ComplexF64}(undef, m, m) for _ = 1:nk] # Make buffers for all wavenumbers. Their size is m x m since for each fundamental domain orbit (there are m of them) we need its interaction with all the other points ala source-target interactions, ofc for each wavenumber each own.
    # (*)   [ pt1-pt1 pt1-pt2 pt1-pt3 ... pt1-pm;
        #     pt2-pt1 pt2-pt2 pt2-pt3 ... pt2-pm;
        #     pt3-pt1 pt3-pt2 pt3-pt3 ... pt3-pm;
        #     ...
        #     ptm-pt1 ptm-pt2 ptm-pt3 ... ptm-pm; ]
    work::Vector{TaylorWorkspace} = [TaylorWorkspace(p) for _ = 1:Threads.maxthreadid()] # make sure each thread has its own workspace to avoid race conditions when reading, even if all the same
    coeff::Vector{Matrix{ComplexF64}} = [Matrix{ComplexF64}(undef, p + 1, m) for _ = 1:Threads.maxthreadid()] # same logic as work, each thread gets its own coefficient matrix to avoid race conditions
    # on each thread we will calculate the Fredholm matrix for the fundamental boundary points
    @use_threads multithreading=multithreaded for bcol = 1:m # fix pt(bcol) column from matrix (*)
        # (**) [pt1-pt(bcol);
        #       pt2-pt(bcol);
        #       pt3-pt(bcol);
        #       ...
        #       ptm-pt(bcol);]
        # a single fundamental boundary column / orbit
        tid::Int = Threads.threadid() # to make sure each thread uses its own workspace and coefficient matrix and oversubscription or race conditions dont cause issues
        w::TaylorWorkspace = work[tid] # get its workspace
        C::Matrix{ComplexF64} = coeff[tid] # get its coefficient matrix to populate
        # Now each column is the Taylor coefficients for the interaction gi-gj ()
        # (***) [a_0 (gi-gj);
        #        a_1 (gi-gj);
        #        a_2 (gi-gj);
        #        ...
        #        a_p (gi-gj);]
        imgs::Vector{Int} = images[bcol] # get the image indices for the current fundamental boundary column
        gj::Int = imgs[1] # the starting point (on the fundamental domain point)
        χ = phase[gj] # irrep phase of gj
        @inbounds for arow = 1:m # at fixed gj visit every fundamental target gi
            gi::Int = fund[arow] # a row (why called arow) in (**) that visits all points
            a::Vector{ComplexF64} = _kernel_taylor!(cache, w, k0, p, gi, gj) # the the Taylor coeffs for the kernel at matrix entry (gi, gj)
            @simd for l = 1:p + 1
                C[l, arow] = -χ * a[l] # populate (***)'s arow column with the Taylor coefficients for the interaction gi-gj
            end
        end
        # since gj was only the first point on the length(imgs)-length orbit, we now have to do the (length(imgs)-1) remaining points (this is how it is tipically done with method of images, K[gi,gj] -> K[gi,gj] - [χ K[gi,gj] for each image])
        # follows same story as above, find the gi, the irrep character of gj, and compute the Taylor coefficients for the interaction gi-gj
        @inbounds for ii = 2:length(imgs)
            gj = imgs[ii]
            χ = phase[gj]
            for arow = 1:m
                gi = fund[arow]
                a = _kernel_taylor!(cache, w, k0, p, gi, gj)
                # this deserves a comment, since as following comment above we subtract (***)'s gj column with all gi's image contributions
                @simd for l = 1:p + 1
                    C[l, arow] -= χ * a[l]
                end
            end
        end
        # finally done, now to transform the K kernels into a Fredholm matrix A = I + (-K), where -K is represented by the Taylor coefficients in C, waiting for the k input
        @inbounds for arow = 1:m
            arow == bcol && (C[1, arow] += 1.0) # this is the diagonal contribution for the identity part of  A = I + (-K)
            for q = 1:nk # foe each k now evaluate the Fredholm matrix entry using the Taylor coefficients
                Tbufs[q][arow, bcol] = _horner(C, arow, δ[q])
            end
        end
    end
    return Tbufs
end

"""
    _construct_matrices_multi_k_taylor(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, ks::AbstractVector{<:Number}, p::Int; multithreaded::Bool=true)::Vector{Matrix{ComplexF64}}

Construct Fredholm matrices at multiple wavenumbers using a single analytic
Taylor expansion about `k0`,

    A(k0 + δ) ≈ Σₗ₌₀ᵖ Aₗ δˡ,    Aₗ = A⁽ˡ⁾(k0) / l!.

For each requested wavenumber `ks[q]`, the Taylor polynomial is evaluated at
`δ_q = ks[q] - k0` by Horner's rule. Both real and complex wavenumbers are
supported. The optimized full-boundary or symmetry-reduced construction is
selected automatically.

## Arguments
- `solver::SweepBIMSolver`: BIM solver defining the Fredholm operator.
- `pts::BoundaryPoints{Float64}`: Full-boundary discretization.
- `k0::Float64`: Center of the shared Taylor expansion.
- `ks::AbstractVector{<:Number}`: Wavenumbers at which the matrices are evaluated.
- `p::Int`: Taylor expansion degree.
- `multithreaded::Bool=true`: Enable multithreaded matrix construction.

## Returns
- `::Vector{Matrix{ComplexF64}}`: Fredholm matrices evaluated at the requested wavenumbers.
"""
function _construct_matrices_multi_k_taylor(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, ks::AbstractVector{<:Number}, p::Int; multithreaded::Bool = true)::Vector{Matrix{ComplexF64}}
    if solver.symmetry === nothing
        return _construct_matrices_multi_k_taylor_full(solver, pts, k0, ks, p; multithreaded)
    end
    return _construct_matrices_multi_k_taylor_reduced(solver, pts, k0, ks, p; multithreaded)
end

####################################
######## HALF BEYN CONTOUR #########
####################################

# At real k0, form real-beta coefficients as Re(a)-beta*Im(a)
# IMPORTANT: Unlike other vers, this one needs in the end the target-normal approach, since the original argument ordering and kernel evaluation are based on the target-normal idea to exclude suprious u(s) -> Ψ contributions. Therefore in the end because of the default source normal convetion a transpose is needed to get target-normal ordering.
#TODO test stability of the claim below
# NOTE: β=1 is chosen as default since it seems that Beyn behaves more nicely when the artifical poles are pushed further away from their real sol partner (using β=0 or incrediably small likes to miss levels).


@experimental function _construct_matrices_multi_k_taylor_real_beta_full(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints{Float64}, k0::Float64, ks::AbstractVector{<:Number}, p::Int; beta::Real=1.0, multithreaded::Bool = true, coefficients=nothing)::Vector{Matrix{ComplexF64}}
    solver.symmetry === nothing || throw(ArgumentError("full real-beta Taylor construction requires solver.symmetry === nothing"))
    coupling = _real_beta_coupling(beta)
    cache = _taylor_cache(solver, pts)
    N::Int = cache.N
    nk::Int = length(ks)
    δ::Vector{ComplexF64} = ComplexF64.(ks .- k0)
    Tbufs::Vector{Matrix{ComplexF64}} = [Matrix{ComplexF64}(undef, N, N) for _ = 1:nk] # for each k in ks construt thr buffers for the A matrices
    work::Vector{TaylorWorkspace} = [TaylorWorkspace(p) for _ = 1:Threads.maxthreadid()]
    @use_threads multithreading=multithreaded for j = 1:N # for each source point j
        w::TaylorWorkspace = work[Threads.threadid()]
        @inbounds for i = 1:N # loop over target points for the fixed source point j
            if coefficients===nothing
                a = _fredholm_real_beta_taylor!(cache, w, k0, p, i, j, coupling)
            else
                # Retain the full coefficients before converting to real beta.
                # The same expansion then validates this window without reassembly.
                a = _fredholm_taylor!(cache, w, k0, p, i, j)
                @simd for l = 1:p+1
                    coefficients[i,j,l] = a[l]
                    a[l] = ComplexF64(real(a[l])-coupling*imag(a[l]), 0.0)
                end
            end
            for q = 1:nk # evaluate the Taylor series at δ[q] for the (i,j)-th entry as T_{ij}(k0 + δ[q]) = sum_{l=0}^{p} (a_{ij}[l+1]) * (δ[q])^l with a from above
                Tbufs[q][i, j] = _horner(a, δ[q])
            end
        end
    end
    return Tbufs
end

# see above for a more nices explanation as symmetries make everything more difficult to follow in the reduced setting
@experimental function _construct_matrices_multi_k_taylor_real_beta_reduced(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints{Float64}, k0::Float64, ks::AbstractVector{<:Number}, p::Int; beta::Real=1.0, multithreaded::Bool = true, coefficients=nothing)::Vector{Matrix{ComplexF64}}
    solver.symmetry === nothing && throw(ArgumentError("Real-beta Taylor construction requires an active symmetry"))
    coupling = _real_beta_coupling(beta)
    cache = solver.grading isa SmoothPeriodicGrading ? _reduced_dlp_taylor_cache(pts) : _taylor_cache(solver, pts)
    orbits = _fold_boundary(Float64, solver.billiard, length(pts.xy), solver.symmetry, solver.sym_characters)
    m::Int = fundamental_size(orbits)
    fund = orbits.fundamental_indices
    orbit_of = orbits.orbit_of
    phase = orbits.phase
    images::Vector{Vector{Int}} = [Int[] for _ = 1:m]
    @inbounds for j = eachindex(orbit_of)
        push!(images[orbit_of[j]], j)
    end
    nk::Int = length(ks)
    δ::Vector{ComplexF64} = ComplexF64.(ks .- k0)
    Tbufs::Vector{Matrix{ComplexF64}} = [Matrix{ComplexF64}(undef, m, m) for _ = 1:nk] # these are now reduced matrices of size (fund orbit, fund orbit) for each k in ks
    work::Vector{TaylorWorkspace} = [TaylorWorkspace(p) for _ = 1:Threads.maxthreadid()]
    coeff::Vector{Matrix{ComplexF64}} = [Matrix{ComplexF64}(undef, p + 1, m) for _ = 1:Threads.maxthreadid()]
    full_coeff = coefficients===nothing ? coeff : [Matrix{ComplexF64}(undef, p+1, m) for _ = 1:Threads.maxthreadid()]
    powers = Matrix{ComplexF64}(undef, p+1, nk)
    @inbounds for q = 1:nk
        powers[1,q] = 1.0
        for l = 1:p
            powers[l+1,q] = powers[l,q]*δ[q]
        end
    end
    evaluation = [Matrix{ComplexF64}(undef, m, nk) for _ = 1:Threads.maxthreadid()]
    @blas_multi 1 begin
        @use_threads multithreading=multithreaded for bcol = 1:m # for each reduced source column
            tid::Int = Threads.threadid()
            w::TaylorWorkspace = work[tid]
            C::Matrix{ComplexF64} = coeff[tid] # C[l,arow] stores degree taylor coefficient for the arow-th reduced target row
            H = full_coeff[tid]
            imgs::Vector{Int} = images[bcol]
            gj::Int = imgs[1] # finds the full boundary index corresponding to the bcol-th reduced source column
            χ = phase[gj] # irrep factor associated with gj
            ########
            # Initialize the column using its first source image.
            ########
            @inbounds for arow = 1:m # for each reduced target row
                gi::Int = fund[arow] # finds the full boundary index corresponding to the arow-th reduced target row
                a::Vector{ComplexF64} = _kernel_taylor!(cache, w, k0, p, gi, gj) # compute the Taylor coefficients for the kernel at the full boundary indices gi, gj
                @simd for l = 1:p + 1 # convert at real k0 before applying the possibly complex irrep phase χ
                    C[l, arow] = -χ * (real(a[l]) - coupling*imag(a[l])) # this source image contributes to I-K; its source-normal derivative and quadrature weight are already in a
                    coefficients===nothing || (H[l,arow] = -χ*a[l])
                end
            end
            ########
            # Add the remaining source images to the same reduced column; boundary quadrature weights are already included in a.
            ########
            @inbounds for ii = 2:length(imgs) # first one already done above
                gj = imgs[ii]
                χ = phase[gj]
                for arow = 1:m
                    gi = fund[arow]
                    a = _kernel_taylor!(cache, w, k0, p, gi, gj)
                    @simd for l = 1:p + 1
                        C[l, arow] -= χ * (real(a[l]) - coupling*imag(a[l]))
                        coefficients===nothing || (H[l,arow] -= χ*a[l])
                    end
                end
            end
            if coefficients!==nothing
                # Fold the full kernel separately: conversion to real beta must
                # precede the irrep phases in C, while H retains the full kernel.
                H[1,bcol] += 1.0
                @inbounds for l = 1:p+1, arow = 1:m
                    coefficients[arow,bcol,l] = H[l,arow]
                end
            end
            @inbounds for arow = 1:m
                # Add the identity (DLP 1/2 jump relation, mul by 2 convetion) once, after summing all source images.
                # It contributes only on the reduced diagonal and at Taylor degree zero.
                arow == bcol && (C[1, arow] += 1.0)
                # C[:,arow] now contains the Taylor coefficients of this one reduced
                # matrix entry, for the current source column bcol. The source-image
                # sum and diagonal identity have already been included.
                #
                # For each requested k=ks[q], evaluate
                # C[1,arow] + C[2,arow]*(k-k0) + ... + C[p+1,arow]*(k-k0)^p,
                # then store it at the same (arow,bcol) position in that k's matrix.
            end
            # Evaluate this source column at all nodes: E=transpose(C)*powers.
            # For each quadrature node q, copy E[:,q] into column bcol.
            E = evaluation[tid]
            mul!(E, transpose(C), powers)
            @inbounds for q = 1:nk
                @simd for arow = 1:m
                    Tbufs[q][arow,bcol] = E[arow,q]
                end
            end
        end
    end
    return Tbufs
end

"""
    beyn_interior_probe_taylor(kernel::DoubleLayerPotentialSolver, pts::BoundaryPoints{Float64}, sources, k0::Float64, p::Int; beta::Real=1.0, multithreaded::Bool=true) -> Matrix{ComplexF64}

Construct Taylor coefficients of the unweighted, symmetry-adapted
interior-source pt convention for the real Green function

    Gβ(x,s;k) = -(Y₀(k|x-s|) + beta*J₀(k|x-s|))/4.

For a symmetry group with characters χ, the probe is

    Vχ(x,s;k) = (2/|𝒢|) Σ_g χ(g) ∂ₙₓ Gβ(x,g*s;k).

(the 2 comes from the Id jump term convention).
The implementation evaluates the boundary-image sum using the `_fold_boundary`
orbit map. With no symmetry, this reduces to (as stated in paper)

    V(x,s;k) = 2∂ₙₓ Gβ(x,s;k).

# Arguments
- `kernel::DoubleLayerPotentialSolver`: Euclidean DLP solver defining the billiard and symmetry sector.
- `pts::BoundaryPoints{Float64}`: Physical-boundary discretization.
- `sources`: Indexable collection of real two-dimensional interior points,
  held fixed throughout the contour calculation. When symmetry is active,
  points may be chosen inside the fundamental interior. The caller must
  ensure that all sources lie strictly inside the physical billiard.
- `k0`: Finite, positive real Taylor expansion center.
- `p`: Taylor poly degree.

# Keywords
- `beta=1.0`: Finite real coupling multiplying the homogeneous J₀ contribution.
- `multithreaded=true`: Parallelize over interior-source columns.

# Returns
- `C::Matrix{ComplexF64}`: A matrix of size `(p+1, m*length(sources))` (for each source its Taylor coeffs), where
`m = boundary_matrix_size(kernel, pts)`. Its entries satisfy

    C[l+1, (j-1)*m+i] = Vχ[i,j]⁽l⁾(k0)/l!,  l = 0:p.

Evaluate an entry at complex wavenumber `k` using `_horner(C, (j-1)*m+i, k-k0)`.
"""
@experimental function beyn_interior_probe_taylor(kernel::DoubleLayerPotentialSolver, pts::BoundaryPoints{Float64}, sources, k0::Float64, p::Int; beta::Real=1.0, multithreaded::Bool=true)
    coupling = _real_beta_coupling(beta)
    N = length(pts)
    if kernel.symmetry === nothing
        images = reshape(collect(1:N), 1, N)
        phases = ones(ComplexF64, 1, N)
    else
        orbits = _fold_boundary(Float64, kernel.billiard, N, kernel.symmetry, kernel.sym_characters)
        images = orbits.fund_to_full
        phases = orbits.fund_to_scale
    end
    ng, m = size(images)
    # Remove ds from the smooth recurrence and normalize the image sum (due to how _kernel_taylor! gives ds-scaled Taylor coefficients).
    # Calling it with (s,x_i) gives ds_i*2∂_{n_xi}G_H(x_i,s).
    # 2 is already in a; only ds_i and the group average remain here.
    factors = Matrix{ComplexF64}(undef, ng, m)
    @inbounds @simd for q in eachindex(images)
        factors[q] = conj(phases[q]) / (ng * pts.ds[images[q]])
    end
    C = Matrix{ComplexF64}(undef, p+1, m*length(sources)) # allocate a matrix of Taylor coefficients for all sources (columns correspond to each source's contribution and it's rows are the Taylor coefficients)
    work = [TaylorWorkspace(p) for _ = 1:Threads.maxthreadid()]
    @use_threads multithreading=multithreaded for j = 1:length(sources) # for each source, compute its Taylor coefficients (col). Each source occupies a block of m columns where each is a fundamental boundary representative
        w = work[Threads.threadid()]
        sx, sy = Float64(sources[j][1]), Float64(sources[j][2]) # extract the x and y coordinates of the j-th source as Float64
        @inbounds for i = 1:m # for each orbit representative, compute its contribution to the Taylor coefficients (this is just annoying bookkeping)
            column = (j-1)*m + i # the column in C corresponding to the i-th orbit representative of the j-th source
            ii = images[1, i] # get the full boundary node corresponding to the i-th orbit representative
            a = _kernel_taylor!(kernel, w, pts, sx, sy, k0, p, ii) # compute the Taylor coefficients of the kernel at the full boundary node ii for the j-th source separation
            factor = factors[1, i] # get the precomputed factor for the first group and i-th orbit representative
            @simd for ell = 1:p+1
                C[ell, column] = factor * (real(a[ell]) - coupling*imag(a[ell]))
            end
            for g = 2:ng # for each image
                ii = images[g, i]
                a = _kernel_taylor!(kernel, w, pts, sx, sy, k0, p, ii) # compute the Taylor coefficients for the g-th group and i-th orbit representative
                factor = factors[g, i] # get the precomputed factor for the g-th group and i-th orbit representative - this is the (2 / (|𝒢| * ds)) in the Vχ(x,s;k) = (2/|𝒢|) Σ_g χ(g) ∂ₙₓ Gβ(x,g*s;k) (as 2 is already absorbed in _kernel_taylor!), ds is used to cancel the already ds scaled Taylor coeeficients of Gβ(x,g*s;k) coming from _kernel_taylor!
                # Add image g to the symmetry-adapted probe. 
                # GLOBAL PICTURE: The symmetry-adapted probe Vχ[i,j](k) is constructed by summing over all images g of the i-th orbit representative for the j-th source, each weighted by the precomputed factor and the kernel's Taylor coefficients:
                # Vχ[i,j](k) = (2/ng)*Σ_g conj(phases[g,i])*∂_{n_gi}Gβ(x_gi,s_j;k),
                # C[q+1,column] = (∂_k^q Vχ[i,j])(k0)/q!, q=0:p,
                # where x_gi=pts.xy[images[g,i]] and s_j=sources[j].
                @simd for ell = 1:p+1
                    C[ell, column] += factor * (real(a[ell]) - coupling*imag(a[ell]))
                end
            end
        end
    end
    return C
end

"""
    _beyn_taylor_residuals(kernel, pts, k0, ks, X, p; block_size::Int=512, block_bytes::Int=512*1024^2, multithreaded::Bool=true)

Compute full Hankel residuals for all candidate vectors in one window:

    δ_j = ks[j]-k0,    A_H,p(k) = Σ_{l=0}^p A_l*(k-k0)^l,
    residuals[j] = ‖A_H,p(ks[j])ᵀ*X[:,j]‖₂.

Taylor coefficients are reused from `coefficients` when supplied; otherwise
they are assembled once and streamed in source-column blocks.
Each block acts on all vectors using BLAS-3 Horner evaluation, without forming
a separate matrix at each candidate. The ordinary transpose matches the
target-normal response; no conjugation, reweighting or normalization is applied.

## Arguments
* `kernel::DoubleLayerPotentialSolver`: Full Hankel Euclidean DLP formulation.
* `pts::BoundaryPoints{Float64}`: Full physical boundary nodes and weights.
* `k0::Float64`: Positive real Taylor center used for the spectral window.
* `ks::AbstractVector`: Candidate wavenumbers in the window.
* `X::AbstractMatrix{ComplexF64}`: N×length(ks) projected vector columns, in
  full or reduced coordinates matching `kernel`.
* `p::Int`: Taylor degree; A_l=A_H^(l)(k0)/l! includes the jump only at l=0
  to get the full Taylor expansion up to degree `p`.

## Keyword Arguments
* `block_size::Int=512`: Preferred maximum number of source columns per block.
* `block_bytes::Int=512*1024^2`: Coefficient-buffer budget in bytes (512 MiB).
  A width-b block stores sizeof(ComplexF64)*N*b*(p+1) bytes. The width is
  reduced to fit, with a minimum of one column; other workspaces are separate.
* `multithreaded::Bool=true`: Enable threaded coefficient assembly.
* `coefficients::Union{Nothing,Array{ComplexF64,3}}=nothing`: Optional
  N×N×(p+1) full Fredholm coefficients retained from this window's assembly.
  Their storage is separate from `block_bytes`; supplying them skips assembly.

## Returns
* `residuals::Vector{Float64}`: Norms in candidate order, with the existing
  vector scaling and Taylor degree preserved. Empty for no candidates.
"""
@experimental function _beyn_taylor_residuals(kernel::DoubleLayerPotentialSolver, pts::BoundaryPoints{Float64}, k0::Float64, ks::AbstractVector, X::AbstractMatrix{ComplexF64}, p::Int; block_size::Int=512, block_bytes::Int=512*1024^2, multithreaded::Bool=true, coefficients=nothing)::Vector{Float64}
    # for innermost loops see _construct_matrices_multi_k_taylor_reduced and the Horner recurrence for the Taylor series, it is basically identical except that we are now also dealing with the need to matmul it with the gotten columns of X[:,j] for each candidate k in ks indexed by j.
    block_size>0 && block_bytes>0 || throw(ArgumentError("Coefficient block size and byte budget must be positive"))
    nr = length(ks); nr==0 && return Float64[]
    N = size(X, 1); delta = ComplexF64.(ks .- k0)
    # A supplied tensor already contains A_l from this contour's assembly.
    # Reuse it directly, including the original weights, phases and jump.
    cache = coefficients===nothing ? _taylor_cache(kernel, pts) : nothing
    orbits = coefficients===nothing && kernel.symmetry!==nothing ? _fold_boundary(Float64, kernel.billiard, length(pts), kernel.symmetry, kernel.sym_characters) : nothing
    images = [Int[] for _ in 1:N]
    if orbits!==nothing
        @inbounds for j in eachindex(orbits.orbit_of)
            push!(images[orbits.orbit_of[j]], j)
        end
    end
    # C stores A_l[:,J] for one block J and all degrees, so its storage is
    # sizeof(ComplexF64)*N*length(J)*(p+1).
    block = min(N, block_size, max(1, block_bytes÷(sizeof(ComplexF64)*N*(p+1)))) # determine the number of source columns of A to assemble in each block based on memory constraint kwarg
    C = coefficients===nothing ? Array{ComplexF64}(undef, N, block, p+1) : coefficients
    Y = Matrix{ComplexF64}(undef, N, nr) # residual vectors A_p(k_j)ᵀ*X[:,j], with dimensions (rows, candidates)
    work = coefficients===nothing ? [TaylorWorkspace(p) for _ in 1:Threads.maxthreadid()] : TaylorWorkspace[]
    coeff = orbits===nothing ? Matrix{ComplexF64}[] : [Matrix{ComplexF64}(undef, p+1, N) for _ in 1:Threads.maxthreadid()]
    for first in 1:block:N # partition source columns of A; these become output rows in Aᵀ*X
        last = min(first+block-1, N); width = last-first+1 # last source column and width of the current block
        if coefficients===nothing
            @use_threads multithreading=multithreaded for j in 1:width # for each column in the current block
                tid = Threads.threadid(); w = work[tid]
                bcol = first+j-1 # running boundary node index
                if orbits===nothing
                    # Each source column is assembled once. The Fredholm contr. func. already includes its quadrature weight and diagonal jump.
                    @inbounds for i in 1:N # for each node on the boundary
                        a = _fredholm_taylor!(cache, w, k0, p, i, bcol) # i is the target row; bcol is the source column
                        @simd for l in 1:p+1
                            C[i,j,l] = a[l]
                        end
                    end # this evaluaes for each bcol (defacto j index) all of its Taylor coefficients up to degree p -> this is the j-th column in the current block. Basically A[:,j] in the current block
                else
                    # Fold the source images exactly as in the reduced Taylor builder for the others. Everything else is same as for orbits==nothing just with the added orbit folding and phase factors
                    B = coeff[tid]; imgs = images[bcol]; gj = imgs[1]; phase = orbits.phase[gj]
                    @inbounds for i in 1:N
                        a = _kernel_taylor!(cache, w, k0, p, orbits.fundamental_indices[i], gj)
                        @simd for l in 1:p+1
                            B[l,i] = -phase*a[l]
                        end
                    end
                    @inbounds for image in 2:length(imgs)
                        gj = imgs[image]; phase = orbits.phase[gj]
                        for i in 1:N
                            a = _kernel_taylor!(cache, w, k0, p, orbits.fundamental_indices[i], gj)
                            @simd for l in 1:p+1
                                B[l,i] -= phase*a[l]
                            end
                        end
                    end
                    B[1,bcol] += 1.0
                    @inbounds for i in 1:N
                        @simd for l in 1:p+1
                            C[i,j,l] = B[l,i]
                        end
                    end
                end
            end
        end
        columns = coefficients===nothing ? (1:width) : (first:last)
        # Start with A_p[:,J]ᵀ*X, then multiply each column by delta[j]
        # and add A_l[:,J]ᵀ*X, for l=p-1,...,0. Store the result in Y[J,:].
        Z = @view Y[first:last,:] # view into the current block of rows in Y
        @blas_multi_then_1 MAX_BLAS_THREADS begin
            # Initialize Horner evaluation with the highest degree
            # Z[:,j] = A_p[:,J]ᵀ*X[:,j], simultaneously for all candidates.
            mul!(Z, transpose(@view(C[:,columns,p+1])), X) 
            # Array index l corresponds to Taylor degree l-1.
            # Each step computes Z[:,j] <- delta[j]*Z[:,j] + A_{l-1}[:,J]ᵀ*X[:,j], that is take Z[:,j] from the previous step, scale it by delta[j], and add the contribution from the next lower Taylor coefficient (the definition of the Horner scheme)
            for l in p:-1:1 
                @inbounds for j in 1:nr # j is the candidate index
                    @simd for i in first:last # for each row in the current block
                        Y[i,j] *= delta[j] # multiply each element in the current block by delta[j] 
                    end
                end # this is step delta[j]*Z[:,j] basically 
                # Now add the contribution from the next lower Taylor coefficient. That is, compute Z[:,j] <- Z[:,j] + A_{l-1}[:,J]ᵀ*X[:,j] for all j simultaneously as BLAS 3 
                mul!(Z, transpose(@view(C[:,columns,l])), X, 1.0, 1.0) # NOTE: this performs Z <- 1.0*Z + 1.0*(A_{l-1}[:,J]ᵀ*X) as required by the Horner scheme
            end
        end
    end
    return [norm(@view Y[:,j]) for j in 1:nr] # finally do residual norms for each candidate 
end

################################################################################
# BEYN CIRCULAR-CONTOUR INTERFACE
################################################################################

"""
    _construct_matrices_multi_k_taylor(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, R::Float64, nq::Int, p::Int; multithreaded::Bool=true)::Vector{Matrix{ComplexF64}}

Construct the Fredholm matrices on a circular Beyn contour using an analytic
Taylor expansion about `k0`,

    A(k0 + δ) ≈ Σₗ₌₀ᵖ Aₗ δˡ,    Aₗ = A⁽ˡ⁾(k0) / l!.

The contour wavenumbers are

    k_q = k0 + R exp(2πim(q-1)/nq),

and matrix construction is delegated to the generic multi-wavenumber Taylor
backend.

## Arguments
- `solver::SweepBIMSolver`: BIM solver defining the Fredholm operator.
- `pts::BoundaryPoints{Float64}`: Full-boundary discretization.
- `k0::Float64`: Center of the circular contour.
- `R::Float64`: Contour radius.
- `nq::Int`: Number of contour quadrature points.
- `p::Int`: Taylor expansion degree.
- `multithreaded::Bool=true`: Enable multithreaded matrix construction.

## Returns
- `::Vector{Matrix{ComplexF64}}`: Fredholm matrices evaluated at the `nq` contour points.
"""
function _construct_matrices_multi_k_taylor(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, R::Float64, nq::Int, p::Int; multithreaded::Bool = true)::Vector{Matrix{ComplexF64}}
    ks::Vector{ComplexF64} = ComplexF64.(k0 .+ R .* cis.(2π .* (0:nq - 1) ./ nq))
    return _construct_matrices_multi_k_taylor(solver, pts, k0, ks, p; multithreaded)
end

############################################
######## BEYN HALF NODE INTERFACE ##########
############################################

"""
    _construct_matrices_multi_k_taylor_real_beta(solver, pts, k0, ks, p; beta::Real=1.0, multithreaded::Bool=true)

Construct source-normal DLP Fredholm matrices for `G_beta=-(Y0+beta*J0)/4`
using one analytic Taylor expansion about the positive real center `k0`,

    A_beta(k0 + δ) ≈ Σₗ₌₀ᵖ A_beta,l δˡ,    A_beta,l = A_beta⁽ˡ⁾(k0) / l!.

The argument order and matrix return type match the original outgoing-Hankel
constructor. The distinct function name is the only entry into this path.
The full-boundary or symmetry-reduced builder is selected automatically.
Conversion precedes symmetry-character multiplication, including for complex
rotational irreps. Real and complex `ks` evaluate the same Taylor polynomial.

## Arguments
- `solver::DoubleLayerPotentialSolver`: Euclidean DLP solver.
- `pts::BoundaryPoints{Float64}`: Full-boundary discretization.
- `k0::Float64`: Positive real center of the shared Taylor expansion.
- `ks::AbstractVector{<:Number}`: Wavenumbers at which the matrices are evaluated.
- `p::Int`: Taylor expansion degree.
- `beta::Real=1.0`: Finite real coupling in `G_beta`.
- `multithreaded::Bool=true`: Enable multithreaded matrix construction.
- `coefficients::Union{Nothing,Array{ComplexF64,3}}=nothing`: Optional output
  buffer for full Hankel coefficients, indexed by target, source and degree+1.
  The returned real-beta matrices keep the original coefficient conversion.

## Returns
- `::Vector{Matrix{ComplexF64}}`: Fredholm matrices evaluated at the requested wavenumbers.
"""
function _construct_matrices_multi_k_taylor_real_beta(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints{Float64}, k0::Float64, ks::AbstractVector{<:Number}, p::Int; beta::Real=1.0, multithreaded::Bool = true, coefficients=nothing)::Vector{Matrix{ComplexF64}}
    if solver.symmetry === nothing
        return _construct_matrices_multi_k_taylor_real_beta_full(solver, pts, k0, ks, p; beta, multithreaded, coefficients)
    end
    return _construct_matrices_multi_k_taylor_real_beta_reduced(solver, pts, k0, ks, p; beta, multithreaded, coefficients)
end


"""
    _construct_matrices_multi_k_taylor_real_beta(solver, pts, k0, R, nq, p; beta::Real=1.0, multithreaded::Bool=true)

Construct real-beta DLP matrices on the original circular Beyn contour nodes

    k_q = k0 + R exp(2πim(q-1)/nq),

using the vector overload above. The finite real `beta` coupling is forwarded
unchanged. The original unshifted endpoint nodes are retained; no half-contour
pairing or change to Beyn's quadrature is performed here.

## Arguments
- `solver::DoubleLayerPotentialSolver`: Euclidean DLP solver.
- `pts::BoundaryPoints{Float64}`: Full-boundary discretization.
- `k0::Float64`: Positive real center of the circular contour.
- `R::Float64`: Contour radius.
- `nq::Int`: Number of contour quadrature points.
- `p::Int`: Taylor expansion degree.
- `beta::Real=1.0`: Finite real coupling in `G_beta`.
- `multithreaded::Bool=true`: Enable multithreaded matrix construction.

## Returns
- `::Vector{Matrix{ComplexF64}}`: Fredholm matrices evaluated at the `nq` contour points.
"""
function _construct_matrices_multi_k_taylor_real_beta(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints{Float64}, k0::Float64, R::Float64, nq::Int, p::Int; beta::Real=1.0, multithreaded::Bool = true)::Vector{Matrix{ComplexF64}}
    ks::Vector{ComplexF64} = ComplexF64.(k0 .+ R .* cis.(2π .* (0:nq - 1) ./ nq))
    return _construct_matrices_multi_k_taylor_real_beta(solver, pts, k0, ks, p; beta, multithreaded)
end

################################################################################
# MULTI-WAVENUMBER TAYLOR CONSTRUCTION WITH DERIVATIVES - EBIM
################################################################################

"""
    _construct_matrices_multi_k_taylor_derivatives_full(solver, pts, k0, ks, p;
        multithreaded::Bool=true)

Evaluate the full-boundary Fredholm Taylor polynomial and its first two
wavenumber derivatives,

    A(δ) = Σ_{l=0}^p A_l*δ^l,    δ=k-k0.

For each matrix entry, simultaneous Horner evaluation returns A, A' and A''.
## Arguments
- `solver::SweepBIMSolver`: BIM solver with `symmetry===nothing`.
- `pts::BoundaryPoints{Float64}`: Full-boundary discretization.
- `k0::Float64`: Positive real Taylor center.
- `ks::AbstractVector{<:Real}`: Requested real wavenumbers.
- `p::Int`: Maximum Taylor degree.

## Keywords
- `multithreaded::Bool=true`: Parallelize over source columns.

## Returns
- `Abufs::Vector{Matrix{ComplexF64}}`: N×N matrices A(ks[q]-k0).
- `dAbufs::Vector{Matrix{ComplexF64}}`: Their first polynomial derivatives.
- `ddAbufs::Vector{Matrix{ComplexF64}}`: Their second polynomial derivatives.

The three vectors are returned as `(Abufs, dAbufs, ddAbufs)`.
"""
function _construct_matrices_multi_k_taylor_derivatives_full(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, ks::AbstractVector{<:Real}, p::Int; multithreaded::Bool = true)
    # NOTE: For most of this function see _construct_matrices_multi_k_taylor_full since the extra derivate pathway already 
    # follows directly
    solver.symmetry === nothing || throw(ArgumentError("full Taylor construction requires solver.symmetry === nothing"))
    cache = _taylor_cache(solver, pts)
    N::Int = cache.N
    nk::Int = length(ks)
    δ::Vector{Float64} = Float64.(ks .- k0)
    # just like the non-derivative version, we preallocate buffers for each k and each derivative order
    Abufs::Vector{Matrix{ComplexF64}} = [Matrix{ComplexF64}(undef, N, N) for _ = 1:nk]
    dAbufs::Vector{Matrix{ComplexF64}} = [Matrix{ComplexF64}(undef, N, N) for _ = 1:nk]
    ddAbufs::Vector{Matrix{ComplexF64}} = [Matrix{ComplexF64}(undef, N, N) for _ = 1:nk]
    work::Vector{TaylorWorkspace} = [TaylorWorkspace(p) for _ = 1:Threads.maxthreadid()] # preallocate a workspace for each thread like the non-derivative version
    @use_threads multithreading=multithreaded for j = 1:N
        w::TaylorWorkspace = work[Threads.threadid()]
        @inbounds for i = 1:N
            a::Vector{ComplexF64} = _fredholm_taylor!(cache, w, k0, p, i, j) # get the Taylor coefficients for the (i,j) entry of the kernel matrix just like in the non-derivative version
            for q = 1:nk
                v, dv, ddv = _horner_with_12_derivatives(a, δ[q]) # here is the difference, we also compute the first and second derivatives with respect to k, simply by differentiation of the Taylor series polynomial under the hood (see _horner_with_12_derivatives)
                Abufs[q][i, j] = v
                dAbufs[q][i, j] = dv
                ddAbufs[q][i, j] = ddv
            end
        end
    end
    return Abufs, dAbufs, ddAbufs
end

"""
    _construct_matrices_multi_k_taylor_derivatives_reduced(solver, pts, k0, ks, p;
        multithreaded::Bool=true)

Evaluate the symmetry-reduced Fredholm matrix and its first two
wavenumber derivatives from the symmetry-reduced Fredholm Taylor polynomial
at the wavenumbers `ks`.

    A_red,ab,l = δ_l0*δ_ab - Σ_{j∈O_b} phase[j]*K_fund[a],j,l.

Source-image coefficients are combined with their symmetry phases, and the
reduced identity is added once at degree zero. Simultaneous Horner evaluation
then returns the polynomial and its derivatives at each requested wavenumber.

## Arguments
- `solver::SweepBIMSolver`: BIM solver with active symmetry reduction.
- `pts::BoundaryPoints{Float64}`: Full-boundary discretization.
- `k0::Float64`: Positive real Taylor center.
- `ks::AbstractVector{<:Real}`: Requested real wavenumbers.
- `p::Int`: Maximum Taylor degree.

## Keywords
- `multithreaded::Bool=true`: Parallelize over reduced source columns.

## Returns
- `Abufs::Vector{Matrix{ComplexF64}}`: m×m reduced polynomial matrices.
- `dAbufs::Vector{Matrix{ComplexF64}}`: Their first polynomial derivatives.
- `ddAbufs::Vector{Matrix{ComplexF64}}`: Their second polynomial derivatives.

The three vectors are returned as `(Abufs, dAbufs, ddAbufs)`, in the order
of `ks`; m is the number of fundamental boundary representatives.
"""
function _construct_matrices_multi_k_taylor_derivatives_reduced(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, ks::AbstractVector{<:Real}, p::Int; multithreaded::Bool = true)
    # see _construct_matrices_multi_k_taylor_reduced for the non-derivative version and the idea behind the implementation.
    # The idea behind it is that we generate the exact same Taylor coeffs as _construct_matrices_multi_k_taylor_reduced (which is why it is basically identical except for the extra derivative buffers) and then just do polynomial differentiation on the final method of images symmetrized fredholm matrix!
    solver.symmetry === nothing && throw(ArgumentError("reduced Taylor construction requires an active symmetry"))
    cache = _taylor_cache(solver, pts)
    orbits = solver isa CompositeBIMSolver ? _composite_symmetry_orbits(Float64, solver, pts) : _fold_boundary(Float64, solver.billiard, length(pts.xy), solver.symmetry, solver.sym_characters)
    m::Int = fundamental_size(orbits)
    fund = orbits.fundamental_indices
    orbit_of = orbits.orbit_of
    phase = orbits.phase
    images::Vector{Vector{Int}} = [Int[] for _ = 1:m]
    @inbounds for j = eachindex(orbit_of)
        push!(images[orbit_of[j]], j)
    end
    nk::Int = length(ks)
    δ::Vector{Float64} = Float64.(ks .- k0)
    Abufs::Vector{Matrix{ComplexF64}} = [Matrix{ComplexF64}(undef, m, m) for _ = 1:nk]
    dAbufs::Vector{Matrix{ComplexF64}} = [Matrix{ComplexF64}(undef, m, m) for _ = 1:nk]
    ddAbufs::Vector{Matrix{ComplexF64}} = [Matrix{ComplexF64}(undef, m, m) for _ = 1:nk]
    work::Vector{TaylorWorkspace} = [TaylorWorkspace(p) for _ = 1:Threads.maxthreadid()]
    coeff::Vector{Matrix{ComplexF64}} = [Matrix{ComplexF64}(undef, p + 1, m) for _ = 1:Threads.maxthreadid()]
    @use_threads multithreading=multithreaded for bcol = 1:m
        tid::Int = Threads.threadid()
        w::TaylorWorkspace = work[tid]
        C::Matrix{ComplexF64} = coeff[tid]
        imgs::Vector{Int} = images[bcol]
        gj::Int = imgs[1]
        χ = phase[gj]
        @inbounds for arow = 1:m
            gi::Int = fund[arow]
            a::Vector{ComplexF64} = _kernel_taylor!(cache, w, k0, p, gi, gj)
            @simd for l = 1:p + 1
                C[l, arow] = -χ * a[l]
            end
        end
        @inbounds for ii = 2:length(imgs)
            gj = imgs[ii]
            χ = phase[gj]
            for arow = 1:m
                gi = fund[arow]
                a = _kernel_taylor!(cache, w, k0, p, gi, gj)
                @simd for l = 1:p + 1
                    C[l, arow] -= χ * a[l]
                end
            end
        end
        @inbounds for arow = 1:m
            arow == bcol && (C[1, arow] += 1.0)
            for q = 1:nk
                v, dv, ddv = _horner_with_12_derivatives(C, arow, δ[q])
                Abufs[q][arow, bcol] = v
                dAbufs[q][arow, bcol] = dv
                ddAbufs[q][arow, bcol] = ddv
            end
        end
    end
    return Abufs, dAbufs, ddAbufs
end

"""
    _construct_matrices_multi_k_taylor_derivatives(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, ks::AbstractVector{<:Real}, p::Int; multithreaded::Bool=true)

Construct the Fredholm matrix and its first two wavenumber derivatives at
multiple real wavenumbers using a single analytic Taylor expansion about `k0`,

    A(k0 + δ) ≈ Σₗ₌₀ᵖ Aₗ δˡ,    Aₗ = A⁽ˡ⁾(k0) / l!.

For each requested wavenumber `ks[q]`, the Taylor polynomial and its first two
derivatives are evaluated at `δ_q = ks[q] - k0` by a simultaneous Horner
recurrence. The optimized full-boundary or symmetry-reduced construction is
selected automatically.

## Arguments
- `solver::SweepBIMSolver`: BIM solver defining the Fredholm operator.
- `pts::BoundaryPoints{Float64}`: Full-boundary discretization.
- `k0::Float64`: Center of the shared Taylor expansion.
- `ks::AbstractVector{<:Real}`: Wavenumbers at which the matrices are evaluated.
- `p::Int`: Taylor expansion degree.
- `multithreaded::Bool=true`: Enable multithreaded matrix construction.

## Returns
- `Abufs::Vector{Matrix{ComplexF64}}`: Fredholm matrices `A(ks[q])`.
- `dAbufs::Vector{Matrix{ComplexF64}}`: First derivatives `A'(ks[q])`.
- `ddAbufs::Vector{Matrix{ComplexF64}}`: Second derivatives `A''(ks[q])`.
"""
function _construct_matrices_multi_k_taylor_derivatives(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, ks::AbstractVector{<:Real}, p::Int; multithreaded::Bool = true)
    if solver.symmetry === nothing
        return _construct_matrices_multi_k_taylor_derivatives_full(solver, pts, k0, ks, p; multithreaded)
    end
    return _construct_matrices_multi_k_taylor_derivatives_reduced(solver, pts, k0, ks, p; multithreaded)
end
