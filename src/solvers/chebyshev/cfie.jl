################################################################################
# CHEBYSHEV-ACCELERATED CFIE FREDHOLM ASSEMBLY
#
# This file provides Chebyshev-accelerated assembly of the CFIE Fredholm
# operator
#
#                       A(k) = I - (D(k) + ikS(k)).
#
# The boundary discretization, Kress quadrature scheme, diagonal limits, and
# symmetry reduction are identical to the direct CFIE implementation. Only the
# repeated radial special-function evaluations in the off-diagonal kernels are
# replaced by the piecewise-Chebyshev plans defined in bessels.jl.
#
# The Kress splitting of the double-layer contribution requires H₁⁽¹⁾ and J₁,
# while the single-layer contribution requires H₀⁽¹⁾ and J₀. The value-only
# path therefore evaluates all four functions through Chebyshev interpolation.
# The derivative path inserts the same values into the analytic first- and
# second-k-derivative formulas used by EBIM; the Chebyshev polynomials
# themselves are not differentiated.
#
# For Beyn contour calculations, the multi-k routines traverse each boundary
# pair only once and evaluate all four special functions for every contour
# wavenumber during that visit, avoiding repeated streaming of the O(N²)
# geometry data.
#
# All required plans are constructed by the tuning functions in
# optimalpanelization.jl. Plans in a multi-wavenumber set share the same radial
# panelization so that panel indices and local Chebyshev coordinates can be
# reused across wavenumbers.
################################################################################

function _cfie_fredholm_full_cheb!(F::AbstractMatrix{ComplexF64}, pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, C::ChebRadialLookupCache, k::ComplexF64, plan0::ChebHankelPlanH, plan1::ChebHankelPlanH, planj0::ChebJPlan, planj1::ChebJPlan; multithreaded::Bool=true) where {T<:Real}
    invtwopi = inv(2*pi); αL1 = -k*invtwopi; αL2 = im*k/2; αM1 = -invtwopi; αM2 = ComplexF64(0,0.5); ik = im*k
    euler_over_pi = Float64(Base.MathConstants.eulergamma)/pi; N = length(pts)
    fill!(F,zero(ComplexF64))
    @inbounds for i in 1:N
        si = Float64(G.speed[i]); wi = Float64(pts.ws[i]); m1 = αM1*si
        dval = ComplexF64(wi*G.kappa[i],0.0)
        m2 = ((αM2-euler_over_pi)-invtwopi*log((k^2/4)*si^2))*si
        sval = ComplexF64(Rmat[i,i]*m1,0.0)+wi*m2
        F[i,i] = one(ComplexF64)-(dval+ik*sval)
    end
    @use_threads multithreading=(multithreaded && N>=32) for j in 2:N
        sj = Float64(G.speed[j]); wj = Float64(pts.ws[j])
        @inbounds for i in 1:j-1
            si = Float64(G.speed[i]); wi = Float64(pts.ws[i])
            r = Float64(G.R[i,j]); invr = Float64(G.invR[i,j]); lt = Float64(G.logterm[i,j])
            inn_ij = Float64(G.inner[i,j]); inn_ji = Float64(G.inner[j,i])
            ph = C.pidx_h[i,j]; th = C.t_h[i,j]; pj = C.pidx_j[i,j]; tj = C.t_j[i,j]
            h1 = eval_h(plan1,ph,th,r); h0 = eval_h(plan0,ph,th,r); j1 = eval_j(planj1,pj,tj,r); j0 = eval_j(planj0,pj,tj,r)
            l1 = αL1*j1; h = αL2*h1; m1j = αM1*j0*sj; m2j = αM2*h0*sj-m1j*lt
            c_ij = inn_ij*invr; d_ij = c_ij*(Rmat[i,j]*l1+wj*h-wj*lt*l1); s_ij = Rmat[i,j]*m1j+wj*m2j
            m1i = αM1*j0*si; m2i = αM2*h0*si-m1i*lt
            c_ji = inn_ji*invr; d_ji = c_ji*(Rmat[j,i]*l1+wi*h-wi*lt*l1); s_ji = Rmat[j,i]*m1i+wi*m2i
            F[i,j] = -(d_ij+ik*s_ij); F[j,i] = -(d_ji+ik*s_ji)
        end
    end
    return F
end

@inline function _cfie_kernel_entry_cheb(pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, C::ChebRadialLookupCache, k::ComplexF64, plan0::ChebHankelPlanH, plan1::ChebHankelPlanH, planj0::ChebJPlan, planj1::ChebJPlan, i::Int, j::Int) where {T<:Real}
    invtwopi = inv(2*pi); ik = im*k
    if i==j
        si = Float64(G.speed[i]); wi = Float64(pts.ws[i]); dval = ComplexF64(wi*G.kappa[i],0.0)
        euler_over_pi = Float64(Base.MathConstants.eulergamma)/pi; m1 = -invtwopi*si
        m2 = ((ComplexF64(0,0.5)-euler_over_pi)-invtwopi*log((k^2/4)*si^2))*si
        sval = ComplexF64(Rmat[i,i]*m1,0.0)+wi*m2
        return dval+ik*sval
    end
    r = Float64(G.R[i,j]); invr = Float64(G.invR[i,j]); lt = Float64(G.logterm[i,j]); inn = Float64(G.inner[i,j])
    sj = Float64(G.speed[j]); wj = Float64(pts.ws[j])
    ph = C.pidx_h[i,j]; th = C.t_h[i,j]; pj = C.pidx_j[i,j]; tj = C.t_j[i,j]
    h1 = eval_h(plan1,ph,th,r); h0 = eval_h(plan0,ph,th,r); j1 = eval_j(planj1,pj,tj,r); j0 = eval_j(planj0,pj,tj,r)
    αL1 = -k*invtwopi; αL2 = im*k/2; αM1 = -invtwopi; αM2 = ComplexF64(0,0.5)
    l1 = αL1*j1; h = αL2*h1; c = inn*invr
    dval = c*(Rmat[i,j]*l1+wj*h-wj*lt*l1)
    m1 = αM1*j0*sj; m2 = αM2*h0*sj-m1*lt
    sval = Rmat[i,j]*m1+wj*m2
    return dval+ik*sval
end

function _cfie_fredholm_reduced_cheb!(F::AbstractMatrix{ComplexF64}, pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, C::ChebRadialLookupCache, orbits::SymmetryOrbitMap{T}, k::ComplexF64, plan0::ChebHankelPlanH, plan1::ChebHankelPlanH, planj0::ChebJPlan, planj1::ChebJPlan; multithreaded::Bool=true) where {T<:Real}
    m = fundamental_size(orbits); N = length(orbits); fund = orbits.fundamental_indices; orbit_of = orbits.orbit_of; phase = orbits.phase
    images = [Int[] for _ in 1:m]
    @inbounds for j in 1:N
        push!(images[orbit_of[j]],j)
    end
    invtwopi = inv(2*pi); αL1 = -k*invtwopi; αL2 = im*k/2; αM1 = -invtwopi; αM2 = ComplexF64(0,0.5); ik = im*k
    euler_over_pi = Float64(Base.MathConstants.eulergamma)/pi
    fill!(F,zero(ComplexF64))
    @use_threads multithreading=(multithreaded && m>=32) for b in 1:m
        @inbounds for a in 1:m
            i = fund[a]; acc = zero(ComplexF64)
            for j in images[b]
                phase_j = phase[j]
                if i==j
                    si = Float64(G.speed[i]); wi = Float64(pts.ws[i]); dval = ComplexF64(wi*G.kappa[i],0.0); m1 = αM1*si
                    m2 = ((αM2-euler_over_pi)-invtwopi*log((k^2/4)*si^2))*si
                    sval = ComplexF64(Rmat[i,i]*m1,0.0)+wi*m2
                    acc += phase_j*(dval+ik*sval)
                else
                    r = Float64(G.R[i,j]); invr = Float64(G.invR[i,j]); lt = Float64(G.logterm[i,j]); inn = Float64(G.inner[i,j])
                    Rij = Float64(Rmat[i,j]); wj = Float64(pts.ws[j]); sj = Float64(G.speed[j])
                    ph = C.pidx_h[i,j]; th = C.t_h[i,j]; pj = C.pidx_j[i,j]; tj = C.t_j[i,j]
                    h1 = eval_h(plan1,ph,th,r); h0 = eval_h(plan0,ph,th,r); j1 = eval_j(planj1,pj,tj,r); j0 = eval_j(planj0,pj,tj,r)
                    c = phase_j*inn*invr; c1 = c*Rij; c2 = c*wj; c3 = c2*lt; l1 = αL1*j1
                    dval = c1*l1+c2*αL2*h1-c3*l1
                    m1 = αM1*j0*sj; m2 = αM2*h0*sj-m1*lt
                    sval = Rij*m1+wj*m2
                    acc += dval+phase_j*ik*sval
                end
            end
            F[a,b] = -acc
        end
        F[b,b] += one(ComplexF64)
    end
    return F
end

@inline function _cfie_kernel_entry_with_derivatives_cheb(pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, C::ChebRadialLookupCache, k::ComplexF64, plan0::ChebHankelPlanH, plan1::ChebHankelPlanH, planj0::ChebJPlan, planj1::ChebJPlan, i::Int, j::Int) where {T<:Real}
    invtwopi = inv(2*pi); ik = im*k
    if i==j
        si = Float64(G.speed[i]); wi = pts.ws[i]; Dval = ComplexF64(wi*G.kappa[i],0.0)
        euler_over_pi = Float64(Base.MathConstants.eulergamma)/pi; m1 = -invtwopi*si
        m2 = ((ComplexF64(0,0.5)-euler_over_pi)-invtwopi*log((k^2/4)*si^2))*si
        Sval = ComplexF64(Rmat[i,i]*m1,0.0)+wi*m2
        val = Dval+ik*Sval
        dm2 = -si/(pi*k); ddm2 = si/(pi*k^2); dSval = wi*dm2; ddSval = wi*ddm2
        dval = im*Sval+ik*dSval; ddval = 2*im*dSval+ik*ddSval
        return val,dval,ddval
    end
    r = Float64(G.R[i,j]); invr = Float64(G.invR[i,j]); lt = Float64(G.logterm[i,j]); inn = Float64(G.inner[i,j])
    sj = Float64(G.speed[j]); wj = pts.ws[j]
    ph = C.pidx_h[i,j]; th = C.t_h[i,j]; pj = C.pidx_j[i,j]; tj = C.t_j[i,j]
    h1 = eval_h(plan1,ph,th,r); h0 = eval_h(plan0,ph,th,r); j1 = eval_j(planj1,pj,tj,r); j0 = eval_j(planj0,pj,tj,r)
    αL1 = -k*invtwopi; αL2 = im*k/2; αM1 = -invtwopi; αM2 = ComplexF64(0,0.5)
    l1,dl1,ddl1 = _ebim_lin1_deriv(αL1*inn,r,invr,j0,j1,k)
    l2a,dl2a,ddl2a = _ebim_lin1_deriv(αL2*inn,r,invr,h0,h1,k)
    l2 = l2a-l1*lt; dl2 = dl2a-dl1*lt; ddl2 = ddl2a-ddl1*lt
    Dval = Rmat[i,j]*l1+wj*l2; dDval = Rmat[i,j]*dl1+wj*dl2; ddDval = Rmat[i,j]*ddl1+wj*ddl2
    m1,dm1,ddm1 = _ebim_const0_deriv(αM1*sj,r,k,j0,j1)
    m2a,dm2a,ddm2a = _ebim_const0_deriv(αM2*sj,r,k,h0,h1)
    m2 = m2a-m1*lt; dm2 = dm2a-dm1*lt; ddm2 = ddm2a-ddm1*lt
    Sval = Rmat[i,j]*m1+wj*m2; dSval = Rmat[i,j]*dm1+wj*dm2; ddSval = Rmat[i,j]*ddm1+wj*ddm2
    val = Dval+ik*Sval; dval = dDval+im*Sval+ik*dSval; ddval = ddDval+2*im*dSval+ik*ddSval
    return val,dval,ddval
end

function _cfie_fredholm_full_multi_k_cheb!(Fs::Vector{<:AbstractMatrix{ComplexF64}}, pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, C::ChebRadialLookupCache, zj::Vector{ComplexF64}, plans0::Vector{ChebHankelPlanH}, plans1::Vector{ChebHankelPlanH}, plansj0::Vector{ChebJPlan}, plansj1::Vector{ChebJPlan}; multithreaded::Bool=true) where {T<:Real}
    Mk = length(zj)
    @assert length(Fs)==Mk && length(plans0)==Mk && length(plans1)==Mk && length(plansj0)==Mk && length(plansj1)==Mk
    invtwopi = inv(2*pi); N = length(pts); euler_over_pi = Float64(Base.MathConstants.eulergamma)/pi
    αM1 = -invtwopi; αM2 = ComplexF64(0,0.5)
    αL1 = Vector{ComplexF64}(undef,Mk); αL2 = Vector{ComplexF64}(undef,Mk); ik = Vector{ComplexF64}(undef,Mk)
    @inbounds for mm in 1:Mk
        k = zj[mm]; αL1[mm] = -k*invtwopi; αL2[mm] = im*k/2; ik[mm] = im*k
        fill!(Fs[mm],zero(ComplexF64))
        for i in 1:N
            si = Float64(G.speed[i]); wi = Float64(pts.ws[i]); dval = ComplexF64(wi*G.kappa[i],0.0); m1 = αM1*si
            m2 = ((αM2-euler_over_pi)-invtwopi*log((k^2/4)*si^2))*si
            sval = ComplexF64(Rmat[i,i]*m1,0.0)+wi*m2
            Fs[mm][i,i] = one(ComplexF64)-(dval+ik[mm]*sval)
        end
    end
    nt = _cheb_nthreads_buf()
    h0_tls = [Vector{ComplexF64}(undef,Mk) for _ in 1:nt]; h1_tls = [Vector{ComplexF64}(undef,Mk) for _ in 1:nt]
    j0_tls = [Vector{ComplexF64}(undef,Mk) for _ in 1:nt]; j1_tls = [Vector{ComplexF64}(undef,Mk) for _ in 1:nt]
    @use_threads multithreading=(multithreaded && N>=32) for j in 2:N
        sj = Float64(G.speed[j]); wj = Float64(pts.ws[j]); tid = Threads.threadid()
        h0vals = h0_tls[tid]; h1vals = h1_tls[tid]; j0vals = j0_tls[tid]; j1vals = j1_tls[tid]
        @inbounds for i in 1:j-1
            si = Float64(G.speed[i]); wi = Float64(pts.ws[i])
            r = Float64(G.R[i,j]); invr = Float64(G.invR[i,j]); lt = Float64(G.logterm[i,j])
            inn_ij = Float64(G.inner[i,j]); inn_ji = Float64(G.inner[j,i])
            ph = C.pidx_h[i,j]; th = C.t_h[i,j]; pj = C.pidx_j[i,j]; tj = C.t_j[i,j]
            h0_h1_j0_j1_multi_ks_at_r!(h0vals,h1vals,j0vals,j1vals,plans0,plans1,plansj0,plansj1,ph,th,pj,tj,r)
            Rij = Float64(Rmat[i,j]); Rji = Float64(Rmat[j,i])
            c_ij = inn_ij*invr; c_ji = inn_ji*invr
            cR_ij = c_ij*Rij; cw_ij = c_ij*wj; cwl_ij = cw_ij*lt
            cR_ji = c_ji*Rji; cw_ji = c_ji*wi; cwl_ji = cw_ji*lt
            for mm in 1:Mk
                l1 = αL1[mm]*j1vals[mm]; h = αL2[mm]*h1vals[mm]
                dval_ij = cR_ij*l1+cw_ij*h-cwl_ij*l1
                m1_ij = αM1*j0vals[mm]*sj; m2_ij = αM2*h0vals[mm]*sj-m1_ij*lt
                sval_ij = Rij*m1_ij+wj*m2_ij
                dval_ji = cR_ji*l1+cw_ji*h-cwl_ji*l1
                m1_ji = αM1*j0vals[mm]*si; m2_ji = αM2*h0vals[mm]*si-m1_ji*lt
                sval_ji = Rji*m1_ji+wi*m2_ji
                Fs[mm][i,j] = -(dval_ij+ik[mm]*sval_ij)
                Fs[mm][j,i] = -(dval_ji+ik[mm]*sval_ji)
            end
        end
    end
    return Fs
end

function _cfie_fredholm_reduced_multi_k_cheb!(Fs::Vector{<:AbstractMatrix{ComplexF64}}, pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, C::ChebRadialLookupCache, orbits::SymmetryOrbitMap{T}, zj::Vector{ComplexF64}, plans0::Vector{ChebHankelPlanH}, plans1::Vector{ChebHankelPlanH}, plansj0::Vector{ChebJPlan}, plansj1::Vector{ChebJPlan}; multithreaded::Bool=true) where {T<:Real}
    Mk = length(zj)
    @assert length(Fs)==Mk && length(plans0)==Mk && length(plans1)==Mk && length(plansj0)==Mk && length(plansj1)==Mk
    m = fundamental_size(orbits); N = length(orbits); fund = orbits.fundamental_indices; orbit_of = orbits.orbit_of; phase = orbits.phase
    images = [Int[] for _ in 1:m]
    @inbounds for j in 1:N
        push!(images[orbit_of[j]],j)
    end
    invtwopi = inv(2*pi); euler_over_pi = Float64(Base.MathConstants.eulergamma)/pi
    αM1 = -invtwopi; αM2 = ComplexF64(0,0.5)
    αL1 = Vector{ComplexF64}(undef,Mk); αL2 = Vector{ComplexF64}(undef,Mk); ik = Vector{ComplexF64}(undef,Mk)
    @inbounds for mm in 1:Mk
        k = zj[mm]; αL1[mm] = -k*invtwopi; αL2[mm] = im*k/2; ik[mm] = im*k
        fill!(Fs[mm],zero(ComplexF64))
    end
    nt = _cheb_nthreads_buf()
    h0_tls = [Vector{ComplexF64}(undef,Mk) for _ in 1:nt]; h1_tls = [Vector{ComplexF64}(undef,Mk) for _ in 1:nt]
    j0_tls = [Vector{ComplexF64}(undef,Mk) for _ in 1:nt]; j1_tls = [Vector{ComplexF64}(undef,Mk) for _ in 1:nt]
    acc_tls = [Vector{ComplexF64}(undef,Mk) for _ in 1:nt]
    @use_threads multithreading=(multithreaded && m>=32) for b in 1:m
        tid = Threads.threadid(); h0vals = h0_tls[tid]; h1vals = h1_tls[tid]; j0vals = j0_tls[tid]; j1vals = j1_tls[tid]; acc = acc_tls[tid]
        @inbounds for a in 1:m
            i = fund[a]; fill!(acc,zero(ComplexF64))
            for j in images[b]
                phase_j = phase[j]
                if i==j
                    si = Float64(G.speed[i]); wi = Float64(pts.ws[i]); dval = ComplexF64(wi*G.kappa[i],0.0); m1 = αM1*si
                    for mm in 1:Mk
                        k = zj[mm]; m2 = ((αM2-euler_over_pi)-invtwopi*log((k^2/4)*si^2))*si
                        sval = ComplexF64(Rmat[i,i]*m1,0.0)+wi*m2
                        acc[mm] += phase_j*(dval+ik[mm]*sval)
                    end
                else
                    r = Float64(G.R[i,j]); invr = Float64(G.invR[i,j]); lt = Float64(G.logterm[i,j]); inn = Float64(G.inner[i,j])
                    Rij = Float64(Rmat[i,j]); wj = Float64(pts.ws[j]); sj = Float64(G.speed[j])
                    ph = C.pidx_h[i,j]; th = C.t_h[i,j]; pj = C.pidx_j[i,j]; tj = C.t_j[i,j]
                    h0_h1_j0_j1_multi_ks_at_r!(h0vals,h1vals,j0vals,j1vals,plans0,plans1,plansj0,plansj1,ph,th,pj,tj,r)
                    c = phase_j*inn*invr; c1 = c*Rij; c2 = c*wj; c3 = c2*lt
                    pR = phase_j*Rij; pw = phase_j*wj
                    for mm in 1:Mk
                        l1 = αL1[mm]*j1vals[mm]
                        dval = c1*l1+c2*αL2[mm]*h1vals[mm]-c3*l1
                        m1 = αM1*j0vals[mm]*sj; m2 = αM2*h0vals[mm]*sj-m1*lt
                        sval = pR*m1+pw*m2
                        acc[mm] += dval+ik[mm]*sval
                    end
                end
            end
            for mm in 1:Mk
                Fs[mm][a,b] = -acc[mm]
            end
        end
        for mm in 1:Mk
            Fs[mm][b,b] += one(ComplexF64)
        end
    end
    return Fs
end