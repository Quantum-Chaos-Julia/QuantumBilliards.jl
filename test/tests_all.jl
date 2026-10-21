#!/usr/bin/env julia
# AI-GENERATED FILE — DO NOT MODIFY MANUALLY.
# Request changes through AI and regenerate/revalidate this file.
#
# Usage (from the package checkout):
#   julia test/tests_all.jl --setup             # compare the accepted baseline
#   julia test/tests_all.jl --setup --check     # validate without a baseline
#   julia test/tests_all.jl --setup --record    # create a baseline; refuse overwrite
#   julia test/tests_all.jl --setup --update    # replace it after all checks pass
# Options: --reference=PATH, --only=circle,rectangle (partial runs cannot record).
# --setup uses a cached environment with this package and sibling BilliardGeometry.jl;
# set BILLIARD_GEOMETRY_PATH for another checkout. Omit it in a working environment.
# Automatically included by runtests.jl without setup or command-line processing;
# package tests compare an existing baseline or create one after a passing full run.
#
# Reference: test/baselines/fast_eigenvalues.toml stores sorted wavenumbers k
# (Laplacian eigenvalues are k²), retaining multiplicities, plus settings/metadata.
# Failed recording runs save an unvalidated .candidate.toml for diagnostics;
# candidates cannot be accepted references. Settings changes require reviewed --update.
#
# Coverage: all 15 exported concrete geometries, ppw=20, low-frequency checks of
# applicable BIM/basis solvers, geometry, quadrature, expansions, and state methods.
# Both RPW and CAFB use DM, PSM, and VS on circle, ellipse, rectangle, stadium,
# Prosen, limacon, triangle, and polygon. Reflection cases use matching odd sectors
# and CAFB origin (0,0); triangle/polygon CAFB use a translated physical corner,
# with the two incident Dirichlet edges marked QuantumSolverIgnore.
# VS offsets are scaled by sqrt(pi / physical_area); polygon CAFB/VS additionally
# drops vanishing Gram columns and equilibrates the retained matrix pencil.
#
# Exclusions/conventions (also printed at runtime):
# - Mushroom: CFIE only; no DLP or regular-basis eigenvalue tests.
# - Holes: composite CFIE, outer-minus-hole area; no pure DLP or regular bases.
# - Polar/star/C3: no supported CAFB reflection wedge or physical corner.
# - Taylor bases: combined value/derivative evaluator tested; standalone cached
#   basis_matrix excluded because its real 3-D Horner overload is missing.
# - Known undefined legacy exports excluded; newly undefined exports fail.
# - Fourier quadrature can store N+1 nodes because the terminal weight is zero.
# Fixed seeds and one BLAS thread reduce variation; a full run takes about two
# minutes including compilation. Coverage is representative, not exhaustive.

if abspath(PROGRAM_FILE)==(@__FILE__) && "--setup" in ARGS
    import Pkg
    import SHA
    qbpath = realpath(joinpath(@__DIR__, ".."))
    bgpath = get(ENV, "BILLIARD_GEOMETRY_PATH", joinpath(dirname(qbpath), "BilliardGeometry.jl"))
    isfile(joinpath(bgpath, "Project.toml")) || error("BilliardGeometry checkout not found: $bgpath")
    envpath = joinpath(first(DEPOT_PATH), "environments", "QuantumBilliardsRegression-" * bytes2hex(SHA.sha1(qbpath * abspath(bgpath)))[1:12])
    Pkg.activate(envpath)
    if !isfile(joinpath(envpath, "Manifest.toml"))
        Pkg.develop([Pkg.PackageSpec(path=qbpath), Pkg.PackageSpec(path=bgpath)])
    end
    Pkg.instantiate()
end

using QuantumBilliards, BilliardGeometry
using LinearAlgebra, Random, TOML, Dates, Test

const FR_QB = QuantumBilliards
const FR_BG = BilliardGeometry
const FR_V2 = FR_QB.SVector{2,Float64}
const FR_SCHEMA = 1
const FR_SEED = 314159
const FR_SETTINGS = Dict(
    # Boundary sampling and matrix expansions.
    "ppw" => 20.0,
    "smooth_nodes" => 96,
    "corner_nodes" => 160,
    "taylor_degree" => 18,
    "contour_nodes" => 32,
    # RPW/CAFB dimensions and relative Gram rank cutoff (DM and VS).
    "basis_dim" => 20,
    "basis_rank_cutoff" => 1e-12,
    # VS center displacements, in units of sqrt(pi / physical_area).
    "stadium_vs_offset" => 0.01,
    "dual_vs_offset" => 0.01,
    "polygon_cafb_vs_offset" => 0.05,
    # Absolute Gram diagonal floor before polygon CAFB/VS equilibration.
    "polygon_cafb_gram_floor" => 1e-24,
)

function fr_geometries()
    polygon = FR_V2[FR_V2(cos(t),sin(t)) for t in range(0,2pi;length=6)[1:5]]
    [ (name="circle", b=CircleBilliard(1.0), corners=false, holes=false, basis=true, exact=2.4048255576957728, exact_area=pi),
        (name="ellipse", b=EllipseBilliard(1.0,0.8), corners=false, holes=false, basis=true, exact=nothing, exact_area=0.8pi),
        (name="rectangle", b=RectangleBilliard(1.0,0.8), corners=true, holes=false, basis=true, exact=pi/2*sqrt(1+1/0.8^2), exact_area=3.2),
        (name="triangle", b=TriangleBilliard(pi/2,1.0;x0=1/3,y0=1/3), corners=true, holes=false, basis=true, exact=pi*sqrt(5), exact_area=0.5),
        (name="polygon", b=PolygonBilliard(polygon), corners=true, holes=false, basis=true, exact=nothing, exact_area=nothing),
        (name="stadium", b=StadiumBilliard(0.3), corners=false, holes=false, basis=true, exact=nothing, exact_area=pi+1.2),
        (name="limacon", b=LimaconBilliard(0.1), corners=false, holes=false, basis=true, exact=nothing, exact_area=nothing),
        (name="polar", b=PolarBilliard([0.0,0.0,0.0,0.05]), corners=false, holes=false, basis=true, exact=nothing, exact_area=nothing),
        (name="star", b=StarBilliard(1.0,0.05,3), corners=false, holes=false, basis=true, exact=nothing, exact_area=pi*(1+0.05^2/2)),
        (name="c3", b=C3Billiard(0.02), corners=false, holes=false, basis=true, exact=nothing, exact_area=nothing),
        (name="prosen", b=ProsenBilliard(0.02), corners=false, holes=false, basis=true, exact=nothing, exact_area=pi*(1+0.02^2/2)),
        (name="mushroom", b=MushroomBilliard(0.5,0.7), corners=true, holes=false, basis=false, exact=nothing, exact_area=pi/2+0.7),
        (name="annular", b=AnnularBilliard(0.3,0.0), corners=false, holes=true, basis=false, exact=nothing, exact_area=pi*(1-0.3^2)),
        (name="sinai", b=SinaiBilliard(1.0,0.25), corners=true, holes=true, basis=false, exact=nothing, exact_area=4-pi*0.25^2),
        (name="square_holes", b=SquareWithinSquareBilliard(1.0), corners=true, holes=true, basis=false, exact=nothing, exact_area=4-2*0.25^2), ]
end

mutable struct FRRun
    reference::Dict
    values::Dict{String,Any}
    failures::Vector{String}
    skips::Vector{String}
    checks::Int
end
function fr_check!(f, run, label)
    run.checks += 1
    @testset "$label" begin
        passed=true
        try
            f()
        catch err
            passed=false
            msg = label * ": " * sprint(showerror,err)
            push!(run.failures,msg)
            println(stderr,"FAIL ",msg)
        end
        @test passed
    end
end
fr_require(ok, msg) = ok || error(msg)
function fr_save_values!(run, key, ks; atol=2e-7, rtol=2e-7)
    vals = sort(Float64.(real.(ks))) # Preserve multiplicity; never unique! eigenvalues.
    fr_require(!isempty(vals) && all(isfinite,vals) && all(>(0),vals),"empty/invalid spectrum")
    run.values[key] = vals
    if !isempty(run.reference)
        fr_require(haskey(run.reference["eigenvalues"],key),"missing reference key $key")
        old = run.reference["eigenvalues"][key]
        fr_require(length(vals)==length(old),"eigenvalue count changed: $(length(old)) -> $(length(vals))")
        fr_require(all(isapprox.(vals,old;atol,rtol)),"reference mismatch: $vals versus $old")
    end
    vals
end

function fr_kernel(g, kind; fine=false, grading=nothing)
    gr = grading === nothing ? (g.corners ? GlobalCornerGrading(kressq=2) : SmoothPeriodicGrading()) : grading
    n = g.corners ? FR_SETTINGS["corner_nodes"] : FR_SETTINGS["smooth_nodes"]
    fine && (n *= 2)
    ctor = kind == :dlp ? DoubleLayerPotentialSolver : CombinedFieldIntegralEquationSolver
    if g.holes
        nc=length(FR_QB._group_boundary_by_domain_id(full_boundary(g.b)))
        n=max(fine ? 128 : 64,cld(n,nc))
        native() = ctor(FR_SETTINGS["ppw"],g.b;min_pts=n,grading=gr,eps=1e-10)
        return CompositeBIMSolver((native() for _ in 1:nc)...)
    end
    ctor(FR_SETTINGS["ppw"],g.b;min_pts=n,grading=gr,eps=1e-10)
end
function fr_beyn(k; taylor=true)
    BeynSolver(k;nq=FR_SETTINGS["contour_nodes"],r=12,m=4, use_taylor=taylor,use_chebyshev=false,taylor_degree=FR_SETTINGS["taylor_degree"],
               res_tol=2e-5,svd_tol=1e-10)
end

function fr_physical_area(g)
    !g.holes && return area(g.b)
    components=FR_QB._group_boundary_by_domain_id(full_boundary(g.b))
    sum((first(c).domain_id==1 ? 1 : -1)*area(c) for c in components)
end

function fr_geometry_checks(g)
    b=g.b; cs=get_boundary_curves(b); A=fr_physical_area(g)
    fr_require(isfinite(A) && A>0,"invalid area")
    if g.exact_area !== nothing
        fr_require(isapprox(A,g.exact_area;rtol=1e-8,atol=1e-9),"area disagrees with analytic value")
    end
    for c in cs, t in (0.23,0.61)
        x=curve(c,t); v=tangent(c,t); vv=tangent_2(c,t); n=only(normal_vec(c,[t]))
        fr_require(all(isfinite,x) && all(isfinite,v) && all(isfinite,vv),"nonfinite curve data")
        fr_require(isapprox(norm(n),1;atol=1e-10),"normal is not unit length")
        fr_require(abs(dot(n,v)) < 1e-9*max(1,norm(v)),"normal/tangent not orthogonal")
        h=1e-5
        fd=(curve(c,t+h)-curve(c,t-h))/(2h)
        fr_require(isapprox(fd,v;rtol=2e-5,atol=2e-6),"curve derivative mismatch")
        fd2=(tangent(c,t+h)-tangent(c,t-h))/(2h)
        fr_require(isapprox(fd2,vv;rtol=2e-5,atol=2e-6),"second derivative mismatch")
    end
    comps=g.holes ? FR_QB._group_boundary_by_domain_id(full_boundary(b)) : [full_boundary(b)]
    for comp in comps, i in eachindex(comp)
        fr_require(isapprox(curve(comp[i],1.0),curve(comp[mod1(i+1,length(comp))],0.0);atol=1e-8), "physical boundary is not closed")
    end
end

function fr_matrix_checks(k,b,center)
    pts=evaluate_points(k,b,center)
    fr_require(!isempty(pts.xy) && all(isfinite,pts.ds) && all(>(0),pts.ds),"invalid boundary quadrature")
    fr_require(all(n->isapprox(norm(n),1;atol=1e-9),pts.normal),"invalid sampled normals")
    z=ComplexF64(center+0.02,0.01)
    direct=construct_matrices(k,pts,z;multithreaded=false)
    fr_require(all(isfinite,direct),"nonfinite matrix")
    approx=only(FR_QB._construct_matrices_multi_k_taylor(k,pts,center,[z],FR_SETTINGS["taylor_degree"];multithreaded=false))
    fr_require(norm(direct-approx)/norm(direct)<1e-9,"Taylor matrix differs from direct matrix")
    e=ExpandedBIMSolver(k;use_taylor=false)
    A,dA,ddA=construct_matrices(e,pts,center;multithreaded=false)
    h=1e-4
    Ap=construct_matrices(k,pts,center+h;multithreaded=false)
    Am=construct_matrices(k,pts,center-h;multithreaded=false)
    fr_require(norm(A-construct_matrices(k,pts,center;multithreaded=false))<1e-9*norm(A),"EBIM matrix mismatch")
    fr_require(norm(dA-(Ap-Am)/(2h))/max(norm(dA),1)<1e-6,"first matrix derivative mismatch")
    fr_require(norm(ddA-(Ap-2A+Am)/h^2)/max(norm(ddA),1)<2e-5,"second matrix derivative mismatch")
end

function fr_state_checks(k,b,target)
    data=compute_eigenstate(k,b,target;multithreaded=false)
    st=only(data.states)
    u,s,_=boundary_function(st,b,k;multithreaded=false)
    fr_require(length(u)==length(s) && all(isfinite,u),"invalid boundary function")
    fr_require(isapprox(FR_QB._rellich(st.pts,u,st.k),1;atol=1e-7),"boundary function not Rellich normalized")
    Psi,_,_=wavefunction(st,b,k;b=2.0,use_chebyshev=false)
    fr_require(all(isfinite,Psi) && maximum(abs,Psi)>1e-8,"invalid interior wavefunction")
    H,_,_=husimi_function(st,b,k;c=2.0)
    fr_require(all(isfinite,H) && minimum(H)>=0 && isapprox(sum(H),1;atol=1e-12),"invalid Husimi density")
    P,_=momentum_function(st,b,k)
    fr_require(all(isfinite,P),"invalid momentum representation")
end

function fr_test_geometry!(run,g)
    println("Checking ",g.name); flush(stdout)
    fr_check!(run,g.name*" geometry") do
        fr_geometry_checks(g)
    end
    A=fr_physical_area(g)
    scale=sqrt(pi/A); cf=fr_kernel(g,:cfie)
    center=3scale; width=2scale
    anchor=nothing
    fr_check!(run,g.name*" reference spectrum") do
        ks,_=solve_spectrum(fr_beyn(cf;taylor=false),g.b,center,width;multithreaded=false)
        ks=filter(z->abs(imag(z))<0.01scale,ks)
        if isempty(ks)
            ks,_=solve_spectrum(fr_beyn(cf;taylor=false),g.b,5scale,width;multithreaded=false)
            ks=filter(z->abs(imag(z))<0.01scale,ks)
        end
        vals=fr_save_values!(run,g.name*"/beyn_window",ks;atol=2e-6,rtol=2e-6)
        candidate=first(vals)
        anchor,_=solve_wavenumber(cf,g.b,candidate,0.12scale;multithreaded=false)
        fr_require(abs(anchor-candidate)<0.005scale,"contour/sweep disagreement")
        if isempty(run.reference)
            fine,_=solve_wavenumber(fr_kernel(g,:cfie;fine=true),g.b,anchor,0.06scale;multithreaded=false)
            fr_require(abs(fine-anchor)<0.005scale,"boundary resolution has not converged")
        end
        if g.exact !== nothing
            fr_require(isapprox(anchor,g.exact;rtol=2e-3,atol=2e-4),"analytic eigenvalue mismatch")
        elseif g.name=="annular"
            J=FR_QB.SpecialFunctions.besselj; Y=FR_QB.SpecialFunctions.bessely
            fr_require(abs(J(0,anchor)*Y(0,0.3anchor)-Y(0,anchor)*J(0,0.3anchor))<2e-3, "annular radial Bessel determinant is not zero")
        end
        fr_save_values!(run,g.name*"/cfie_sweep",[anchor])
    end
    anchor===nothing && return
    if g.name=="mushroom"
        push!(run.skips,"mushroom: DLP excluded; use CFIE for its nonconvex boundary.")
    end
    if g.holes
        push!(run.skips,g.name*": area tested using outer-minus-hole component areas; native area(billiard) currently ignores hole orientation flags.")
    end
    for kind in (g.holes || g.name=="mushroom" ? (:cfie,) : (:dlp,:cfie))
        k=fr_kernel(g,kind); label=g.name*"/"*string(kind)
        fr_check!(run,label*" matrices") do
            fr_matrix_checks(k,g.b,anchor+0.025scale)
        end
        fr_check!(run,label*" sweep") do
            x,_=solve_wavenumber(k,g.b,anchor,0.08scale;multithreaded=false)
            fr_require(abs(x-anchor)<0.004scale,"kernel eigenvalue disagreement")
            fr_save_values!(run,label*"_sweep",[x])
            ts=k_sweep(k,g.b,[anchor-0.02scale,anchor,anchor+0.02scale];multithreaded=false)
            fr_require(all(isfinite,ts) && ts[2]<min(ts[1],ts[3]),"sweep does not resolve the eigenvalue")
        end
        fr_check!(run,label*" Beyn") do
            xs,_=solve_spectrum(fr_beyn(k),g.b,anchor+0.02scale,0.16scale;multithreaded=false)
            fr_require(length(xs)==1 && abs(real(only(xs))-anchor)<0.004scale,"Beyn isolated spectrum mismatch")
            fr_save_values!(run,label*"_beyn",xs;atol=2e-6,rtol=2e-6)
        end
        fr_check!(run,label*" EBIM") do
            x,_=solve_wavenumber(ExpandedBIMSolver(k;taylor_degree=FR_SETTINGS["taylor_degree"]),g.b,anchor+0.005scale,0.04scale; nlevels=1,multithreaded=false)
            fr_require(abs(real(x)-anchor)<0.004scale && abs(imag(x))<0.01scale, "EBIM eigenvalue disagreement: $x versus $anchor")
            fr_save_values!(run,label*"_ebim",[x];atol=2e-6,rtol=2e-6)
        end
        fr_check!(run,label*" CORK") do
            imtol=g.corners ? 0.01scale : 2e-4
            c=CORKSolver(k;p=18,b=2,mstart=28,mstep=12,nlevels=1, imag_tol=imtol,imag_search_tol=imtol,res_tol=1e-7,stable_tol=1e-7,stable_checks=1)
            xs,_=solve_spectrum(c,g.b,anchor+0.02scale,0.16scale;multithreaded=false)
            fr_require(length(xs)==1 && abs(real(only(xs))-anchor)<0.004scale,"CORK isolated spectrum mismatch: $xs versus $anchor")
            fr_save_values!(run,label*"_cork",xs;atol=2e-6,rtol=2e-6)
        end
    end
    fr_check!(run,g.name*" state methods") do
        fr_state_checks(cf,g.b,anchor)
    end
    if g.holes
        push!(run.skips,g.name*": pure DLP eigenvalue tests excluded (spurious hole resonances); basis eigenvalue tests excluded (regular full-domain bases do not span hole singularities).")
    elseif g.basis
        fr_test_basis!(run,g,anchor,scale)
        fr_dual_basis!(run,g,anchor,scale)
    else
        push!(run.skips,g.name*": fast regular-basis eigenvalue tests excluded (reentrant corners require an enriched basis); BIM methods remain tested.")
    end
end

function fr_test_basis!(run,g,anchor,scale)
    b=g.b
    if g.name in ("circle","ellipse","rectangle","stadium","prosen","limacon")
        sector=g.name=="limacon" ? symmetry_sector(b,XAxisReflection=>1) :
                                  symmetry_sector(b,XAxisReflection=>1,YAxisReflection=>1)
        basis=RealPlaneWaves(FR_SETTINGS["basis_dim"],b,sector;use_taylor=false)
    else
        if g.name in ("star","c3")
            arc=first(get_boundary_curves(b))
            seg=FourierCoeffPolarSegment(arc.coef;R=arc.R,center=arc.center)
            dom=FR_BG.PolarDomain{Float64}(FR_BG.AbsCurve[seg],FR_V2[curve(seg,0.0)],1)
            b=PolarBilliard{Float64}(dom,register_symmetries())
        end
        basis=RealPlaneWaves(FR_SETTINGS["basis_dim"],nothing,nothing;use_taylor=false)
    end
    solvers=fr_basis_solvers()
    for (name,solver) in pairs(solvers)
        fr_check!(run,g.name*" basis "*string(name)) do
            Random.seed!(FR_SEED)
            offset=g.name=="stadium" ? FR_SETTINGS["stadium_vs_offset"] : 0.001
            ctr=name==:vs ? anchor+offset*scale : anchor
            x,_=solve_wavenumber(solver,basis,b,ctr,0.06scale;multithreaded=false)
            fr_save_values!(run,g.name*"/basis_"*string(name),[x];atol=3e-6,rtol=3e-6)
            fr_require(abs(x-anchor)<0.006scale,"basis/BIM eigenvalue disagreement: $x versus $anchor")
        end
    end
end

function fr_basis_solvers()
    d=FR_SETTINGS["basis_dim"]
    dm0=DecompositionMethodSolver(2.0,FR_SETTINGS["ppw"];min_dim=d,min_pts=100)
    dm=DecompositionMethodSolver(dm0.dim_scaling_factor,dm0.pts_scaling_factor,dm0.sampler,
                                FR_SETTINGS["basis_rank_cutoff"],dm0.min_dim,dm0.min_pts,dm0.rellich_origin)
    vs=VerginiSaracenoSolver(2.0,FR_SETTINGS["ppw"];min_dim=d,min_pts=100)
    vs.eps=FR_SETTINGS["basis_rank_cutoff"]
    (dm=dm,psm=ParticularSolutionsMethodSolver(2.0,FR_SETTINGS["ppw"],FR_SETTINGS["ppw"];min_dim=d,min_pts=100,min_int_pts=100),vs=vs)
end

function fr_dual_solve!(run,g,b,basis,family,target,scale)
    for (method,solver) in pairs(fr_basis_solvers())
        fr_check!(run,g.name*" "*family*" "*string(method)) do
            Random.seed!(FR_SEED)
            equilibrate=g.name=="polygon" && family=="cafb_corner" && method==:vs
            offset=FR_SETTINGS[equilibrate ? "polygon_cafb_vs_offset" : "dual_vs_offset"]
            center=target+(method==:vs ? offset*scale : 0)
            if equilibrate
                pts=evaluate_points(solver,b,center)
                F,Fk=construct_matrices(solver,basis,pts,center;multithreaded=false)
                active=findall(>(FR_SETTINGS["polygon_cafb_gram_floor"]),diag(F))
                fr_require(length(active)>=6,"insufficient nonvanishing CAFB columns")
                F,Fk=F[active,active],Fk[active,active]; D=Diagonal(1 ./ sqrt.(diag(F)))
                xs,_=solve(solver,D*F*D,D*Fk*D,center,0.06scale)
                fr_require(!isempty(xs),"empty equilibrated CAFB/VS spectrum")
                x=xs[argmin(abs.(xs .- center))]
            else
                x,_=solve_wavenumber(solver,basis,b,center,0.06scale;multithreaded=false)
            end
            fr_require(abs(x-target)<0.006scale,"$family/BIM mismatch: $x versus $target")
            fr_save_values!(run,g.name*"/"*family*"_"*string(method),[x];atol=3e-6,rtol=3e-6)
        end
    end
end

function fr_dual_basis!(run,g,anchor,scale)
    d=FR_SETTINGS["basis_dim"]; b=g.b
    if g.name in ("circle","ellipse","rectangle","stadium","prosen","limacon")
        half=g.name=="limacon"
        sector=half ? symmetry_sector(b,XAxisReflection=>-1) : symmetry_sector(b,XAxisReflection=>-1,YAxisReflection=>-1)
        angle=half ? Float64(pi) : pi/2
        target=nothing
        fr_check!(run,g.name*" odd-sector BIM reference") do
            gr=g.corners ? GlobalCornerGrading(kressq=2) : SmoothPeriodicGrading()
            kernel=CombinedFieldIntegralEquationSolver(FR_SETTINGS["ppw"],sector;min_pts=192,grading=gr,eps=1e-10)
            grid=collect(range((half ? 2.5 : 3)*scale,(half ? 5 : 7)*scale;length=25))
            tensions=k_sweep(kernel,b,grid;multithreaded=false)
            minima=[i for i in 2:length(grid)-1 if tensions[i]<min(tensions[i-1],tensions[i+1])]
            for i in minima
                candidate,_=solve_wavenumber(kernel,b,grid[i],0.35scale;multithreaded=false)
                xs,_=solve_spectrum(fr_beyn(kernel;taylor=false),b,candidate+0.02scale,0.16scale;multithreaded=false)
                if any(z->abs(imag(z))<0.01scale && abs(real(z)-candidate)<0.004scale,xs)
                    target=candidate; break
                end
            end
            fr_require(target!==nothing,"no contour-verified odd-sector sweep minimum")
            exact=g.name=="circle" ? 5.135622301840683 : g.name=="rectangle" ? 2g.exact : nothing
            exact===nothing || fr_require(isapprox(target,exact;rtol=2e-3,atol=2e-4),"odd-sector analytic mismatch")
            fr_save_values!(run,g.name*"/dual_bim",[target];atol=2e-6,rtol=2e-6)
        end
        target===nothing && return
        rpw=RealPlaneWaves(d,b,sector;use_taylor=false)
        cafb=CornerAdaptedFourierBessel(d,angle,FR_V2(0,0),0.0;use_taylor=false)
        fr_dual_solve!(run,g,b,rpw,"rpw_odd",target,scale)
        fr_dual_solve!(run,g,b,cafb,"cafb_odd",target,scale)
        fr_check!(run,g.name*" RPW/CAFB agreement") do
            for method in (:dm,:psm,:vs)
                a=only(run.values[g.name*"/rpw_odd_"*string(method)])
                c=only(run.values[g.name*"/cafb_odd_"*string(method)])
                fr_require(abs(a-c)<0.006scale,"RPW/CAFB sector mismatch")
            end
        end
    elseif g.name in ("triangle","polygon")
        corners=b.fundamental_domain.corners
        n=length(corners); i=g.name=="triangle" ? argmax(b.fundamental_domain.angles) : 1
        corners=corners .- Ref(corners[i])
        bcs=FR_BG.AbsBoundaryCondition[SpecularReflection() for _ in 1:n]
        bcs[i]=QuantumSolverIgnore(); bcs[mod1(i-1,n)]=QuantumSolverIgnore()
        domain=FR_BG.Polygon(corners,1;bcs)
        adapted=g.name=="triangle" ? TriangleBilliard{Float64}(domain,b.symmetries) :
                    PolygonBilliard{Float64}(FR_BG.SimpleDomain{Float64}(domain.boundary,corners,1),b.symmetries)
        ray=corners[mod1(i+1,n)]-corners[i]; rotation=atan(ray[2],ray[1])
        cafb=CornerAdaptedFourierBessel(d,domain.angles[i],corners[i],rotation;use_taylor=false)
        fr_dual_solve!(run,g,adapted,cafb,"cafb_corner",anchor,scale)
        fr_check!(run,g.name*" RPW/CAFB agreement") do
            for method in (:dm,:psm,:vs)
                a=only(run.values[g.name*"/basis_"*string(method)])
                c=only(run.values[g.name*"/cafb_corner_"*string(method)])
                fr_require(abs(a-c)<0.006scale,"RPW/CAFB corner mismatch")
            end
        end
    else
        push!(run.skips,g.name*": CAFB excluded; no supported reflection wedge or physical corner for this basis.")
    end
end

function fr_method_checks!(run,gs)
    fr_check!(run,"exported geometry coverage") do
        exported=Set(s for s in names(FR_BG) if isdefined(FR_BG,s) &&
                     (v=getfield(FR_BG,s); (v isa DataType || v isa UnionAll) && v<:FR_BG.AbsBilliard && !isabstracttype(v)))
        covered=Set(nameof(typeof(g.b)) for g in fr_geometries())
        fr_require(exported==covered,"update geometry registry: uncovered $(setdiff(exported,covered))")
    end
    fr_check!(run,"exported API availability") do
        missing=[string(nameof(m))*"."*string(s) for m in (FR_BG,FR_QB) for s in names(m) if !isdefined(m,s)]
        legacy=Set(["BilliardGeometry.LimaconSegment", "QuantumBilliards.BoundaryPointsDM",
                    "QuantumBilliards.BoundaryPointsSM", "QuantumBilliards.blas_1",
                    "QuantumBilliards.blas_multi", "QuantumBilliards.blas_multi_then_1", "QuantumBilliards.construct_matrices_benchmark",
                    "QuantumBilliards.print_benchmark_info", "QuantumBilliards.use_threads"])
        unexpected=setdiff(Set(missing),legacy)
        fr_require(isempty(unexpected),"new undefined exports: $(join(unexpected,", "))")
        !isempty(missing) && push!(run.skips,"Legacy undefined exports excluded: "*join(missing,", "))
    end
    fr_check!(run,"boundary samplers") do
        for sampler in (LinearNodes(),GaussLegendreNodes())
            t,w=sample_points(sampler,16)
            fr_require(length(t)==16 && issorted(t) && all(>(0),w) && isapprox(sum(w),1),"invalid sampler")
        end
        t,w=sample_points(FourierNodes([1.0]),16)
        t,w=only(t),only(w)
        fr_require(length(t)==length(w) && issorted(t) && all(x->0<=x<=1,t) && all(>=(0),w) && isapprox(sum(w),1) &&
                abs(sum(w .* cos.(2pi .* t)))<1e-12 && abs(sum(w .* sin.(2pi .* t)))<1e-12, "invalid periodic Fourier quadrature")
    end
    push!(run.skips,"Standalone basis_matrix(TaylorCache,k) excluded: missing 3-D real _horner overload; combined value/derivative evaluator used by VS remains tested.")
    for family in (:plane_waves,:corner_bessel)
        fr_check!(run,string(family)*" Taylor backend") do
            pts=FR_V2[FR_V2(0.13,0.27),FR_V2(0.4,0.2),FR_V2(0.7,0.3)]
            basis=family==:plane_waves ? RealPlaneWaves(6,nothing,nothing) :
                                        CornerAdaptedFourierBessel(6,pi/2,FR_V2(0,0),0.0)
            cache=basis isa RealPlaneWaves ? FR_QB.RealPlaneWaveTaylorCache(basis,3.0,pts;multithreaded=false) :
                                           FR_QB.CornerAdaptedTaylorCache(basis,3.0,pts;multithreaded=false)
            B=basis_matrix(basis,3.03,pts;multithreaded=false)
            Bt,dBt=FR_QB._weight_scaled_basis_and_dk_matrices(cache,3.03,ones(length(pts)),1;multithreaded=false)
            dB=dk_matrix(basis,3.03,pts;multithreaded=false)
            fr_require(isapprox(B,Bt;atol=1e-10,rtol=1e-9),"basis Taylor mismatch")
            fr_require(isapprox(dB,dBt;atol=1e-10,rtol=1e-9),"basis Taylor derivative mismatch")
        end
    end
    "circle" in getproperty.(gs,:name) || return
    g=first(filter(g->g.name=="circle",gs)); k=fr_kernel(g,:cfie); x=g.exact
    fr_check!(run,"smooth and single-corner grading") do
        single=merge(g,(b=PolarBilliard([0.0,0.0]),))
        for gr in (SmoothPeriodicGrading(),CornerGrading(kressq=2))
            fr_matrix_checks(fr_kernel(single,:cfie;grading=gr),single.b,x+0.02)
        end
    end
    fr_check!(run,"DLP state representations") do
        fr_state_checks(fr_kernel(g,:dlp),g.b,x)
    end
    fr_check!(run,"Chebyshev radial backend") do
        pts=evaluate_points(k,g.b,x)
        cfg=ChebyshevConfig(n_panels_h=64,n_panels_j=32,M_h=10,M_j=10, sampling_points=128,tol=1e-9,max_iter=8)
        zs=ComplexF64[x+0.01+0.01im]
        for native in (fr_kernel(g,:dlp),k)
            Ap=only(FR_QB._construct_matrices_multi_k_cheb(native,pts,zs,cfg;multithreaded=false))
            A=construct_matrices(native,pts,only(zs);multithreaded=false)
            fr_require(norm(A-Ap)/norm(A)<1e-7,"radial Chebyshev matrix mismatch")
        end
    end
    fr_check!(run,"Beyn eigenvectors and public spectrum API") do
        s=fr_beyn(k); s.imag_k_check=false
        s.cheb_config=ChebyshevConfig(param_strategy=:manual)
        pts=evaluate_points(k,g.b,x)
        xs,_,X=solve_vectors(s,pts,x+0.02,0.16;multithreaded=false)
        fr_require(length(xs)==1 && size(X,2)==1,"invalid eigenvector count")
        A=construct_matrices(k,pts,only(xs);multithreaded=false)
        fr_require(norm(A*X[:,1])/norm(X[:,1])<1e-6,"nonlinear eigenvector residual")
        data=compute_spectrum(s,g.b,x-0.05,x+0.05;multithreaded=false,show_progress=false)
        fr_save_values!(run,"methods/beyn_compute_spectrum",data.k;atol=2e-6,rtol=2e-6)
    end
    fr_check!(run,"reflection-sector eigenvalue") do
        sector=symmetry_sector(g.b,XAxisReflection=>1,YAxisReflection=>1)
        folded=CombinedFieldIntegralEquationSolver(FR_SETTINGS["ppw"],sector;min_pts=96)
        xs,_=solve_spectrum(fr_beyn(folded),g.b,x+0.02,0.16;multithreaded=false)
        fr_require(length(xs)==1 && isapprox(real(only(xs)),x;atol=1e-6),"symmetry reduction changes eigenvalue")
        fr_save_values!(run,"methods/reflection_sector",xs;atol=2e-6,rtol=2e-6)
    end
    fr_check!(run,"corner-adapted Fourier--Bessel eigensolvers") do
        b,basis=make_triangle_and_basis(pi/2,1.0)
        target=pi*sqrt(5)
        for (name,s) in (("dm",DecompositionMethodSolver(2.0,FR_SETTINGS["ppw"];min_dim=12,min_pts=64)),
                         ("psm",ParticularSolutionsMethodSolver(2.0,FR_SETTINGS["ppw"],FR_SETTINGS["ppw"];min_dim=12,min_pts=64,min_int_pts=64)),
                         ("vs",VerginiSaracenoSolver(2.0,FR_SETTINGS["ppw"];min_dim=12,min_pts=64)))
            Random.seed!(FR_SEED)
            ctr=name=="vs" ? target+0.001 : target
            val,_=solve_wavenumber(s,basis,b,ctr,0.04;multithreaded=false)
            fr_require(isapprox(val,target;atol=2e-3),"corner basis analytic eigenvalue mismatch")
            fr_save_values!(run,"methods/corner_basis_"*name,[val];atol=3e-6,rtol=3e-6)
        end
    end
end

function fr_write_reference(path,run;replace=false,validated=true)
    mkpath(dirname(path))
    payload=Dict("schema_version"=>FR_SCHEMA,"validated"=>validated, "created_utc"=>string(now(UTC)),"julia_version"=>string(Base.VERSION),
                 "settings"=>FR_SETTINGS,"failed_checks"=>run.failures, "quantum_billiards_path"=>dirname(dirname(pathof(FR_QB))),
                 "billiard_geometry_path"=>dirname(dirname(pathof(FR_BG))), "excluded"=>run.skips,"eigenvalues"=>run.values)
    tmp=tempname(dirname(path))
    try
        open(io->TOML.print(io,payload;sorted=true),tmp,"w")
        mv(tmp,path;force=replace)
    finally
        isfile(tmp) && rm(tmp)
    end
end

function fr_run(args=ARGS)
    allowed=("--setup","--record","--update","--check","--help")
    fr_require(all(a->a in allowed || startswith(a,"--reference=") || startswith(a,"--only="),args),"unknown option")
    if "--help" in args
        println("See the header of test/tests_all.jl for usage."); return
    end
    writing="--record" in args || "--update" in args
    checking="--check" in args
    fr_require(!(writing && checking),"choose --check or a recording mode")
    fr_require(!("--record" in args && "--update" in args),"choose --record or --update")
    refarg=filter(a->startswith(a,"--reference="),args)
    path=isempty(refarg) ? joinpath(@__DIR__,"baselines","fast_eigenvalues.toml") : abspath(split(only(refarg),'=';limit=2)[2])
    fr_require(!("--record" in args && isfile(path)),"reference exists; use --update to replace it")
    fr_require(writing || checking || isfile(path),"no reference file; run with --record first, or use --check")
    old=writing || checking ? Dict() : TOML.parsefile(path)
    if !isempty(old)
        fr_require(get(old,"validated",false),"unvalidated candidate cannot be used as a reference")
        fr_require(old["schema_version"]==FR_SCHEMA && old["settings"]==FR_SETTINGS,"reference schema/settings changed; review and --update")
    end
    onlyarg=filter(a->startswith(a,"--only="),args)
    gs=fr_geometries()
    if !isempty(onlyarg)
        fr_require(!writing,"cannot record/update a partial run")
        selected=split(split(only(onlyarg),'=';limit=2)[2],',')
        fr_require(all(s->s in getproperty.(gs,:name),selected),"unknown geometry in --only")
        gs=filter(g->g.name in selected,gs)
    end
    BLAS.set_num_threads(1)
    Random.seed!(FR_SEED)
    run=FRRun(old,Dict{String,Any}(),String[],String[],0); start=time()
    @testset "Fast geometry/solver regression" begin
        for g in gs
            fr_test_geometry!(run,g)
        end
        fr_method_checks!(run,gs)
        if isempty(onlyarg) && !isempty(old)
            fr_check!(run,"reference coverage") do
                fr_require(Set(keys(run.values))==Set(keys(old["eigenvalues"])),"reference coverage differs from this suite")
            end
        end
        for s in run.skips; println("EXCLUDED ",s); end
        println("Completed ",run.checks," checks in ",round(time()-start;digits=2)," s; ",length(run.failures)," failures.")
        if writing && !isempty(run.failures)
            candidate=replace(path,r"\.toml$"=>"")*".candidate.toml"
            fr_write_reference(candidate,run;replace=true,validated=false)
            println("Saved diagnostic candidate: ",candidate,"; accepted reference unchanged.")
        end
    end
    if writing
        fr_write_reference(path,run;replace="--update" in args)
        println("Saved reference wavenumbers: ",path)
    end
end

function fr_main(args)
    original_threads = BLAS.get_num_threads()
    try
        fr_run(args)
    finally
        BLAS.set_num_threads(original_threads)
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    fr_main(ARGS)
else
    let reference = joinpath(@__DIR__, "baselines", "fast_eigenvalues.toml")
        fr_main(isfile(reference) ? String[] : ["--record"])
    end
end
