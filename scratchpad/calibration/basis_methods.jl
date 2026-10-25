using Test, QuantumBilliards, BilliardGeometry
include(joinpath(@__DIR__, "..", "..", "test", "integration", "common.jl"))

function probe(name, f; kref=nothing)
    println("== ", name)
    try
        t = @elapsed r = f()
        println("   time=", round(t, digits=2), "  result=", r)
        kref === nothing || println("   |dk|=", abs(r[1] - kref), "  ten=", r[2])
    catch e
        println("   FAILED: ", sprint(showerror, e)[1:min(end, 300)])
    end
end

# analytic Veech-independent: right isosceles triangle, legs 1, level pi*sqrt(5)
tri, tbasis = QuantumBilliards.make_triangle_and_basis(pi/2, 1.0)
kt = pi * sqrt(5)

probe("DM CAFB triangle (2.0, 20.0; min_dim=12,min_pts=64)", () ->
    solve_wavenumber(DecompositionMethodSolver(2.0, 20.0; min_dim=12, min_pts=64), tbasis, tri, kt, 0.04; multithreaded=false); kref=kt)
probe("DM CAFB triangle (2.0, 5.0)", () ->
    solve_wavenumber(DecompositionMethodSolver(2.0, 5.0), tbasis, tri, kt, 0.04; multithreaded=false); kref=kt)
probe("PSM CAFB triangle", () ->
    solve_wavenumber(ParticularSolutionsMethodSolver(2.0, 20.0, 20.0; min_dim=12, min_pts=64, min_int_pts=64), tbasis, tri, kt, 0.04; multithreaded=false); kref=kt)

circ = CircleBilliard(1.0)
fb = FourierBessel(20; sym_x=1, sym_y=1)
probe("PSM FourierBessel circle j01 even-even", () ->
    solve_wavenumber(ParticularSolutionsMethodSolver(2.0, 20.0, 20.0; min_dim=20, min_pts=100, min_int_pts=100), fb, circ, 2.45, 0.1; multithreaded=false); kref=2.404825557695773)
probe("DM FourierBessel circle j01 even-even", () ->
    solve_wavenumber(DecompositionMethodSolver(2.0, 20.0; min_dim=20, min_pts=100), fb, circ, 2.45, 0.1; multithreaded=false); kref=2.404825557695773)

rect = RectangleBilliard(1.0, 0.7)
kr = rectangle_spectrum(2.0, 1.4, 0.0, 4.0)[1]
rpw = RealPlaneWaves(20; sym_x=1, sym_y=1)
probe("DM RPW rect even-even ground kref=$kr", () ->
    solve_wavenumber(DecompositionMethodSolver(2.0, 20.0; min_dim=20, min_pts=100), rpw, rect, kr + 0.05, 0.1; multithreaded=false); kref=kr)
probe("PSM RPW rect even-even ground", () ->
    solve_wavenumber(ParticularSolutionsMethodSolver(2.0, 20.0, 20.0; min_dim=20, min_pts=100, min_int_pts=100), rpw, rect, kr + 0.05, 0.1; multithreaded=false); kref=kr)

# VS Veech triangle
vtri, vbasis = QuantumBilliards.make_veech_right_triangle_and_basis(5)
probe("VS CAFB Veech triangle low k0=110", () ->
    solve_wavenumber(VerginiSaracenoSolver(2.0, 5.0), vbasis, vtri, 110.0, 0.1; multithreaded=false); kref=110.0894534647971)
probe("VS CAFB Veech triangle low spectrum", () ->
    solve_spectrum(VerginiSaracenoSolver(2.0, 5.0), vbasis, vtri, 110.0, 0.1; multithreaded=false))
probe("DM CAFB Veech triangle ground k0=6.1", () ->
    solve_wavenumber(DecompositionMethodSolver(2.0, 5.0), vbasis, vtri, 6.1, 0.1; multithreaded=false); kref=6.065082959967892)
probe("PSM CAFB Veech triangle ground k0=6.1", () ->
    solve_wavenumber(ParticularSolutionsMethodSolver(2.0, 5.0, 2.0), vbasis, vtri, 6.1, 0.1; multithreaded=false); kref=6.065082959967892)

# VS stadium odd-odd
stad = StadiumBilliard(0.5)
sb = RealPlaneWaves(12; sym_x=-1, sym_y=-1)
probe("VS RPW stadium odd-odd low k0=110", () ->
    solve_spectrum(VerginiSaracenoSolver(5.0, 10.0), sb, stad, 110.0, 0.05; multithreaded=false))
