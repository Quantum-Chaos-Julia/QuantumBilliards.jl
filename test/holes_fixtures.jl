# Shared fixtures for the symmetric-holes tests (unit and integration tier). Not a test file: helpers and a minimal billiard type
# only, no @testset. Included by runtests.jl before either tier, so the `hf_` helpers are available everywhere. Copied (reduced) from
# BilliardGeometry.jl/test/unit/holes_fixtures.jl; keep the two in step.
#
# Fixtures (all in the quarter-square billiard [0,1]^2 of RectangleBilliard(1.0, 1.0) with the D2 registry, full frame [-1,1]^2):
#   hf_symmetrized_disc_billiard()  SymmetrizedHoles(DomainHole(disc of radius 0.1 about (0.4, 0.2), domain id 5), D2_symmetry()):
#                                   G' = D2, four hole loops in one boundary group (group 1), area 4 - 4 pi 0.01
#   hf_quarter_disc_billiard()      SymmetricHole of the disc of radius 0.1 about (0.5, 0) built from a quarter-disc piece with
#                                   SymmetryWall(1, 2) and SymmetryWall(3, 4) and D2_symmetry() about (0.5, 0): one loop, G' = {X}
#   hf_domain_hole_billiard()       DomainHole disc of radius 0.1 about (0.4, 0.2) in the same quarter square: G' = {e}

using StaticArrays: SVector

# Minimal billiard holding a MultiplyConnectedDomain and a symmetry registry.
struct hf_HoleBilliard{T} <: AbsBilliard where {T<:Real}
    fundamental_domain::MultiplyConnectedDomain{T}
    symmetries::SymmetryRegistry
end

# Full counterclockwise circle of radius R about (x0, y0) as a SimpleDomain with the given domain id (segment id 1).
function hf_circle_domain(R, x0, y0, id)
    T = typeof(float(R))
    crv = CircleSegment(R, T(2 * pi), zero(T), [x0, y0]; domain_id = id, segment_id = 1)
    return SimpleDomain{T}(AbsCurve[crv], SVector{2,T}[SVector(x0 + R, y0)], id)
end

# Outer region: the quarter square [0,1]^2 closed by the two symmetry walls of RectangleBilliard(1.0, 1.0).
hf_quarter_square_domain() = RectangleBilliard(1.0, 1.0).fundamental_domain

# Quarter disc of radius 0.1 about (0.5, 0) from angle 0 to pi/2 (domain id 2): SymmetryWall(1, 2) (Y) and SymmetryWall(3, 4) (X) of D2_symmetry().
hf_d2_quarter_disc() = CircleWedge(0.1, pi / 2, 0.0, 0.5, 0.0, 2; bcs = AbsBoundaryCondition[SpecularReflection(), SymmetryWall(1, 2), SymmetryWall(3, 4)])
hf_d2_symmetric_hole() = SymmetricHole(hf_d2_quarter_disc(), D2_symmetry(), SVector(0.5, 0.0))

# Four discs of radius 0.1 placed by D2_symmetry() (copies at (+-0.4, +-0.2)).
hf_symmetrized_disc() = SymmetrizedHoles(DomainHole(hf_circle_domain(0.1, 0.4, 0.2, 5)), D2_symmetry())

hf_billiard(hole) = hf_HoleBilliard{Float64}(MultiplyConnectedDomain(hf_quarter_square_domain(), [hole], 1), D2_symmetry())

hf_symmetrized_disc_billiard() = hf_billiard(hf_symmetrized_disc())
hf_quarter_disc_billiard() = hf_billiard(hf_d2_symmetric_hole())
hf_domain_hole_billiard() = hf_billiard(DomainHole(hf_circle_domain(0.1, 0.4, 0.2, 5)))
