"""
    adapt_basis(triangle::Triangle, i::Integer) -> Tuple{Real, SVector{2}, Real, Nothing}

Construct the local polar frame centered at the `i`-th corner of a triangle for use in corner-adapted basis functions.

# Arguments
- `triangle::Triangle`: The triangle geometry object.
- `i::Integer`: The index of the edge (1-based) for which to compute the adapted frame.

# Returns
- `angle::Real`: The internal angle at the selected corner.
- `origin::SVector{2}`: The position of the corner.
- `rot_angle::Real`: The rotation of the local polar axis, aligned with the edge leaving the corner.
- `symmetry::Nothing`: Placeholder for symmetry information (not used currently).
"""
function adapt_basis(triangle::T,i::Ti) where {T<:BilliardGeometry.TriangleBilliard,Ti<:Integer}
    N=3
    c=triangle.fundamental_domain.corners
    i0=mod1(i,N)
    i1=mod1(i+1,N)
    a=c[i1]-c[i0]
    rot_angle=atan(a[2],a[1])
    origin=c[i0]
    return triangle.fundamental_domain.angles[i0],origin,rot_angle,nothing
end

"""
    make_triangle_and_basis(gamma, chi; edge_i=1) -> Tuple{Triangle, CornerAdaptedFourierBessel}

Convenience function to create a triangle and construct a corner-adapted Fourier-Bessel basis at a selected edge.

# Arguments
- `gamma::Real`: Internal angle at the base corner (γ).
- `chi::Real`: Shape control parameter, defines ratio β/α.
- `edge_i::Integer=1`: Index of the real edge used to place and adapt the basis.

# Returns
- `tr::Triangle`: The constructed triangle object with virtual edges applied.
- `basis::CornerAdaptedFourierBessel`: A Fourier-Bessel basis adapted to the corner opposite edge `edge_i`.
"""
function make_triangle_and_basis(gamma,chi; edge_i=1)
    cor=TriangleBilliard(gamma,chi).fundamental_domain.corners
    x0,y0=cor[mod1(edge_i+2,3)]
    bcs = Vector{AbsBoundaryCondition}(undef,3)
    for i in eachindex(bcs)
        bcs[i] = QuantumSolverIgnore()
    end
    bcs[edge_i]= SpecularReflection() 
    tr=TriangleBilliard(gamma,chi;bcs,x0,y0)
    angle,origin,rot_angle,symmetry=adapt_basis(tr,edge_i+2)
    basis=CornerAdaptedFourierBessel(10,angle,origin,rot_angle,symmetry)
    return tr, basis 
end

function make_veech_right_triangle_and_basis(n; edge_i=1)
    if n < 4
        println("Order must be 4 or above.")
        print("Setting n to 4.")
        n = 4
    end
    chi = (n-2)/2
    gamma = pi/2.0
    return make_triangle_and_basis(gamma, chi; edge_i)
end

function make_veech_right_triangle(n)
    if n < 4
        println("Order must be 4 or above.")
        print("Setting n to 4.")
        n = 4
    end
    chi = (n-2)/2
    gamma = pi/2.0
    return TriangleBilliard(gamma, chi)
end