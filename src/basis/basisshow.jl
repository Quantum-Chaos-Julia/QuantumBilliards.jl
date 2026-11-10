################################################################################
# BASIS DISPLAY
#
# `show` for basis types in the style of the solver display (see
# `solverlogging.jl`): thematic sections, one `key = value` row per field.
# Taylor settings are omitted when `use_taylor = false`; the flag is always shown.
################################################################################

function _write_basis_symmetry!(io::IO, ind::Int, symmetries, characters)
    _sec!(io, ind, "Symmetry")
    if isnothing(symmetries) || isempty(symmetries)
        _ln!(io, ind + 2, "none")
        return
    end
    for (i, g) in enumerate(symmetries)
        if isnothing(characters)
            _ln!(io, ind + 2, _symmetry_label(g))
        else
            _ln!(io, ind + 2, _symmetry_label(g), " => ", _character_description(g, characters[i]))
        end
    end
end

function _write_basis_taylor!(io::IO, ind::Int, basis)
    _sec!(io, ind, "Taylor expansion")
    _row!(io, ind + 2, "use_taylor", basis.use_taylor)
    basis.use_taylor || return
    _row!(io, ind + 2, "taylor_degree", basis.taylor_degree)
    _row!(io, ind + 2, "taylor_radius", basis.taylor_radius)
    _row!(io, ind + 2, "taylor_tol", basis.taylor_tol)
end

function _write_basis_geometry!(io::IO, ind::Int, basis)
    _sec!(io, ind, "Geometry")
    _row!(io, ind + 2, "origin", Tuple(basis.origin))
    _row!(io, ind + 2, "rot_angle", basis.rot_angle)
end

function _write_basis_header!(io::IO, basis)
    print(io, nameof(typeof(basis)), ":\n")
    _row!(io, 2, "dim", basis.dim)
end

function Base.show(io::IO, ::MIME"text/plain", basis::FourierBessel)
    buf = IOBuffer()
    _write_basis_header!(buf, basis)
    _write_basis_geometry!(buf, 2, basis)
    _sec!(buf, 2, "Modes")
    _row!(buf, 4, "max_order", isempty(basis.orders) ? 0 : maximum(basis.orders))
    _write_basis_symmetry!(buf, 2, basis.symmetries, basis.sym_characters)
    _write_basis_taylor!(buf, 2, basis)
    print(io, rstrip(String(take!(buf))))
end

function Base.show(io::IO, ::MIME"text/plain", basis::CornerAdaptedFourierBessel)
    buf = IOBuffer()
    _write_basis_header!(buf, basis)
    _write_basis_geometry!(buf, 2, basis)
    _sec!(buf, 2, "Corner")
    _row!(buf, 4, "corner_angle", basis.corner_angle)
    _row!(buf, 4, "nu", basis.nu)
    _row!(buf, 4, "rotation_angle_discontinuity", basis.rotation_angle_discontinuity)
    _write_basis_symmetry!(buf, 2, basis.symmetries, nothing)
    _write_basis_taylor!(buf, 2, basis)
    print(io, rstrip(String(take!(buf))))
end

function Base.show(io::IO, ::MIME"text/plain", basis::RealPlaneWaves)
    buf = IOBuffer()
    _write_basis_header!(buf, basis)
    _sec!(buf, 2, "Plane-wave directions")
    _row!(buf, 4, "angle_arc", basis.angle_arc)
    _row!(buf, 4, "angle_shift", basis.angle_shift)
    _rowstr!(buf, 4, "sampler", _describe(basis.sampler))
    _write_basis_symmetry!(buf, 2, basis.symmetries, basis.sym_characters)
    _write_basis_taylor!(buf, 2, basis)
    print(io, rstrip(String(take!(buf))))
end
