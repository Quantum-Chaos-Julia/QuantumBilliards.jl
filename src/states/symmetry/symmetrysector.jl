################################################################################
############################ SYMMETRY SECTOR ###################################
################################################################################
# `SymmetrySector` is Layer 2 of the symmetry-framework split (Step 15 of the
# migration plan): the one-dimensional irreducible-representation (parity /
# rotation-sector) choice a *specific solve* targets, kept separate from a
# billiard's fixed geometric symmetry group (`BilliardGeometry.AbsSymmetry`,
# `billiard.symmetries::BilliardGeometry.SymmetryRegistry`). A `SymmetrySector`
# is always built *against* a concrete billiard, so a solve can never silently
# request a representation for a symmetry the billiard doesn't actually have.
################################################################################

"""
SymmetrySector{Bi<:BilliardGeometry.AbsBilliard}

`SymmetrySector` represents the irreducible-representation character chosen,
per registered symmetry generator (`sym_id`), for a specific eigenstate
solve — the Layer-2 counterpart of a billiard's fixed geometric symmetry
group (`billiard.symmetries::BilliardGeometry.SymmetryRegistry`).

## Attributes
* `billiard::Bi`: The billiard this sector's `sym_id`s were resolved against.
* `characters::Dict{Int,ComplexF64}`: `sym_id => chosen irrep character`, for every generator this sector constrains.

## API
The following functions can be evaluated for this type:
- [`symmetry_sector`](@ref)
"""
struct SymmetrySector{Bi<:BilliardGeometry.AbsBilliard}
    billiard::Bi
    characters::Dict{Int,ComplexF64}
end

# Validates and converts a requested representation choice for `sym` into its
# one-dimensional irrep character: `±1` for any reflection generator, or
# `exp(2πi*sector*m/N)` for the `m`-th image of an `N`-fold rotation.
function _sector_character(::BilliardGeometry.AbsReflection, val::Integer)
    abs(val) == 1 || throw(ArgumentError("Reflection character must be ±1; received $val"))
    return ComplexF64(val)
end
function _sector_character(sym::BilliardGeometry.NFoldRotation, sector::Integer)
    (0 <= sector < sym.order) || throw(ArgumentError("Rotation sector must be in 0:$(sym.order-1); received $sector"))
    return cis(2*pi*sector*sym.m/sym.order)
end

"""
    symmetry_sector(billiard::BilliardGeometry.AbsBilliard, choices::Pair...) → sector::SymmetrySector

Builds a [`SymmetrySector`](@ref) for `billiard`, resolving each
`GeneratorType => value` pair (e.g. `BilliardGeometry.XAxisReflection => -1`,
`BilliardGeometry.NFoldRotation => 2`) against `billiard.symmetries`.

## Description
Every registered generator in `billiard.symmetries` whose type matches
`GeneratorType` is assigned the requested character (validated as a genuine
one-dimensional irrep value for that generator type — `±1` for a reflection,
an integer rotation sector in `0:N-1` for an `N`-fold rotation, converted to
`exp(2πi*sector*m/N)` for each registered rotation image `m`). Requesting a
generator type the billiard has no registered symmetry for raises an
`ArgumentError` at construction time, instead of silently producing a
mismatched or empty representation later.

## Arguments
* `billiard`: The billiard the requested symmetry sector is resolved against.
* `choices`: `GeneratorType => value` pairs selecting the representation for each active generator type.

## Returns
* `sector`: A [`SymmetrySector`](@ref) usable by [`RealPlaneWaves`](@ref)`(dim, billiard, sector)`.
"""
function symmetry_sector(billiard::BilliardGeometry.AbsBilliard, choices::Pair...)
    characters = Dict{Int,ComplexF64}()
    for (GenType, val) in choices
        matches = filter(s -> s isa GenType, billiard.symmetries)
        isempty(matches) && throw(ArgumentError("Billiard has no registered $(GenType) symmetry generator"))
        for sym in matches
            characters[sym.sym_id] = _sector_character(sym, val)
        end
    end
    return SymmetrySector(billiard, characters)
end
