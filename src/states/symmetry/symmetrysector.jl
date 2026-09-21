################################################################################
# SYMMETRY SECTORS AND BIM SYMMETRY RESOLUTION
#
# This file defines the representation sector selected for a quantum-billiard
# solve and translates that choice into the symmetry representation used by
# boundary-integral solvers.
#
# A billiard stores its geometric symmetry generators independently of any
# particular eigenstate calculation. `SymmetrySector` specifies a
# one-dimensional irreducible-representation character for selected registered
# generators:
#
#                       χ(g) = ±1
#
# for reflections, and
#
#                  χ_s(g^m) = exp(2πi s m/N)
#
# for sector s of an N-fold cyclic rotation. Sectors are constructed against a
# concrete billiard so that every requested generator is validated against its
# registered geometric symmetries.
#
# BIM solvers require this representation information in the lower-level form
#
#                    (symmetry, character).
#
# Reflection sectors are resolved through `CompositeReflection`, which builds
# the corresponding reflection-group closure and character tuple. Rotation
# sectors are resolved by recovering the common cyclic sector from the
# characters of the registered rotation elements. Mixed reflection/rotation
# sectors are rejected because the BIM symmetry reduction currently supports
# reflection and cyclic sectors separately, but not general dihedral
# representations.
#
# Since symmetry IDs are local to a billiard's symmetry registry, a
# `SymmetrySector` may only be resolved against the exact billiard instance
# from which it was constructed.
################################################################################

"""
    SymmetrySector{Bi<:BilliardGeometry.AbsBilliard}

Representation sector for a specific quantum-billiard solve.

A `SymmetrySector` associates selected symmetry generators registered by the
billiard with the one-dimensional irreducible-representation characters used
for the solve. The geometric symmetry group remains a property of the billiard;
the sector specifies which representation of that group is selected.

## Attributes
* `billiard::Bi`: Billiard against which the symmetry generators were resolved.
* `characters::Dict{Int,ComplexF64}`: Map from registered symmetry IDs to their selected irreducible-representation characters.

## API
* [`symmetry_sector`](@ref)
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

Construct a [`SymmetrySector`](@ref) for `billiard` from representation choices
of the form `GeneratorType => value`.

Every registered symmetry whose type matches `GeneratorType` is assigned the
corresponding one-dimensional irreducible-representation character. Reflection
values must be `±1`. For an `N`-fold rotation, an integer sector `s ∈ 0:N-1`
assigns

    χ_s(g^m) = exp(2πi s m/N)

to each registered rotation element `g^m`. Requesting a generator type not
registered by the billiard raises an `ArgumentError`.

## Arguments
* `billiard::BilliardGeometry.AbsBilliard`: Billiard whose registered symmetries define the available generators.
* `choices::Pair...`: `GeneratorType => value` pairs selecting the representation sector.

## Returns
* `sector::SymmetrySector`: Representation sector associated with `billiard`.
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

# Verify that a sector is being resolved against the exact billiard instance
# from which it was constructed. Symmetry IDs are assigned locally by each
# billiard's symmetry registry, so the same integer ID may refer to unrelated
# generators for two distinct billiard instances.
function _check_sector_billiard(billiard::BilliardGeometry.AbsBilliard, sector::SymmetrySector)
    sector.billiard === billiard || throw(ArgumentError(
        "SymmetrySector was built against a different billiard instance than " *
        "the one it is now being resolved against. sym_id assignment is " *
        "per-billiard registration order, so a sector built for one billiard " *
        "cannot safely be reused against another, even a geometrically " *
        "identical one. Rebuild the sector with symmetry_sector(billiard, ...) " *
        "using this exact billiard object."))
    return nothing
end

# Resolve a sector into the `(symmetry, character)` representation stored by a
# BIM solver. The sector is first checked against the exact billiard instance
# from which it was constructed, since symmetry IDs are local to each billiard's
# registry. An empty sector gives `(nothing, ())`; otherwise the stored symmetry
# IDs are resolved to their registered generators and dispatched to the
# reflection or cyclic-rotation resolution path.
function _resolve_bim_symmetry(billiard::BilliardGeometry.AbsBilliard, sector::SymmetrySector)
    _check_sector_billiard(billiard, sector)
    isempty(sector.characters) && return nothing, ()
    gens = Pair{BilliardGeometry.AbsSymmetry,ComplexF64}[BilliardGeometry.symmetry_of(billiard.symmetries, id) => char for (id, char) in sector.characters]
    return _resolve_bim_symmetry(gens)
end

# Dispatch resolved generator-character pairs to the supported BIM symmetry
# reductions. Pure reflection families use the composite-reflection path and
# pure cyclic-rotation families use the rotation-sector path. Mixed families
# require general dihedral representation handling, which is not implemented.
function _resolve_bim_symmetry(gens::Vector{Pair{BilliardGeometry.AbsSymmetry,ComplexF64}})
    if all(g -> g.first isa BilliardGeometry.AbsReflection, gens)
        return _resolve_bim_reflection_sector(gens)
    elseif all(g -> g.first isa BilliardGeometry.NFoldRotation, gens)
        return _resolve_bim_rotation_sector(gens)
    else
        throw(ArgumentError("A SymmetrySector mixing reflection and rotation generators cannot be resolved into a BIM solver symmetry: no dihedral (2D-irrep) representation machinery exists in this package. Received generator types $(unique(typeof(g.first) for g in gens))."))
    end
end

# Resolve a family of reflection characters through `CompositeReflection`,
# which constructs the closure of the generated reflection group and validates
# character consistency. A lone `XYAxisReflection` is rejected because its
# character does not uniquely determine the two independent D2 reflection
# characters χx and χy.
function _resolve_bim_reflection_sector(gens::Vector{Pair{BilliardGeometry.AbsSymmetry,ComplexF64}})
    if length(gens) == 1 && gens[1].first isa BilliardGeometry.XYAxisReflection
        throw(ArgumentError(
            "A lone XYAxisReflection character does not determine a unique " *
            "1D representation of the registered D2 symmetry group (both " *
            "(χx,χy)=(+1,-1) and (-1,+1) give the same combined character). " *
            "Specify the individual axis reflections instead, e.g. " *
            "symmetry_sector(billiard, XAxisReflection=>χx, YAxisReflection=>χy)."))
    end
    generators = BilliardGeometry.AbsReflection[g.first for g in gens]
    characters = ComplexF64[g.second for g in gens]
    return BilliardGeometry.CompositeReflection(generators), Tuple(characters)
end

# Recover the cyclic sector `s` whose character on the registered element g^m
# equals `char`. The finite set s=0,...,N-1 is searched directly because the
# phase is defined modulo 2π; dividing the principal argument by `m` does not
# in general invert the map s ↦ sm mod N.
function _match_rotation_sector(gen::BilliardGeometry.NFoldRotation, char::ComplexF64)
    order = gen.order
    for s in 0:order-1
        isapprox(cis(2*pi*s*gen.m/order), char; atol=1e-9) && return s
    end
    throw(ArgumentError("Character $char is not a valid NFoldRotation irrep value for m=$(gen.m), order=$order"))
end

# Resolve several registered elements of one cyclic rotation group to a common
# sector index. All elements must have the same group order and independently
# recover the same sector, ensuring that their characters define one consistent
# one-dimensional representation.
function _resolve_bim_rotation_sector(gens::Vector{Pair{BilliardGeometry.AbsSymmetry,ComplexF64}})
    order = gens[1].first.order
    all(g -> g.first.order == order, gens) || throw(ArgumentError("Mismatched NFoldRotation orders within one SymmetrySector"))
    sectors = [_match_rotation_sector(gen, char) for (gen, char) in gens]
    all(==(sectors[1]), sectors) || throw(ArgumentError("Inconsistent rotation sector recovered across sym_ids: $sectors"))
    return gens[1].first, (sectors[1],)
end
