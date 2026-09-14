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

################################################################################
##################### BIM SOLVER RESOLUTION (Step 16) #########################
################################################################################
# Resolves a SymmetrySector (Layer 2: per-solve representation choice) into
# the `(symmetry::AbsSymmetry, character::Tuple)` pair a BIM solver
# (DoubleLayerPotentialSolver/CombinedFieldIntegralEquationSolver) actually
# stores, using each matched sym_id's own registered generator
# (`BilliardGeometry.symmetry_of`) rather than assuming a fixed generator
# count or shape. Reflection generators are always folded via a freshly
# built `CompositeReflection`, reusing BilliardGeometry.jl's existing
# group-closure algorithm (a lone generator or several independent axes are
# both handled uniformly, with automatic consistency validation for
# over-specified requests); NFoldRotation generators recover a single
# integer sector, cross-checked across every matched sym_id using that
# entry's own `m` (never assuming `m=1`).
################################################################################

"""
    _resolve_bim_symmetry(billiard::BilliardGeometry.AbsBilliard, sector::SymmetrySector) → (symmetry, character::Tuple)

Resolves `sector`'s per-`sym_id` characters, looked up against `billiard`'s
own [`BilliardGeometry.SymmetryRegistry`](@ref) (`BilliardGeometry.symmetry_of`,
which throws if `sector` names a `sym_id` `billiard` never registered — this
is what closes the "sector built against the wrong billiard" gap), into the
bare `(symmetry::BilliardGeometry.AbsSymmetry, character::Tuple)` pair
consumed by [`DoubleLayerPotentialSolver`](@ref)/
[`CombinedFieldIntegralEquationSolver`](@ref)'s unvalidated constructor
keywords.

An empty `sector` (no characters requested) resolves to the trivial
representation, `(nothing, ())`.
"""
function _resolve_bim_symmetry(billiard::BilliardGeometry.AbsBilliard, sector::SymmetrySector)
    isempty(sector.characters) && return nothing, ()
    gens = Pair{BilliardGeometry.AbsSymmetry,ComplexF64}[BilliardGeometry.symmetry_of(billiard.symmetries, id) => char for (id, char) in sector.characters]
    return _resolve_bim_symmetry(gens)
end

function _resolve_bim_symmetry(gens::Vector{Pair{BilliardGeometry.AbsSymmetry,ComplexF64}})
    if all(g -> g.first isa BilliardGeometry.AbsReflection, gens)
        return _resolve_bim_reflection_sector(gens)
    elseif all(g -> g.first isa BilliardGeometry.NFoldRotation, gens)
        return _resolve_bim_rotation_sector(gens)
    else
        throw(ArgumentError("A SymmetrySector mixing reflection and rotation generators cannot be resolved into a BIM solver symmetry: no dihedral (2D-irrep) representation machinery exists in this package. Received generator types $(unique(typeof(g.first) for g in gens))."))
    end
end

# Reflection family: always fold via the generated group's full closure,
# reusing BilliardGeometry.jl's existing CompositeReflection algorithm
# (handles 1 generator, 2 independent generators i.e. a full D2 fold, or any
# larger combination, uniformly and with automatic consistency validation).
# The one irreducible exception is a lone XYAxisReflection character, which
# cannot determine a unique 1D representation of the D2 group on its own
# (both (χx,χy)=(+1,-1) and (-1,+1) give the same combined character).
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

# NFoldRotation family: Cn_symmetry(n) registers n-1 non-identity elements of
# the *same* cyclic group (m=1:n-1), so a "pick sector s" choice legitimately
# produces several matched sym_ids at once. Recover s from each matched
# entry using *that entry's own* m and cross-check all matched entries
# agree, instead of assuming m=1 for whichever entry happens to be
# inspected first.
#
# Matching is done by brute-force search over the (tiny, order<=~12 in
# practice) candidate sector values rather than inverting
# `character = cis(2π*s*m/order)` by real-valued division: dividing the
# *wrapped* angle by `m` does not correctly invert "multiply by m mod
# order" whenever `s*m/order >= 1` causes the angle to wrap (confirmed by
# direct testing on C3Billiard's sector=2, m=2 entry, which the naive
# division recovers as the non-integer 0.5, not the correct s=2).
function _match_rotation_sector(gen::BilliardGeometry.NFoldRotation, char::ComplexF64)
    order = gen.order
    for s in 0:order-1
        isapprox(cis(2*pi*s*gen.m/order), char; atol=1e-9) && return s
    end
    throw(ArgumentError("Character $char is not a valid NFoldRotation irrep value for m=$(gen.m), order=$order"))
end

function _resolve_bim_rotation_sector(gens::Vector{Pair{BilliardGeometry.AbsSymmetry,ComplexF64}})
    order = gens[1].first.order
    all(g -> g.first.order == order, gens) || throw(ArgumentError("Mismatched NFoldRotation orders within one SymmetrySector"))
    sectors = [_match_rotation_sector(gen, char) for (gen, char) in gens]
    all(==(sectors[1]), sectors) || throw(ArgumentError("Inconsistent rotation sector recovered across sym_ids: $sectors"))
    return gens[1].first, (sectors[1],)
end
