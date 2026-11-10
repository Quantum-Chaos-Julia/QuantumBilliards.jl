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
# A two-dimensional irrep E_k of a dihedral group D_d restricts to the cyclic
# subgroup as the sum of the sectors s = k and s = d-k. The BIM solve in the
# cyclic sector s = k therefore contains every E_k level once; each level is
# twofold degenerate in the full problem. `symmetry_sector(billiard; irrep2d = k)`
# selects that cyclic sector and records the degeneracy.
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
* `characters::Dict{Int,ComplexF64}`: Map from registered symmetry IDs to their selected irreducible-representation characters. For a two-dimensional irrep `E_k` these are the characters of the cyclic sector `s = k`.
* `degeneracy::Int`: Dimension of the selected irrep: `1`, or `2` for a dihedral `E_k` sector.

## API
* [`symmetry_sector`](@ref)
* [`symmetry_degeneracy`](@ref)
"""
struct SymmetrySector{Bi<:BilliardGeometry.AbsBilliard}
    billiard::Bi
    characters::Dict{Int,ComplexF64}
    degeneracy::Int
end
SymmetrySector(billiard::BilliardGeometry.AbsBilliard, characters::Dict{Int,ComplexF64}) = SymmetrySector(billiard, characters, 1)

"""
    symmetry_degeneracy(sector::SymmetrySector) → n::Int

Dimension of the irrep selected by `sector`; every level of the full problem in
this sector is `n`-fold degenerate.
"""
symmetry_degeneracy(sector::SymmetrySector) = sector.degeneracy

# Validates and converts a requested representation choice for `sym` into its
# one-dimensional irrep character: `±1` for any reflection generator, or
# `exp(2πi*sector*m/N)` for the `m`-th image of an `N`-fold rotation.
function _sector_character(::BilliardGeometry.AbsReflection, val::Integer, d::Integer = 0)
    abs(val) == 1 || throw(ArgumentError("Reflection character must be ±1; received $val"))
    return ComplexF64(val)
end
function _sector_character(sym::BilliardGeometry.NFoldRotation, sector::Integer, d::Integer = sym.order)
    (0 <= sector < d) || throw(ArgumentError("Rotation sector must be in 0:$(d-1); received $sector"))
    return cis(2*pi*sector*sym.m/sym.order)
end

# Order of the rotation subgroup of a symmetry group: the identity and every orientation-preserving element.
_rotation_subgroup_order(group) = 1 + count(g -> !BilliardGeometry.orientation_reversing(g), group)

"""
    symmetry_sector(billiard::BilliardGeometry.AbsBilliard, choices::Pair...; irrep2d=nothing) → sector::SymmetrySector

Construct a [`SymmetrySector`](@ref) for `billiard` from representation choices
of the form `GeneratorType => value`.

Every symmetry of the billiard's symmetry group `G'` (see
`get_symmetry_group`; for a billiard
with holes, the registered elements that leave every hole invariant) whose type
matches `GeneratorType` is assigned the
corresponding one-dimensional irreducible-representation character. Reflection
values must be `±1`. For an `N`-fold rotation, an integer sector `s` in
`0:d-1`, with `d` the order of the rotation subgroup of `G'`, assigns

    χ_s(g^m) = exp(2πi s m/N)

to each rotation element `g^m` of `G'`. Requesting a generator type that the
billiard does not register raises an `ArgumentError`; requesting one that is
registered but broken by a hole raises an `ArgumentError` naming `G'`.

## Arguments
* `billiard::BilliardGeometry.AbsBilliard`: Billiard whose registered symmetries define the available generators.
* `choices::Pair...`: `GeneratorType => value` pairs selecting the representation sector.

## Keyword Arguments
* `irrep2d::Union{Nothing,Integer} = nothing`: Index `k` of the two-dimensional irrep `E_k` of a dihedral group `D_d`, `1 ≤ k < d/2`. The sector is the cyclic sector `s = k`, which contains every `E_k` level once; each level is twofold degenerate in the full problem (`symmetry_degeneracy(sector) == 2`). Cannot be combined with `choices`.

## Returns
* `sector::SymmetrySector`: Representation sector associated with `billiard`.
"""
function symmetry_sector(billiard::BilliardGeometry.AbsBilliard, choices::Pair...; irrep2d::Union{Nothing,Integer} = nothing)
    isnothing(irrep2d) || return _irrep2d_sector(billiard, choices, irrep2d)
    allunique(first.(choices)) || throw(ArgumentError("symmetry_sector received duplicate generator types among choices; each GeneratorType may appear at most once"))
    group = BilliardGeometry.get_symmetry_group(billiard)
    d = _rotation_subgroup_order(group)
    characters = Dict{Int,ComplexF64}()
    for (GenType, val) in choices
        matches = filter(s -> s isa GenType, group)
        if isempty(matches)
            any(s -> s isa GenType, billiard.symmetries) && throw(ArgumentError("$(GenType) is registered by the billiard but not in its symmetry group G' = $(_group_description(group)); a hole breaks it"))
            throw(ArgumentError("Billiard has no registered $(GenType) symmetry generator"))
        end
        for sym in matches
            characters[sym.sym_id] = _sector_character(sym, val, d)
        end
    end
    return SymmetrySector(billiard, characters)
end

# Number of 2D irreps of a dihedral group with `d` rotations: k = 1, ..., ⌈d/2⌉-1.
_n_irreps_2d(d::Integer) = (d - 1) ÷ 2

function _irrep2d_sector(billiard::BilliardGeometry.AbsBilliard, choices, k::Integer)
    isempty(choices) || throw(ArgumentError("irrep2d cannot be combined with generator choices; a two-dimensional irrep is selected by its index alone"))
    group = BilliardGeometry.get_symmetry_group(billiard)
    d = _rotation_subgroup_order(group)
    nmirror = length(group) + 1 - d
    (nmirror == d && d >= 3) || throw(ArgumentError("irrep2d requires a dihedral symmetry group D_d with d ≥ 3; the billiard's group is $(_group_description(group))"))
    1 <= k <= _n_irreps_2d(d) || throw(ArgumentError("irrep2d must be in 1:$(_n_irreps_2d(d)) for D_$d; received $k"))
    characters = Dict{Int,ComplexF64}()
    for sym in group
        sym isa BilliardGeometry.NFoldRotation && (characters[sym.sym_id] = _sector_character(sym, k, d))
    end
    return SymmetrySector(billiard, characters, 2)
end

_group_description(group) = isempty(group) ? "{e}" : "{e, " * join(string.(group), ", ") * "}"

"""
    symmetry_irreps(billiard::BilliardGeometry.AbsBilliard) → info::NamedTuple

Classify the symmetry group `G'` of `billiard` (see
`get_symmetry_group`) and list its
one-dimensional irreps in the form accepted by
`symmetry_sector(billiard, characters...)`.

## Returns
* `info.group`: Group label, `"trivial"`, `"C_d"` (rotations only) or `"D_d"` (`d` rotations and `d` mirrors, `D_1` being a single mirror).
* `info.order`: Group order `|G'|`.
* `info.generators`: Minimal generators, in the order the `characters` entries refer to.
* `info.characters`: One `Tuple{Vararg{Int}}` per 1D irrep: `±1` per mirror generator, or the sector `s ∈ 0:d-1` of a cyclic group.
* `info.n_2d`: Number of 2D irreps `E_k`.
* `info.irreps_2d`: Indices `k` accepted by `symmetry_sector(billiard; irrep2d = k)`.
"""
function symmetry_irreps(billiard::BilliardGeometry.AbsBilliard)
    group = BilliardGeometry.get_symmetry_group(billiard)
    order = length(group) + 1
    isempty(group) && return (group = "trivial", order = 1, generators = (), characters = [()], n_2d = 0, irreps_2d = Int[])
    gens = BilliardGeometry.get_symmetries(group)
    d = _rotation_subgroup_order(group)
    nmirror = order - d
    if nmirror == 0
        return (group = "C_$d", order = order, generators = gens, characters = [(s,) for s in 0:d-1], n_2d = 0, irreps_2d = Int[])
    end
    nmirror == d || throw(ArgumentError("symmetry group of order $order with $d rotations and $nmirror mirrors is not dihedral"))
    all(g -> g isa BilliardGeometry.AbsReflection, gens) || throw(ArgumentError("dihedral group generators are expected to be mirrors; received $(map(typeof, gens))"))
    ng = length(gens)
    all_chars = collect(Iterators.product(ntuple(_ -> (1, -1), ng)...))[:]
    chars = isodd(d) && d > 1 ? filter(allequal, all_chars) : all_chars
    return (group = "D_$d", order = order, generators = gens, characters = Tuple{Vararg{Int}}[c for c in chars], n_2d = _n_irreps_2d(d), irreps_2d = collect(1:_n_irreps_2d(d)))
end

_symmetry_label(g::BilliardGeometry.AbsSymmetry) = string(g)
_symmetry_label(g::BilliardGeometry.NFoldRotation) = "Rotation(2π*$(g.m)/$(g.order))"
_irrep_description(::BilliardGeometry.AbsReflection, χ::Integer) = χ == 1 ? "+1 (Neumann on the mirror)" : "-1 (Dirichlet on the mirror)"
function _irrep_description(::BilliardGeometry.NFoldRotation, s::Integer, d::Integer)
    d1 = s == 0 || 2s == d
    return d1 ? "s=$s (real)" : "s=$s (complex, degenerate with s=$(d - s))"
end

"""
    log_symmetries(billiard::BilliardGeometry.AbsBilliard) → nothing

Log with `@info` the symmetry group `G'` of `billiard`, its irreps and, for each
irrep, the character tuple to pass to [`symmetry_sector`](@ref), e.g.
`symmetry_sector(billiard, (1, -1))`. Each generator's meaning is on its own line.

## Character tuples
A character tuple has one entry per generator in `info.generators` (the order of
`get_symmetries`) and selects one irrep.
It is accepted by `symmetry_sector(billiard, tuple)` and by the `sym_characters`
keyword of the BIM solvers.

* Mirror generator: `+1` (Neumann on the mirror) or `-1` (Dirichlet on the mirror).
* Rotation group `C_d`: a single entry, the cyclic sector `s ∈ 0:d-1`, meaning the rotation by `2π/d` acts on the wavefunction as the phase `exp(2πi s/d)`. Sectors `s` and `d-s` are complex conjugates and degenerate.
* Dihedral group `D_d`, one-dimensional irreps: one `±1` per mirror generator. For odd `d` both mirrors must carry the same sign.
* Dihedral group `D_d`, two-dimensional irrep `E_k`: a single entry `k ∈ 1:⌊(d-1)/2⌋`, which is solved in the cyclic sector `s = k`. Every level is twofold degenerate in the full problem.
"""
function log_symmetries(billiard::BilliardGeometry.AbsBilliard)
    name = nameof(typeof(billiard))
    info = symmetry_irreps(billiard)
    if info.order == 1
        @info "$name: no symmetry, a single (trivial) sector; use symmetry_sector(billiard) with no characters"
        return nothing
    end
    gens = info.generators
    d = _rotation_subgroup_order(BilliardGeometry.get_symmetry_group(billiard))
    lines1d = String[]
    for c in info.characters
        parts = [_symmetry_label(gens[i]) * " => " * (gens[i] isa BilliardGeometry.NFoldRotation ? _irrep_description(gens[i], c[i], d) : _irrep_description(gens[i], c[i])) for i in eachindex(gens)]
        push!(lines1d, "    " * string(c) * " :\n        " * join(parts, "\n        "))
    end
    lines2d = String[]
    for k in info.irreps_2d
        parts = ["Rotation(2π/$d) => two eigenvalues exp(+2πi*$k/$d) and exp(-2πi*$k/$d) (cyclic sectors s=$k and s=$(d - k))"]
        append!(parts, _symmetry_label(g) * " => swaps the two rotation eigenvalues; no single ±1 character" for g in gens)
        push!(lines2d, "    " * string((k,)) * " :  irrep E_$k, every level twofold degenerate\n        " * join(parts, "\n        ") * "\n        (solved once in cyclic sector s=$k, which contains each E_$k level once)")
    end
    irreps_text = "  $(length(lines1d)) one-dimensional irrep(s):\n" * join(lines1d, "\n")
    isempty(lines2d) || (irreps_text *= "\n  $(length(lines2d)) two-dimensional irrep(s):\n" * join(lines2d, "\n"))
    msg = "$name:\n" *
          "  Symmetry group:\n" *
          "    $(info.group) of order $(info.order)\n" *
          "  Generators:\n" *
          join(("    " * _symmetry_label(g) for g in gens), "\n") * "\n" *
          irreps_text
    @info msg
    return nothing
end

"""
    symmetry_sector(billiard::BilliardGeometry.AbsBilliard, characters::Integer...) → sector::SymmetrySector

Construct a [`SymmetrySector`](@ref) for `billiard` by assigning one
representation character to each of `billiard`'s minimal fundamental symmetry
generators, in the order returned by
`get_symmetries`.

## Description
This is a positional shortcut for the `GeneratorType => value` form: it pairs
`characters[i]` with `get_symmetries(billiard)[i]`'s generator type and
delegates to `symmetry_sector(billiard, GeneratorType => characters[i], ...)`.
For a `D₂`-symmetric billiard, `get_symmetries` returns
`(YAxisReflection, XAxisReflection)`, so
`symmetry_sector(billiard, -1, +1)` assigns `χ=-1` to the `YAxisReflection`
generator and `χ=+1` to the `XAxisReflection` generator. For a cyclic
`Nᵗʰ`-order rotational billiard, `get_symmetries` returns the single `m=1`
generator, so a single integer sector `s ∈ 0:N-1` selects
`χ_s(g^m) = exp(2πi s m/N)` for every registered rotation image, exactly as
the `NFoldRotation => s` form does.

## Arguments
* `billiard::BilliardGeometry.AbsBilliard`: Billiard whose minimal symmetry generators (via `get_symmetries`) receive the given characters.
* `characters::Integer...`: One character per generator returned by `get_symmetries(billiard)`, in the same order.

## Returns
* `sector::SymmetrySector`: Representation sector associated with `billiard`.
"""
function symmetry_sector(billiard::BilliardGeometry.AbsBilliard, character1::Integer, characters::Integer...)
    all_characters = (character1, characters...)
    gens = BilliardGeometry.get_symmetries(billiard)
    if length(all_characters) == 1 && length(gens) != 1 && !isempty(symmetry_irreps(billiard).irreps_2d)
        return symmetry_sector(billiard; irrep2d = character1)
    end
    length(all_characters) == length(gens) || throw(ArgumentError(
        "symmetry_sector(billiard, characters...) requires one character per " *
        "get_symmetries(billiard) generator ($(length(gens)) expected " *
        "$(map(typeof, gens))); received $(length(all_characters))"))
    return symmetry_sector(billiard, (typeof(sym) => val for (sym, val) in zip(gens, all_characters))...)
end

"""
    symmetry_partner(f, billiard::BilliardGeometry.AbsBilliard) → g

Degenerate partner `g(x) = f(σx)` of a function `f` in a two-dimensional dihedral
irrep sector, with `σ` the first mirror of the symmetry group `G'` of `billiard`.
The partner is linearly independent of `f` and has the same eigenvalue.
"""
function symmetry_partner(f, billiard::BilliardGeometry.AbsBilliard)
    group = BilliardGeometry.get_symmetry_group(billiard)
    i = findfirst(g -> g isa BilliardGeometry.AbsReflection, group)
    isnothing(i) && throw(ArgumentError("the symmetry group $(_group_description(group)) of the billiard contains no mirror"))
    σ = group[i]
    return pt -> f(BilliardGeometry.apply_symmetry(σ, pt))
end

"""
    symmetry_sector(billiard, characters::Tuple{Vararg{Integer}}) → sector::SymmetrySector

Tuple form of the positional method, equivalent to
`symmetry_sector(billiard, characters...)`. The irrep dimension is inferred from
the number of entries: as many entries as generators of
`get_symmetries(billiard)` is a one-dimensional irrep, and a single entry `k` on
a dihedral group `D_d` with `d ≥ 3` (two mirror generators) is the
two-dimensional irrep `E_k`, i.e. `irrep2d = k`.
"""
symmetry_sector(billiard::BilliardGeometry.AbsBilliard, characters::Tuple{Integer,Vararg{Integer}}) = symmetry_sector(billiard, characters...)

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
    gens = Pair{BilliardGeometry.AbsSymmetry,ComplexF64}[BilliardGeometry.symmetry_of(BilliardGeometry.get_symmetry_group(billiard), id) => char for (id, char) in sector.characters]
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

# Recover every cyclic sector `s` whose character on the registered element g^m
# equals `char`. The finite set s=0,...,N-1 is searched directly because the
# phase is defined modulo 2π; dividing the principal argument by `m` does not
# in general invert the map s ↦ sm mod N. Several sectors match when m and N
# share a factor.
function _matching_rotation_sectors(gen::BilliardGeometry.NFoldRotation, char::ComplexF64, d::Int = gen.order)
    order = gen.order
    return [s for s in 0:d-1 if isapprox(cis(2*pi*s*gen.m/order), char; atol=1e-9)]
end

# Smallest sector matching `char` on the element g^m.
function _match_rotation_sector(gen::BilliardGeometry.NFoldRotation, char::ComplexF64)
    sectors = _matching_rotation_sectors(gen, char)
    isempty(sectors) && throw(ArgumentError("Character $char is not a valid NFoldRotation irrep value for m=$(gen.m), order=$(gen.order)"))
    return first(sectors)
end

# Resolve several registered elements of one cyclic rotation group to a common
# sector index. All elements must have the same group order; the sector is the
# intersection of the sectors compatible with each element's character, which
# must be a single index to define one consistent one-dimensional representation.
#
# The sector `s` is defined on the m = 1 generator, χ_s(g^m) = exp(2πi s m/N), whereas
# `BilliardGeometry.symmetry_index_orbits(T, billiard, N, g, s')` assigns the phase
# exp(2πi s' l/d) to the l-th power of the element `g` passed to it, with d the order
# of g. The returned generator is the registered element of smallest m that generates the
# whole group (independent of the iteration order of `gens`), and the sector is
# converted to its own convention, s' = s m/gcd(N, m) mod d, so the folded phases
# stay χ_s(g^m).
function _resolve_bim_rotation_sector(gens::Vector{Pair{BilliardGeometry.AbsSymmetry,ComplexF64}})
    order = gens[1].first.order
    all(g -> g.first.order == order, gens) || throw(ArgumentError("Mismatched NFoldRotation orders within one SymmetrySector"))
    dgroup = order ÷ foldl(gcd, (g.first.m for g in gens); init = order)
    candidates = [_matching_rotation_sectors(gen, char, dgroup) for (gen, char) in gens]
    any(isempty, candidates) && throw(ArgumentError("Character is not a valid NFoldRotation irrep value for the registered elements: candidate sectors $candidates"))
    sectors = reduce(intersect, candidates)
    isempty(sectors) && throw(ArgumentError("Inconsistent rotation sector recovered across sym_ids: $candidates"))
    length(sectors) == 1 || throw(ArgumentError("Ambiguous rotation sector: the registered elements are compatible with sectors $sectors"))
    gen = first(sort!(BilliardGeometry.AbsSymmetry[g.first for g in gens]; by = g -> (gcd(order, g.m) != 1, g.m)))
    d = order ÷ gcd(order, gen.m)
    return gen, (mod(only(sectors) * (gen.m ÷ gcd(order, gen.m)), d),)
end
