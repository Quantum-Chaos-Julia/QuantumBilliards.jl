# Unit tests for
#   src/spectra/spectralutils.jl   (SpectralData, k_range, filter, append!/vcat, match_wavenumbers,
#                                   overlap_and_merge!, overlap_and_merge_ebim!, _local_gap,
#                                   sweep_windows, weyl_window_count)
#   src/solvers/acceleratedmethods/beyn.jl   (weyl_window_width, plan_weyl_windows,
#                                   beyn_disks_from_windows, beyn_buffer_matrices)
# No eigen-solves: every expected value is analytic or built by hand.

using Random: MersenneTwister
using QuantumBilliards: match_wavenumbers, is_equal, _local_gap, _require_states

# Minimal stand-in eigenstate; `id` lets tests track which state survived a merge.
struct SpectraTestState <: QuantumBilliards.AbsState
    id::Int
end

@testset "SpectralData construction" begin
    b = RectangleBilliard(1.0, 0.5)
    k = [1.0, 2.0, 3.0]; ten = [1e-3, 2e-3, 3e-3]; ctrl = [false, true, false]

    @testset "real wavenumbers" begin
        data = SpectralData(b, :solver, k, ten, ctrl)
        @test data isa SpectralData{Float64}
        @test data.billiard === b
        @test data.solver === :solver
        @test data.k == k
        @test data.k_im == zeros(3)
        @test data.ten == ten
        @test data.control == ctrl
        @test data.states === nothing
        @test length(data) == 3
    end

    @testset "complex wavenumbers split into k and k_im" begin
        kc = [1.0 + 0.1im, 2.0 - 0.2im, 3.0 + 0.0im]
        data = SpectralData(b, :solver, kc, ten, ctrl)
        @test data.k == real.(kc)
        @test data.k_im == imag.(kc)
        @test eltype(data.k) == Float64
    end

    @testset "element type follows tension" begin
        data = SpectralData(b, :solver, k, Float32.(ten), ctrl)
        @test data isa SpectralData{Float32}
        @test eltype(data.k) == Float32
        @test eltype(data.k_im) == Float32
        @test data.k == Float32.(k)
    end

    @testset "states keyword" begin
        states = [SpectraTestState(i) for i in 1:3]
        data = SpectralData(b, :solver, k, ten, ctrl; states)
        @test data.states === states
        @test data isa SpectralData{Float64,<:Any,Symbol,Vector{SpectraTestState}}
    end
end

@testset "SpectralData indexing and iteration" begin
    b = RectangleBilliard(1.0, 0.5)
    states = [SpectraTestState(i) for i in 1:4]
    data = SpectralData(b, :solver, [1.0 + 0.1im, 2.0, 3.0, 4.0], [1e-3, 2e-3, 3e-3, 4e-3], [false, true, false, true]; states)
    plain = SpectralData(b, :solver, [1.0, 2.0, 3.0, 4.0], [1e-3, 2e-3, 3e-3, 4e-3], [false, true, false, true])

    @testset "integer index" begin
        d = data[1]
        @test d isa SpectralData
        @test length(d) == 1
        @test d.k == [1.0]
        @test d.k_im == [0.1]
        @test d.ten == [1e-3]
        @test d.control == [false]
        @test [s.id for s in d.states] == [1]
        @test d.billiard === b
        @test d.solver === :solver
        @test plain[2].states === nothing
    end

    @testset "vector and mask index" begin
        d = data[[4, 2]]
        @test d.k == [4.0, 2.0]
        @test d.ten == [4e-3, 2e-3]
        @test d.control == [true, true]
        @test [s.id for s in d.states] == [4, 2]
        m = plain[[true, false, true, false]]
        @test m.k == [1.0, 3.0]
        @test m.states === nothing
        @test length(plain[Int[]]) == 0
    end

    @testset "iteration" begin
        items = collect(data)
        @test length(items) == length(data)
        @test all(d -> d isa SpectralData && length(d) == 1, items)
        @test [only(d.k) for d in items] == data.k
        @test [only(d.states).id for d in items] == [1, 2, 3, 4]
    end

    @testset "out-of-range index throws" begin
        @test_throws BoundsError data[5]
        @test_throws BoundsError plain[0]
    end
end

@testset "k_range" begin
    b = RectangleBilliard(1.0, 0.5)
    states = [SpectraTestState(i) for i in 1:5]
    data = SpectralData(b, :solver, [1.0, 2.0, 3.0, 4.0, 5.0], fill(1e-3, 5), fill(false, 5); states)

    sub = k_range(data, 2.0, 4.0)
    @test sub.k == [2.0, 3.0, 4.0]   # closed interval: both endpoints are kept
    @test [s.id for s in sub.states] == [2, 3, 4]
    @test k_range(data, 2.5, 3.5).k == [3.0]
    @test length(k_range(data, 10.0, 20.0)) == 0
    @test length(k_range(data, 4.0, 2.0)) == 0
    @test k_range(data, 0.0, 10.0).k == data.k
end

@testset "SpectralData filter" begin
    b = RectangleBilliard(1.0, 0.5)
    states = [SpectraTestState(i) for i in 1:4]
    data = SpectralData(b, :solver, [1.0, 2.0, 3.0, 4.0], [1e-5, 1e-2, 1e-6, 1e-1], [false, true, false, true]; states)

    good = filter(d -> only(d.ten) < 1e-3, data)
    @test good isa SpectralData
    @test good.k == [1.0, 3.0]
    @test [s.id for s in good.states] == [1, 3]
    @test filter(d -> only(d.control), data).k == [2.0, 4.0]
    @test length(filter(d -> false, data)) == 0
    @test filter(d -> true, data).k == data.k
end

@testset "_require_states" begin
    b = RectangleBilliard(1.0, 0.5)
    states = [SpectraTestState(1)]
    with = SpectralData(b, :solver, [1.0], [1e-3], [false]; states)
    without = SpectralData(b, :solver, [1.0], [1e-3], [false])
    @test _require_states(with) === states
    @test_throws ArgumentError _require_states(without)
end

@testset "SpectralData append!" begin
    b = RectangleBilliard(1.0, 0.5)
    other = RectangleBilliard(1.0, 0.5)
    mk(k, ten) = SpectralData(b, :solver, k, ten, fill(false, length(k)))

    @testset "disjoint spectra concatenate in sorted order" begin
        d1 = mk([1.0, 2.0, 3.0], [1e-3, 1e-3, 1e-3])
        d2 = mk([3.5, 4.0], [2e-3, 3e-3])
        r = append!(d1, d2)
        @test r === d1
        @test d1.k == [1.0, 2.0, 3.0, 3.5, 4.0]
        @test d1.ten == [1e-3, 1e-3, 1e-3, 2e-3, 3e-3]
        @test d1.k_im == zeros(5)
        @test length(d1.control) == 5
    end

    @testset "empty operands" begin
        d1 = mk([1.0, 2.0], [1e-3, 1e-3])
        @test append!(d1, mk(Float64[], Float64[])).k == [1.0, 2.0]
        e = mk(Float64[], Float64[])
        append!(e, mk([5.0], [1e-3]))
        @test e.k == [5.0]
        @test e.control == [false]
    end

    @testset "states are carried along and re-sorted" begin
        s1 = [SpectraTestState(1), SpectraTestState(2)]
        s2 = [SpectraTestState(3)]
        d1 = SpectralData(b, :solver, [1.0, 2.0], [1e-3, 1e-3], [false, false]; states=s1)
        d2 = SpectralData(b, :solver, [3.0], [1e-3], [false]; states=s2)
        append!(d1, d2)
        @test [s.id for s in d1.states] == [1, 2, 3]
    end

    @testset "argument errors" begin
        d1 = mk([1.0], [1e-3])
        @test_throws ArgumentError append!(d1, SpectralData(other, :solver, [2.0], [1e-3], [false]))
        @test_throws ArgumentError append!(d1, SpectralData(b, 1.0, [2.0], [1e-3], [false]))
    end

    @testset "overlapping spectra are deduplicated" begin
        # `append!` passes kl = data1.k[end] > kr = data2.k[1] to `overlap_and_merge!`, whose
        # window (kl-tol, kr+tol) is then empty, so nothing is matched and duplicates survive.
        # Finding: src/spectra/spectralutils.jl, `Base.append!(::SpectralData, ::SpectralData)`.
        d1 = mk([1.0, 2.0, 3.0], [1e-3, 1e-3, 1e-3])
        d2 = mk([2.0001, 3.0001, 4.0], [1e-4, 1e-4, 1e-3])
        append!(d1, d2)
        @test_broken length(d1.k) == 4
    end
end

@testset "SpectralData vcat" begin
    b = RectangleBilliard(1.0, 0.5)
    d1 = SpectralData(b, :solver, [1.0, 2.0], [1e-3, 1e-3], [false, false])
    d2 = SpectralData(b, :solver, [3.0, 4.0], [1e-3, 1e-3], [false, false])
    r = vcat(d1, d2)
    @test r isa SpectralData
    @test r.k == [1.0, 2.0, 3.0, 4.0]
    @test d1.k == [1.0, 2.0]    # operands are not mutated
    @test d2.k == [3.0, 4.0]
    @test r.k !== d1.k
    @test_throws ArgumentError vcat(d1, SpectralData(RectangleBilliard(1.0, 0.5), :solver, [5.0], [1e-3], [false]))
end

@testset "is_equal" begin
    @test is_equal(1.0, 0.1, 1.15, 0.1)          # intervals [0.9,1.1] and [1.05,1.25] overlap
    @test !is_equal(1.0, 0.1, 1.5, 0.1)
    @test is_equal(1.0, 0.25, 1.5, 0.25)         # touching intervals count as equal
    @test is_equal(1.0 + 5.0im, 0.1, 1.05, 0.1)  # only the real part is compared
    @test is_equal(2.0, 0.0, 2.0, 0.0)
    @test @inferred(is_equal(1.0, 0.1, 1.15, 0.1)) isa Bool
end

@testset "match_wavenumbers" begin
    @testset "matched pair keeps lower-tension representative" begin
        ks, ts, ctrl, st = match_wavenumbers([1.0, 2.0, 3.0], [0.01, 0.01, 0.01], [2.005, 3.5], [0.02, 0.01])
        @test ks == [1.0, 2.0, 3.0, 3.5]
        @test ts == [0.01, 0.01, 0.01, 0.01]
        @test ctrl == [false, true, false, false]
        @test st === nothing
        ks, ts, ctrl, _ = match_wavenumbers([2.0], [0.02], [2.005], [0.01])
        @test ks == [2.005]
        @test ts == [0.01]
        @test ctrl == [true]
    end

    @testset "no matches interleaves both spectra" begin
        ks, ts, ctrl, _ = match_wavenumbers([1.0, 3.0], [1e-3, 1e-3], [2.0, 4.0], [2e-3, 2e-3])
        @test ks == [1.0, 2.0, 3.0, 4.0]
        @test ts == [1e-3, 2e-3, 1e-3, 2e-3]
        @test ctrl == fill(false, 4)
    end

    @testset "empty inputs" begin
        ks, ts, ctrl, _ = match_wavenumbers(Float64[], Float64[], [1.0, 2.0], [1e-3, 2e-3])
        @test ks == [1.0, 2.0]
        @test ts == [1e-3, 2e-3]
        @test ctrl == [false, false]
        ks, _, ctrl, _ = match_wavenumbers([1.0], [1e-3], Float64[], Float64[])
        @test ks == [1.0]
        @test ctrl == [false]
        ks, ts, ctrl, _ = match_wavenumbers(Float64[], Float64[], Float64[], Float64[])
        @test isempty(ks) && isempty(ts) && isempty(ctrl)
    end

    @testset "states follow the retained wavenumbers" begin
        sl = [SpectraTestState(10), SpectraTestState(20)]
        sr = [SpectraTestState(30), SpectraTestState(40)]
        # 2.0 (left, 0.01) vs 2.005 (right, 0.02): left wins; 5.0 unmatched on the right
        ks, _, ctrl, st = match_wavenumbers([1.0, 2.0], [0.01, 0.01], [2.005, 5.0], [0.02, 0.01]; states_l=sl, states_r=sr)
        @test ks == [1.0, 2.0, 5.0]
        @test ctrl == [false, true, false]
        @test [s.id for s in st] == [10, 20, 40]
    end

    @testset "complex wavenumbers" begin
        ks, ts, ctrl, _ = match_wavenumbers([1.0 + 1e-6im, 2.0 + 0im], [1e-3, 1e-3], [1.0005 + 0im], [5e-3])
        @test ks == [1.0 + 1e-6im, 2.0 + 0im]
        @test ctrl == [true, false]
    end
end

@testset "overlap_and_merge!" begin
    # Overlap window (kl, kr) = (2.0, 3.0), tol = 1e-3.
    # Left covers [1, 3), right covers (kl, ...). Right entry 1.9 lies below kl - tol and is dropped.
    kl, kr = 2.0, 3.0
    mkleft() = ([1.0, 2.2, 2.8], [1e-3, 4e-3, 2e-3], [false, false, false])
    kright = [1.9, 2.2002, 2.8004, 3.5, 4.0]
    tright = [1e-3, 1e-3, 5e-3, 1e-3, 1e-3]

    @testset "merges overlap window and appends the right tail" begin
        k, t, c = mkleft()
        ret = overlap_and_merge!(k, t, copy(kright), copy(tright), c, kl, kr; tol=1e-3)
        @test ret === nothing
        @test k == [1.0, 2.2002, 2.8, 3.5, 4.0]
        @test t == [1e-3, 1e-3, 2e-3, 1e-3, 1e-3]
        @test c == [false, true, true, false, false]
        @test issorted(k)
    end

    @testset "states follow the retained candidates" begin
        k, t, c = mkleft()
        sl = [SpectraTestState(1), SpectraTestState(2), SpectraTestState(3)]
        sr = [SpectraTestState(11), SpectraTestState(12), SpectraTestState(13), SpectraTestState(14), SpectraTestState(15)]
        overlap_and_merge!(k, t, copy(kright), copy(tright), c, kl, kr; tol=1e-3, states_left=sl, states_right=sr)
        @test [s.id for s in sl] == [1, 12, 3, 14, 15]
        @test length(sl) == length(k)
    end

    @testset "empty left copies right with cleared controls" begin
        k = Float64[]; t = Float64[]; c = Bool[]
        overlap_and_merge!(k, t, [1.0, 2.0], [1e-3, 2e-3], c, kl, kr)
        @test k == [1.0, 2.0]
        @test t == [1e-3, 2e-3]
        @test c == [false, false]
    end

    @testset "empty right leaves left unchanged" begin
        k, t, c = mkleft()
        overlap_and_merge!(k, t, Float64[], Float64[], c, kl, kr)
        @test k == [1.0, 2.2, 2.8]
        @test t == [1e-3, 4e-3, 2e-3]
        @test c == [false, false, false]
    end

    @testset "disjoint right spectrum is appended whole" begin
        k = [1.0, 1.5]; t = [1e-3, 1e-3]; c = [false, false]
        overlap_and_merge!(k, t, [5.0, 6.0], [1e-3, 1e-3], c, 2.0, 3.0)
        @test k == [1.0, 1.5, 5.0, 6.0]
        @test c == fill(false, 4)
    end

    @testset "complex wavenumbers" begin
        k = ComplexF64[1.0, 2.2 + 1e-8im]; t = [1e-3, 4e-3]; c = [false, false]
        overlap_and_merge!(k, t, ComplexF64[2.2002 + 1e-9im, 3.5], [1e-3, 1e-3], c, 2.0, 3.0; tol=1e-3)
        @test real.(k) == [1.0, 2.2002, 3.5]
        @test imag(k[2]) == 1e-9
        @test c == [false, true, false]
    end
end

@testset "_local_gap" begin
    @test _local_gap([0.0, 1.0, 2.0, 4.0, 7.0], 3) == 1.5      # median of (1,1,2,3)
    @test _local_gap([0.0, 1.0, 3.0, 6.0], 2) == 2.0           # odd count: median of (1,2,3)
    @test _local_gap([0.0, 1.0, 2.0, 4.0, 7.0], 1; w=1) == 1.0 # window truncated to xs[1:2]
    @test _local_gap([0.0, 1.0, 2.0, 4.0, 7.0], 5; w=1) == 3.0 # window truncated to xs[4:5]
    @test _local_gap([1.0, 1.0, 3.0], 2) == 2.0                # zero gaps are ignored
    @test _local_gap([1.0], 1) == Inf                          # no gaps at all
    @test _local_gap([2.0, 2.0, 2.0], 2) == Inf
    @test _local_gap([0.0, 1.0, 10.0, 11.0], 1; w=1) == 1.0    # w limits how far the window reaches
    @test @inferred(_local_gap([0.0, 1.0, 2.0], 2)) isa Float64
    @test _local_gap(Float32[0, 1, 2, 4, 7], 3) isa Float32
end

@testset "overlap_and_merge_ebim!" begin
    @testset "duplicate corrections merge, best-scoring one wins" begin
        # Score = log10(|Im k|) + log10(|tension|); lower is better.
        kL = ComplexF64[1.0 + 1e-3im, 2.0 + 1e-6im, 3.0 + 1e-3im]; tL = [1e-3, 1e-3, 1e-3]; cL = [false, false, false]
        kR = ComplexF64[2.000001 + 1e-8im, 3.5 + 1e-3im]; tR = [1e-4, 1e-3]
        ret = overlap_and_merge_ebim!(kL, tL, kR, tR, cL)
        @test ret === nothing
        @test real.(kL) == [1.0, 2.000001, 3.0, 3.5]
        @test imag(kL[2]) == 1e-8
        @test tL == [1e-3, 1e-4, 1e-3, 1e-3]
        @test cL == [false, true, false, false]
    end

    @testset "worse right candidate is discarded" begin
        kL = ComplexF64[2.0 + 1e-8im]; tL = [1e-4]; cL = [false]
        kR = ComplexF64[2.000001 + 1e-5im]; tR = [1e-2]
        overlap_and_merge_ebim!(kL, tL, kR, tR, cL)
        @test length(kL) == 1
        @test real(kL[1]) == 2.0
        @test tL == [1e-4]
        @test cL == [true]
    end

    @testset "separation above tolmax is not merged" begin
        kL = ComplexF64[1.0, 2.0]; tL = [1e-3, 1e-3]; cL = [false, false]
        kR = ComplexF64[2.01, 3.0]; tR = [1e-3, 1e-3]
        overlap_and_merge_ebim!(kL, tL, kR, tR, cL)   # 2.01 - 2.0 = 1e-2 > tolmax = 5e-3
        @test real.(kL) == [1.0, 2.0, 2.01, 3.0]
        @test cL == fill(false, 4)
    end

    @testset "tolmax keyword controls the cutoff" begin
        kL = ComplexF64[2.0 + 1e-6im]; tL = [1e-3]; cL = [false]
        kR = ComplexF64[2.01 + 1e-8im]; tR = [1e-3]
        overlap_and_merge_ebim!(kL, tL, kR, tR, cL; tolmax=0.1, spacing_frac=1.0)
        @test length(kL) == 1
        @test real(kL[1]) == 2.01
        @test cL == [true]
    end

    @testset "empty right leaves left unchanged" begin
        kL = ComplexF64[1.0, 2.0]; tL = [1e-3, 2e-3]; cL = [false, true]
        @test overlap_and_merge_ebim!(kL, tL, ComplexF64[], Float64[], cL) === nothing
        @test kL == ComplexF64[1.0, 2.0]
        @test tL == [1e-3, 2e-3]
        @test cL == [false, true]
    end

    @testset "empty left adopts the right spectrum" begin
        kL = ComplexF64[]; tL = Float64[]; cL = Bool[]
        overlap_and_merge_ebim!(kL, tL, ComplexF64[1.0, 2.0], [1e-3, 1e-3], cL)
        @test real.(kL) == [1.0, 2.0]
        @test cL == [false, false]
    end

    @testset "unsorted input is sorted by real part" begin
        kL = ComplexF64[3.0, 1.0]; tL = [1e-3, 1e-3]; cL = [false, false]
        overlap_and_merge_ebim!(kL, tL, ComplexF64[2.0], [1e-3], cL)
        @test real.(kL) == [1.0, 2.0, 3.0]
    end

    @testset "states follow the retained candidates" begin
        kL = ComplexF64[2.0 + 1e-6im, 3.0]; tL = [1e-3, 1e-3]; cL = [false, false]
        kR = ComplexF64[2.000001 + 1e-8im, 4.0]; tR = [1e-4, 1e-3]
        sL = [SpectraTestState(1), SpectraTestState(2)]
        sR = [SpectraTestState(11), SpectraTestState(12)]
        overlap_and_merge_ebim!(kL, tL, kR, tR, cL; states_left=sL, states_right=sR)
        @test real.(kL) == [2.000001, 3.0, 4.0]
        @test [s.id for s in sL] == [11, 2, 12]
    end
end

@testset "sweep_windows" begin
    @testset "constant step" begin
        centers, widths = sweep_windows(0.0, 1.0, 0.25)
        @test centers == [0.0, 0.25, 0.5, 0.75]
        @test widths == fill(0.25, 4)
        @test @inferred(sweep_windows(0.0, 1.0, 0.25)) isa Tuple{Vector{Float64},Vector{Float64}}
    end

    @testset "coverage property" begin
        centers, widths = sweep_windows(3.0, 10.0, 0.3)
        @test centers[1] == 3.0
        @test centers[end] < 10.0
        @test centers[end] + widths[end] >= 10.0
        @test all(diff(centers) .≈ 0.3)
        @test length(centers) == length(widths)
    end

    @testset "adaptive step" begin
        centers, widths = sweep_windows(1.0, 4.0, k -> k / 2)
        @test centers == [1.0, 1.5, 2.25, 3.375]
        @test widths == [0.5, 0.75, 1.125, 1.6875]
        @test centers[2:end] == centers[1:end-1] .+ widths[1:end-1]
    end

    @testset "type promotion" begin
        centers, _ = sweep_windows(0, 1.0, 0.5)
        @test eltype(centers) == Float64
        @test centers == [0.0, 0.5]
        centers, widths = sweep_windows(0, 4, 2)
        @test eltype(centers) == Int
        @test centers == [0, 2]
        @test widths == [2, 2]
        @test eltype(first(sweep_windows(1.0f0, 2.0f0, 0.5f0))) == Float32
    end

    @testset "single window when dk exceeds the interval" begin
        centers, widths = sweep_windows(1.0, 2.0, 5.0)
        @test centers == [1.0]
        @test widths == [5.0]
    end

    @testset "argument errors" begin
        @test_throws ArgumentError sweep_windows(2.0, 1.0, 0.1)
        @test_throws ArgumentError sweep_windows(1.0, 1.0, 0.1)
        @test_throws ArgumentError sweep_windows(0.0, 1.0, 0.0)
        @test_throws ArgumentError sweep_windows(0.0, 1.0, -0.1)
        @test_throws ArgumentError sweep_windows(0.0, 1.0, k -> -1.0)
    end
end

@testset "weyl_window_count" begin
    a, bb = 1.0, 0.5
    b = RectangleBilliard(a, bb)         # full rectangle 2a x 2b, area 4ab
    A_full = 4a * bb
    k, dk = 10.0, 0.5

    @test weyl_window_count(b, k, dk; fundamental=false) ≈ A_full * ((k + dk)^2 - k^2) / (4pi)
    @test weyl_window_count(b, k, dk; fundamental=false) ≈ weyl_law(k + dk, A_full, 0.0) - weyl_law(k, A_full, 0.0)
    @test weyl_window_count(b, k, dk) ≈ weyl_window_count(b, k, dk; fundamental=false) / 4   # D2 reduction
    @test weyl_window_count(b, k, 0.0) == 0.0
    @test weyl_window_count(b, k, dk) > 0
    @test weyl_window_count(b, k, 2dk) > 2 * weyl_window_count(b, k, dk)                    # grows quadratically
    # Finding: `area`/`fundamental_area` (BilliardGeometry) infer as `Any` for RectangleBilliard
    # (Base.return_types(area, (typeof(b),)) == [Any]), so billiard-level Weyl helpers are not type-stable.
    @test_broken @inferred(weyl_window_count(b, k, dk)) isa Float64
end

@testset "weyl_window_width" begin
    b = RectangleBilliard(1.0, 0.5)
    k, m = 10.0, 7
    w = weyl_window_width(b, k, m)
    @test w > 0
    @test weyl_window_count(b, k, w) ≈ m                                   # width contains exactly m Weyl states
    wfull = weyl_window_width(b, k, m; fundamental=false)
    @test weyl_window_count(b, k, wfull; fundamental=false) ≈ m
    @test wfull < w                                                        # larger area -> narrower window
    @test weyl_window_width(b, 20.0, m) < w                                # density grows with k
    @test weyl_window_width(b, k, 2m) > w
    @test_broken @inferred(weyl_window_width(b, k, m)) isa Float64   # same `area` instability as weyl_window_count
end

@testset "plan_weyl_windows" begin
    b = RectangleBilliard(1.0, 0.5)

    @testset "adjacent windows covering [k1,k2]" begin
        wins = plan_weyl_windows(b, 10.0, 30.0; m=10, Rmax=100.0)
        @test wins isa Vector{Tuple{Float64,Float64}}
        @test wins[1][1] == 10.0
        @test wins[end][2] ≈ 30.0
        @test all(i -> wins[i][2] == wins[i+1][1], 1:length(wins)-1)
        @test all(w -> w[1] < w[2], wins)
    end

    @testset "uncapped windows hold m states, last window at most m" begin
        wins = plan_weyl_windows(b, 10.0, 30.0; m=10, Rmax=100.0)
        counts = [weyl_window_count(b, kL, kR - kL) for (kL, kR) in wins]
        @test all(c -> isapprox(c, 10.0; rtol=1e-8), counts[1:end-1])
        @test 0 < counts[end] <= 10.0 * (1 + 1e-8)
    end

    @testset "Rmax caps the window width" begin
        wins = plan_weyl_windows(b, 10.0, 12.0; m=1000, Rmax=0.25)
        widths = [kR - kL for (kL, kR) in wins]
        @test all(w -> w <= 0.5 + 1e-12, widths)
        @test all(w -> isapprox(w, 0.5), widths[1:end-1])
        @test length(wins) == 4
    end

    @testset "fundamental flag changes window width" begin
        wf = plan_weyl_windows(b, 10.0, 30.0; m=10, Rmax=100.0)
        wfull = plan_weyl_windows(b, 10.0, 30.0; m=10, Rmax=100.0, fundamental=false)
        @test length(wfull) > length(wf)
    end

    @testset "empty and degenerate intervals" begin
        @test plan_weyl_windows(b, 10.0, 10.0) == Tuple{Float64,Float64}[]
        @test plan_weyl_windows(b, 12.0, 10.0) == Tuple{Float64,Float64}[]
        @test plan_weyl_windows(b, 12.0, 10.0; m=0) == Tuple{Float64,Float64}[]   # emptiness is checked first
    end

    @testset "argument errors" begin
        @test_throws ArgumentError plan_weyl_windows(b, 10.0, 20.0; m=0)
        @test_throws ArgumentError plan_weyl_windows(b, 10.0, 20.0; m=-3)
        @test_throws ArgumentError plan_weyl_windows(b, 10.0, 20.0; Rmax=0.0)
        @test_throws ArgumentError plan_weyl_windows(b, 10.0, 20.0; Rmax=-1.0)
    end

    @testset "type stability" begin
        @test @inferred(plan_weyl_windows(b, 10.0, 20.0; m=10, Rmax=1.0)) isa Vector{Tuple{Float64,Float64}}
    end
end

@testset "beyn_disks_from_windows" begin
    k0, R = beyn_disks_from_windows([(1.0, 2.0), (2.0, 4.0), (4.0, 4.5)])
    @test k0 == ComplexF64[1.5, 3.0, 4.25]
    @test R == [0.5, 1.0, 0.25]
    @test k0 isa Vector{ComplexF64}
    @test R isa Vector{Float64}
    @test all(iszero, imag.(k0))

    k0e, Re = beyn_disks_from_windows(Tuple{Float64,Float64}[])
    @test isempty(k0e) && isempty(Re)

    k32, R32 = beyn_disks_from_windows([(1.0f0, 3.0f0)])
    @test k32 isa Vector{ComplexF32}
    @test R32 isa Vector{Float32}
    @test k32 == ComplexF32[2.0]
    @test @inferred(beyn_disks_from_windows([(1.0, 2.0)])) isa Tuple{Vector{ComplexF64},Vector{Float64}}
end

@testset "beyn_buffer_matrices" begin
    V, X, A0, A1 = beyn_buffer_matrices(Float64, 6, 3, MersenneTwister(1))
    @test size(V) == size(X) == size(A0) == size(A1) == (6, 3)
    @test eltype(V) == ComplexF64
    @test eltype(X) == ComplexF64
    @test eltype(A0) == ComplexF64
    @test eltype(A1) == ComplexF64
    @test all(iszero, A0) && all(iszero, A1)
    @test !all(iszero, V)                                  # random probing matrix
    @test A0 !== A1 && V !== X                             # independent buffers
    V2, _, _, _ = beyn_buffer_matrices(Float64, 6, 3, MersenneTwister(1))
    @test V == V2                                          # reproducible from the rng
    V3, _, _, _ = beyn_buffer_matrices(Float64, 6, 3, MersenneTwister(2))
    @test V != V3
    V32, X32, A032, _ = beyn_buffer_matrices(Float32, 4, 2, MersenneTwister(1))
    @test eltype(V32) == ComplexF32 && eltype(X32) == ComplexF32 && eltype(A032) == ComplexF32
    @test @inferred(beyn_buffer_matrices(Float64, 4, 2, MersenneTwister(1))) isa NTuple{4,Matrix{ComplexF64}}
end
