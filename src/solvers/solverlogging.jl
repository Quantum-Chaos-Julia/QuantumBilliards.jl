################################################################################
# SOLVER PARAMETER LOGGING
#
# Every user-facing solver constructor reports its configuration with a single
# `@info` message, grouped into thematic sections in the style of
# `log_symmetries`. Parameters that are inactive because another parameter
# disables them (e.g. the Taylor settings when `use_taylor = false`, the
# Chebyshev settings when the Taylor backend takes precedence) are omitted;
# the controlling flag is always shown. Composite objects (symmetry reduction,
# boundary grading, Chebyshev configuration, wrapped kernels, component
# solvers) are expanded into their own rows.
################################################################################

function Base.show(io::IO, ::MIME"text/plain", solver::AbsSolver)
    buf = IOBuffer()
    _write_solver!(buf, solver, 2)
    print(io, nameof(typeof(solver)), ":\n", rstrip(String(take!(buf))))
end

"""
    log_solver(solver::AbsSolver) → nothing

Log with `@info` the parameters of `solver`, grouped into sections
(boundary discretization, symmetry reduction, matrix backend, ...). Parameters
disabled by another parameter are not listed. The same text is displayed
by `show` when a solver is returned at the REPL.
"""
function log_solver(solver::AbsSolver)
    @info repr(MIME("text/plain"), solver)
    return nothing
end

_ln!(io::IO, ind::Int, parts...) = println(io, " "^ind, parts...)
_sec!(io::IO, ind::Int, title::AbstractString) = _ln!(io, ind, title, ":")
_row!(io::IO, ind::Int, key::AbstractString, val) = _ln!(io, ind, key, " = ", repr(val))
_rowstr!(io::IO, ind::Int, key::AbstractString, str) = _ln!(io, ind, key, " = ", str)

function _describe(x)
    T = typeof(x)
    fieldcount(T) == 0 && return string(nameof(T))
    return string(nameof(T), "(", join((string(f, "=", repr(getfield(x, f))) for f in fieldnames(T)), ", "), ")")
end

function _describe_samplers(samplers::AbstractVector)
    desc = join(unique(_describe.(samplers)), ", ")
    return length(samplers) > 1 ? string(desc, " (", length(samplers), " entries)") : desc
end

_write_grading!(io::IO, ind::Int, ::SmoothPeriodicGrading) = _rowstr!(io, ind, "grading", "SmoothPeriodicGrading")
function _write_grading!(io::IO, ind::Int, g::Union{CornerGrading,GlobalCornerGrading})
    _rowstr!(io, ind, "grading", nameof(typeof(g)))
    _row!(io, ind + 2, "kressq", g.kressq)
    _row!(io, ind + 2, "min_t_spacing", g.min_t_spacing)
end
_write_grading!(io::IO, ind::Int, g::BoundaryGrading) = _rowstr!(io, ind, "grading", _describe(g))

_symmetry_character_pairs(symmetry::BilliardGeometry.CompositeReflection, characters::Tuple) = zip(symmetry.reflections, characters)
_symmetry_character_pairs(symmetry::BilliardGeometry.AbsSymmetry, characters::Tuple) = length(characters) == 1 ? ((symmetry, characters[1]),) : ((symmetry, characters),)

_character_description(g::BilliardGeometry.AbsReflection, c::Number) = _irrep_description(g, Int(round(real(c))))
_character_description(g::BilliardGeometry.NFoldRotation, c::Number) = _irrep_description(g, Int(round(real(c))), g.order ÷ gcd(g.order, g.m))
_character_description(g, c) = string(c)

function _write_symmetry!(io::IO, ind::Int, symmetry, characters::Tuple)
    _sec!(io, ind, "Symmetry")
    if isnothing(symmetry)
        _ln!(io, ind + 2, "none (full boundary, all levels)")
        return
    end
    for (g, c) in _symmetry_character_pairs(symmetry, characters)
        _ln!(io, ind + 2, _symmetry_label(g), " => ", _character_description(g, c))
    end
end

function _write_bim_discretization!(io::IO, ind::Int, solver)
    _row!(io, ind, "pts_scaling_factor", solver.pts_scaling_factor)
    _row!(io, ind, "min_pts", solver.min_pts)
    _write_grading!(io, ind, solver.grading)
end

function _write_kernel!(io::IO, ind::Int, kernel)
    _ln!(io, ind, "Kernel: ", nameof(typeof(kernel)))
    _write_solver!(io, kernel, ind + 2)
end

function _write_chebyshev_config!(io::IO, ind::Int, cfg::ChebyshevConfig)
    _row!(io, ind, "n_panels_h", cfg.n_panels_h)
    _row!(io, ind, "M_h", cfg.M_h)
    _row!(io, ind, "n_panels_j", cfg.n_panels_j)
    _row!(io, ind, "M_j", cfg.M_j)
    _row!(io, ind, "param_strategy", cfg.param_strategy)
    if cfg.param_strategy !== :manual
        _row!(io, ind, "tol", cfg.tol)
        _row!(io, ind, "max_iter", cfg.max_iter)
        _row!(io, ind, "sampling_points", cfg.sampling_points)
        _row!(io, ind, "grow_panels", cfg.grow_panels)
        _row!(io, ind, "grow_M", cfg.grow_M)
    end
end

################################################################################
# BIM SWEEP SOLVERS
################################################################################

function _write_solver!(io::IO, s::Union{DoubleLayerPotentialSolver,CombinedFieldIntegralEquationSolver}, ind::Int)
    _sec!(io, ind, "Boundary discretization")
    _rowstr!(io, ind + 2, "billiard", nameof(typeof(s.billiard)))
    _write_bim_discretization!(io, ind + 2, s)
    _write_symmetry!(io, ind, s.symmetry, s.sym_characters)
end

function _write_solver!(io::IO, s::CompositeBIMSolver, ind::Int)
    _sec!(io, ind, "Boundary discretization")
    _rowstr!(io, ind + 2, "billiard", nameof(typeof(s.component_solvers[1].billiard)))
    _ln!(io, ind + 2, "components (", length(s.component_solvers), "):")
    for (i, cs) in enumerate(s.component_solvers)
        role = i == 1 ? "outer boundary" : "hole $(i - 1)"
        _ln!(io, ind + 4, "[", i, "] ", role, ": ", nameof(typeof(cs)))
        _write_bim_discretization!(io, ind + 6, cs)
    end
    _write_symmetry!(io, ind, s.symmetry, s.sym_characters)
end

################################################################################
# BIM ACCELERATED SOLVERS
################################################################################

function _write_solver!(io::IO, s::BeynSolver, ind::Int)
    _write_kernel!(io, ind, s.kernel)
    _sec!(io, ind, "Contour integration")
    _row!(io, ind + 2, "nlevels", s.nlevels)
    _row!(io, ind + 2, "Rmax", s.Rmax)
    _row!(io, ind + 2, "nq", s.nq)
    _row!(io, ind + 2, "r", s.r)
    _row!(io, ind + 2, "svd_tol", s.svd_tol)
    _sec!(io, ind, "Root validation")
    _row!(io, ind + 2, "res_tol", s.res_tol)
    _row!(io, ind + 2, "auto_discard_spurious", s.auto_discard_spurious)
    _row!(io, ind + 2, "imag_k_check", s.imag_k_check)
    if s.imag_k_check
        _row!(io, ind + 2, "imag_k_pad", s.imag_k_pad)
        _row!(io, ind + 2, "imag_k_group_size", s.imag_k_group_size)
    end
    _sec!(io, ind, "Matrix backend")
    if s.use_taylor
        _rowstr!(io, ind + 2, "backend", "taylor")
        _row!(io, ind + 2, "taylor_degree", s.taylor_degree)
        _row!(io, ind + 2, "taylor_tol", s.taylor_tol)
    elseif s.use_chebyshev
        _rowstr!(io, ind + 2, "backend", "chebyshev")
        _write_chebyshev_config!(io, ind + 2, s.cheb_config)
    else
        _rowstr!(io, ind + 2, "backend", "direct")
    end
    _sec!(io, ind, "Output")
    _row!(io, ind + 2, "eigenvectors", s.eigenvectors)
end

function _write_solver!(io::IO, s::ExpandedBIMSolver, ind::Int)
    _write_kernel!(io, ind, s.kernel)
    _sec!(io, ind, "Matrix backend")
    if s.use_taylor
        _rowstr!(io, ind + 2, "backend", "taylor")
        _row!(io, ind + 2, "taylor_degree", s.taylor_degree)
        _row!(io, ind + 2, "taylor_radius", s.taylor_radius)
        _row!(io, ind + 2, "taylor_tol", s.taylor_tol)
    else
        _rowstr!(io, ind + 2, "backend", "direct")
    end
    _sec!(io, ind, "Krylov eigensolver")
    _row!(io, ind + 2, "krylov_tol", s.krylov_tol)
    _row!(io, ind + 2, "maxiter", s.maxiter)
    _row!(io, ind + 2, "krylovdim", s.krylovdim)
end

function _write_solver!(io::IO, s::CORKSolver, ind::Int)
    _write_kernel!(io, ind, s.kernel)
    _sec!(io, ind, "Chebyshev polynomial")
    _row!(io, ind + 2, "p", s.p)
    _row!(io, ind + 2, "guard", s.guard)
    _row!(io, ind + 2, "taylor_tol", s.taylor_tol)
    _sec!(io, ind, "Spectral windows")
    _row!(io, ind + 2, "nlevels", s.nlevels)
    _row!(io, ind + 2, "Rmax", s.Rmax)
    _sec!(io, ind, "Krylov iteration")
    _row!(io, ind + 2, "block_size", s.block_size)
    _row!(io, ind + 2, "mstart", s.mstart)
    _row!(io, ind + 2, "mstep", s.mstep)
    _row!(io, ind + 2, "stable_checks", s.stable_checks)
    _row!(io, ind + 2, "stable_tol", s.stable_tol)
    _sec!(io, ind, "Root acceptance")
    _row!(io, ind + 2, "imag_tol", s.imag_tol)
    _row!(io, ind + 2, "imag_search_tol", s.imag_search_tol)
    _row!(io, ind + 2, "edge_tol", s.edge_tol)
    _row!(io, ind + 2, "res_tol", s.res_tol)
    _sec!(io, ind, "Output")
    _row!(io, ind + 2, "eigenvectors", s.eigenvectors)
end

################################################################################
# BASIS SOLVERS
################################################################################

function _write_solver!(io::IO, s::DecompositionMethodSolver, ind::Int)
    _sec!(io, ind, "Basis")
    _row!(io, ind + 2, "dim_scaling_factor", s.dim_scaling_factor)
    _row!(io, ind + 2, "min_dim", s.min_dim)
    _sec!(io, ind, "Boundary sampling")
    _row!(io, ind + 2, "pts_scaling_factor", s.pts_scaling_factor)
    _row!(io, ind + 2, "min_pts", s.min_pts)
    _rowstr!(io, ind + 2, "sampler", _describe_samplers(s.sampler))
    _sec!(io, ind, "Decomposition")
    _row!(io, ind + 2, "eig_tol", s.eig_tol)
    _row!(io, ind + 2, "rellich_origin", s.rellich_origin)
end

function _write_solver!(io::IO, s::ParticularSolutionsMethodSolver, ind::Int)
    _sec!(io, ind, "Basis")
    _row!(io, ind + 2, "dim_scaling_factor", s.dim_scaling_factor)
    _row!(io, ind + 2, "min_dim", s.min_dim)
    _sec!(io, ind, "Boundary sampling")
    _row!(io, ind + 2, "pts_scaling_factor", s.pts_scaling_factor)
    _row!(io, ind + 2, "min_pts", s.min_pts)
    _rowstr!(io, ind + 2, "sampler", _describe_samplers(s.sampler))
    _sec!(io, ind, "Interior sampling")
    _row!(io, ind + 2, "int_pts_scaling_factor", s.int_pts_scaling_factor)
    _row!(io, ind + 2, "min_int_pts", s.min_int_pts)
    _sec!(io, ind, "Singular value filtering")
    _row!(io, ind + 2, "eps", s.eps)
end

function _write_solver!(io::IO, s::VerginiSaracenoSolver, ind::Int)
    _sec!(io, ind, "Basis")
    _row!(io, ind + 2, "dim_scaling_factor", s.dim_scaling_factor)
    _row!(io, ind + 2, "min_dim", s.min_dim)
    _sec!(io, ind, "Boundary sampling")
    _row!(io, ind + 2, "pts_scaling_factor", s.pts_scaling_factor)
    _row!(io, ind + 2, "min_pts", s.min_pts)
    _rowstr!(io, ind + 2, "sampler", _describe_samplers(s.sampler))
    _sec!(io, ind, "Scaling eigenproblem")
    _row!(io, ind + 2, "eig_tol", s.eig_tol)
    _sec!(io, ind, "Output")
    _row!(io, ind + 2, "eigenvectors", s.eigenvectors)
end
