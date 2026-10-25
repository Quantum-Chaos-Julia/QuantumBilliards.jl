"""
    _polar_coords!(r, φ, pts, origin, rot_angle, rotation_angle_discontinuity) → (r, φ)

Fill `r` and `φ` in place with the polar coordinates of `pts` in the local frame
with the given `origin` and axes rotated by `rot_angle`, i.e. of
\$R(-\\theta)(x - x_0)\$.

The local point is additionally rotated by `rotation_angle_discontinuity` before
taking `atan`, so that the branch cut lies outside the domain, and the angle is
shifted back afterwards. Both rotations are combined into a single `sincos`.
"""
@inline function _polar_coords!(r::AbstractVector,φ::AbstractVector,pts,origin,rot_angle,rotation_angle_discontinuity)
    θ0=rotation_angle_discontinuity
    s,c=sincos(θ0-rot_angle)
    x0=origin[1]
    y0=origin[2]
    @inbounds for j in eachindex(pts)
        p=pts[j]
        dx=p[1]-x0
        dy=p[2]-y0
        xr=dx*c-dy*s
        yr=dx*s+dy*c
        r[j]=hypot(xr,yr)
        φ[j]=atan(yr,xr)-θ0
    end
    return r,φ
end
