import math
import meep as mp


def arc_wall(
    *,
    source_pos=(0.0, 0.0),
    distance=2.0,
    opening_deg=90.0,
    thickness=0.2,
    smoothness=1.0,
    orientation_deg=180.0,
    height=mp.inf,
    epsilon=1.0
):

    # inspect parameter
    if distance <= 0:
        raise ValueError("[warning]distance have to > 0")
    if thickness <= 0:
        raise ValueError("[warning]thickness have to > 0")
    if not (0 < opening_deg < 360):
        raise ValueError("[warning]opening_deg have to in (0, 360)")
    if smoothness <= 0:
        raise ValueError("[warning]smoothness have to > 0")

    # compute the inner r and outer r
    r_inner = distance
    r_outer = distance + thickness

    # degree to radian
    theta_open = math.radians(opening_deg)
    theta_center = math.radians(orientation_deg)
    th0 = theta_center - theta_open / 2
    th1 = theta_center + theta_open / 2

    # compute the number of plot points
    num_pts = max(2, int(16 * smoothness * (opening_deg / 90.0)) + 1)

    # locate the plot points
    cx, cy = source_pos
    outer_arc = [
        mp.Vector3(cx + r_outer * math.cos(th0 + (th1 - th0) * i / (num_pts - 1)),
                   cy + r_outer * math.sin(th0 + (th1 - th0) * i / (num_pts - 1)))
        for i in range(num_pts)
    ]
    inner_arc = [
        mp.Vector3(cx + r_inner * math.cos(th1 - (th1 - th0) * i / (num_pts - 1)),
                   cy + r_inner * math.sin(th1 - (th1 - th0) * i / (num_pts - 1)))
        for i in range(num_pts)
    ]

    # close points into a frame
    vertices = outer_arc + inner_arc

    # use "Prism" to export the geometry
    wall = mp.Prism(
        vertices=vertices,
        height=height,
        material = mp.Medium(epsilon=epsilon)
    )

    return wall

#================================================================================

def arc_mirro(
    *,
    source_pos=(0.0, 0.0),
    distance=2.0,
    opening_deg=90.0,
    orientation_deg=180.0,
    height=mp.inf,
):

    thickness=0.4
    smoothness=1.6
    # inspect parameters
    if distance <= 0:
        raise ValueError("[warning]distance have to > 0")

    if not (0 < opening_deg < 360):
        raise ValueError("[warning]opening_deg have to in (0, 360)")

    # compute the inner r and outer r
    r_inner = distance
    r_outer = distance + thickness

    # degree to radian
    theta_open = math.radians(opening_deg)
    theta_center = math.radians(orientation_deg)
    th0 = theta_center - theta_open / 2
    th1 = theta_center + theta_open / 2

    # compute the number of plot points
    num_pts = max(2, int(16 * smoothness * (opening_deg / 90.0)) + 1)

    # locate the plot points
    cx, cy = source_pos
    outer_arc = [
        mp.Vector3(cx + r_outer * math.cos(th0 + (th1 - th0) * i / (num_pts - 1)),
                   cy + r_outer * math.sin(th0 + (th1 - th0) * i / (num_pts - 1)))
        for i in range(num_pts)
    ]
    inner_arc = [
        mp.Vector3(cx + r_inner * math.cos(th1 - (th1 - th0) * i / (num_pts - 1)),
                   cy + r_inner * math.sin(th1 - (th1 - th0) * i / (num_pts - 1)))
        for i in range(num_pts)
    ]

    # close points into a frame
    vertices = outer_arc + inner_arc

    # use "Prism" to export the geometry
    wall = mp.Prism(
        vertices=vertices,
        height=height,
        material = mp.Medium(epsilon=1e6)
    )

    return wall

#=====================================================
def reverse_arc_halfmirro(
    *,
    source_pos=(0.0, 0.0),
    distance=2.0,
    opening_deg=90.0,
    orientation_deg=180.0,
    height=mp.inf,
    epsilon=3.0
):

    thickness=0.4
    smoothness=1.6
    # inspect parameters
    if distance <= 0:
        raise ValueError("[warning]distance have to > 0")

    if not (0 < opening_deg < 360):
        raise ValueError("[warning]opening_deg have to in (0, 360)")

    # compute the inner r and outer r
    r_inner = distance
    r_outer = distance + thickness

    # degree to radian
    theta_open = math.radians(opening_deg)
    theta_center = math.radians((orientation_deg+180.0)%360.0)
    th0 = theta_center - theta_open / 2
    th1 = theta_center + theta_open / 2

    # compute the number of plot points
    num_pts = max(2, int(16 * smoothness * (opening_deg / 90.0)) + 1)

    # locate the plot points
    cx, cy = source_pos
    outer_arc = [
        mp.Vector3(cx + r_outer * math.cos(th0 + (th1 - th0) * i / (num_pts - 1)),
                   cy + r_outer * math.sin(th0 + (th1 - th0) * i / (num_pts - 1)))
        for i in range(num_pts)
    ]
    inner_arc = [
        mp.Vector3(cx + r_inner * math.cos(th1 - (th1 - th0) * i / (num_pts - 1)),
                   cy + r_inner * math.sin(th1 - (th1 - th0) * i / (num_pts - 1)))
        for i in range(num_pts)
    ]

    # close points into a frame
    vertices = outer_arc + inner_arc

    # use "Prism" to export the geometry
    wall = mp.Prism(
        vertices=vertices,
        height=height,
        material = mp.Medium(epsilon=epsilon)
    )

    return wall

#===============================
import math
import meep as mp


def conic_mirror(
    *,
    source_pos=(0.0, 0.0),
    distance=2.0,                 # source -> apex along -u0 
    orientation_deg=0.0,          
    y_max=10.0,
    x_edge=3.0,                   # x_draw(y_max) = x_edge (>0)
    e=1.0,                        # eccentricity
    thickness=0.4,                
    n_pts=61,
    height=mp.inf,
    eps_mirror=1e6,
    return_meta=False,
    force_open_to_plus_u=True,    
    edge_tol=1e-9,
    debug=False,
    fail_mode="raise",   # "raise" or "return_none"

):
    # --- checks ---
    if distance <= 0 or y_max <= 0 or x_edge <= 0 or e <= 0:
        raise ValueError("distance,y_max,x_edge,e must be > 0")
    if orientation_deg % 180 != 0:
        raise ValueError("orientation_deg should be 0 or 180 for x-axis symmetry")
    if n_pts < 9:
        raise ValueError("n_pts should be >= 9")

    ### [ADD-1] failure helper
    def _fail(msg: str):
        if fail_mode == "raise":
            raise ValueError(msg)
        return (None, {"failed": True, "reason": msg}) if return_meta else None

    # --- fixed axes (DO NOT flip these) ---
    th = math.radians(orientation_deg)
    u0x, u0y = math.cos(th), math.sin(th)     # opening direction in global coords
    v0x, v0y = -u0y, u0x                      # perpendicular

    # --- apex placement (never depends on e) ---
    sx, sy = source_pos
    apex_x = sx - distance * u0x
    apex_y = sy - distance * u0y

    # --- vertex-referenced conic in local coordinates ---
    # y^2 = 2 R x - q x^2, q = 1 - e^2
    q = 1.0 - e * e
    r = y_max
    xe = x_edge
    if q < 0.0 and (r*r + q*xe*xe) <= 0.0:
        e_max = math.sqrt(1.0 + (r/xe)**2)
        xe_max = r / math.sqrt(e*e - 1.0)
        return _fail(
            "Infeasible hyperbola parameters for vertex-at-apex conic.\n"
            "Need y_max^2 > (e^2-1)*x_edge^2.\n"
            f"Given y_max={r}, x_edge={xe}, e={e}.\n"
            f"Either reduce e <= {e_max:.6g}, or reduce x_edge <= {xe_max:.6g}, or increase y_max."
        )
    # Solve R from x(r)=xe in the vertex-referenced form (consistent with sag below)
    if abs(q) < 1e-12:            # parabola
        R = (r * r) / (2.0 * xe)
    else:
        R = (q * xe * xe + r * r) / (2.0 * xe)

    def sag_raw(y_local: float) -> float:
        """
        Return x(y) in a vertex-referenced local frame, GUARANTEED to satisfy x(0)=0.
        This requires choosing the correct quadratic root depending on sign(R).

        Quadratic: q x^2 - 2 R x + y^2 = 0
        Solution: x = (R ± sqrt(R^2 - q y^2)) / q
        Enforce x(0)=0:
          - if R >= 0 -> choose minus
          - if R <  0 -> choose plus
        """
        y2 = y_local * y_local

        # parabola limit: x = y^2 / (2R) -> x(0)=0 automatically
        if abs(q) < 1e-12:
            return y2 / (2.0 * R)

        inside = R * R - q * y2
        if inside < 0.0:
            inside = 0.0
        s = math.sqrt(inside)

        if R >= 0.0:
            return (R - s) / q
        else:
            return (R + s) / q
    ### [ADD-4] vertex consistency check
    if abs(sag_raw(0.0)) > 1e-12:
        return _fail(f"vertex constraint violated: sag_raw(0)={sag_raw(0.0)} (should be 0).")

    # --- choose sign WITHOUT changing axes ---
    # We want x_draw(y_max) = +x_edge in the +u0 direction.
    x_at_edge_raw = sag_raw(y_max)
    sign = 1.0
    if force_open_to_plus_u and x_at_edge_raw < 0.0:
        sign = -1.0

    def x_draw(y_local: float) -> float:
        return sign * sag_raw(y_local)

    # --- strict edge check in draw-coordinates ---
    x_edge_actual = x_draw(y_max)
    if abs(x_edge_actual - x_edge) > edge_tol:
        return _fail(
            f"edge constraint mismatch: got x(y_max)={x_edge_actual}, expected {x_edge}. "
            "This indicates inconsistent R/sag or numerical issues."
        )


    # --- global mapping (axes fixed) ---
    def to_global(x_local: float, y_local: float) -> mp.Vector3:
        gx = apex_x + x_local * u0x + y_local * v0x
        gy = apex_y + x_local * u0y + y_local * v0y
        return mp.Vector3(gx, gy)

    ys = [(-y_max + 2.0 * y_max * i / (n_pts - 1)) for i in range(n_pts)]

    inner_curve = [to_global(x_draw(y), y) for y in ys]
    outer_curve = [to_global(x_draw(y) + thickness, y) for y in ys[::-1]]
    vertices = outer_curve + inner_curve

    # --- debug (MPI-safe) ---
    if debug and mp.am_master():
        p_vertex = to_global(0.0, 0.0)  # should be apex
        p_all_minx = min(vertices, key=lambda p: p.x)
        print("apex      =", (apex_x, apex_y), flush=True)
        print("vertex    =", (p_vertex.x, p_vertex.y), flush=True)
        print("R,q       =", (R, q), flush=True)
        print("sag_raw(0)=", sag_raw(0.0), "x_draw(0)=", x_draw(0.0), flush=True)
        print("sign      =", sign, "x_edge_raw=", x_at_edge_raw, "x_edge=", x_edge_actual, flush=True)
        print("min-x pt  =", (p_all_minx.x, p_all_minx.y), flush=True)

    wall = mp.Prism(
        vertices=vertices,
        height=height,
        material=mp.Medium(epsilon=eps_mirror),
    )

    if not return_meta:
        return wall

    meta = {
        "source_pos": tuple(source_pos),
        "apex_pos": (apex_x, apex_y),
        "distance_source_to_apex": distance,
        "orientation_deg": orientation_deg,
        "y_max": y_max,
        "x_edge_target": x_edge,
        "e": e,
        "q": q,
        "R": R,
        "sign": sign,
        "x_edge_raw": x_at_edge_raw,
        "x_edge_draw": x_edge_actual,
        "sag_raw_0": sag_raw(0.0),
        "equation": "q x^2 - 2R x + y^2 = 0 ; choose root so that x(0)=0 ; x_draw=sign*x_raw",
        "note": "Branch selection enforces vertex at apex; axes never flip; thickness offsets along +u0.",
        "vertices": [(v.x, v.y) for v in vertices],
    }
    return wall, meta





