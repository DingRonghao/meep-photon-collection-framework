import meep as mp
import os
import math
import numpy as np
from geom_2D import arc_wall,arc_mirro

def cave_1(
    #----parameter for control----
    wall_1_to_source = 1.5,
    wall_1_open_deg = 200.0,
    wall_1_thick = 0.8,
    wall_1_smoothness = 1.6, #limited in(1.2-3.0)
    wall_1_epsilon = 3.0,
    flux_rigion_center_x = -2.0
) :
    #------size in real world---
    wave_length = 1000.0e-9
    fiber_ridus = 10.0e-6

    #------define space and objects------
    cell_x = 64
    cell_y = 32
    cell_z = 0
    cell = mp.Vector3(cell_x, cell_y, cell_z) #space
    source_1 = mp.Vector3(-15.0,0.0)
    resolution = 20

    fiber_epsilon = 2.09

    fiber_core = mp.Cylinder(
        center=source_1,  
        radius=0.8,               
        material=mp.Medium(epsilon=fiber_epsilon)
    )

    geometry = [
        arc_wall(
            distance=wall_1_to_source,
            opening_deg=wall_1_open_deg,
            thickness=wall_1_thick,
            smoothness=wall_1_smoothness,
            epsilon=wall_1_epsilon,
            source_pos=(source_1.x,source_1.y)
            ),
        fiber_core
        
        ]

    # ------define source------
    sources = [
        mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.2), 
                        component=mp.Ez,
                        center=source_1)
    ]

    # ------define simulation-----
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=[mp.PML(1.0)],
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution)

    #-------flux record region-----
    flux_rigion_width = 2*(source_1.x-flux_rigion_center_x)*np.tan(np.pi/12.0)
    flux_gather_points = 21
    f0 = 1.0
    df = 0.1
    flux_region = mp.FluxRegion(center=mp.Vector3(flux_rigion_center_x, 0), size=mp.Vector3(0, flux_rigion_width))
    flux_obj = sim.add_flux(
        f0,         # center frequency
        df,         # width of spectrem
        flux_gather_points, # gather points
        flux_region  # region
)

    #-------simulation run--------
    sim.run(until=50) 

    #--------gather flux-------
    # export flux inpoints as lists
    flux_vals = mp.get_fluxes(flux_obj)
    # average
    phi_avg = sum(flux_vals) / len(flux_vals)
    # center point
    phi_center = flux_vals[flux_gather_points//2]
    # bandwidth
    BW_ratio = phi_avg / phi_center
    print(f"[info]avg flux={phi_avg:.3e},  center flux={phi_center:.3e},  BW={BW_ratio:.2f}")

    return phi_avg
#==========================================================================================================================
from rate import compute_Q_from_spectrum
from rate import compute_mode_volume_from_arrays
import gc

def cave_1_mirro(
    #----parameter for control----
    wall_1_to_source = 1.5,
    wall_1_open_deg = 200.0,
    flux_rigion_center_x = -2.0,
    wall_1_thick = 0.5,
    wall_1_smoothness = 1.6, #limited in(1.2-3.0)
    wall_1_epsilon = 3.0,
    cell_x = 64,
    cell_y = 32,
    source_1 = (-15.0,0.0)
) :
    sim = None
    eps_snapshot = None
    ez_snapshot = None

    try:
        #------size in real world---
        wave_length = 1000.0e-9
        fiber_ridus = 10.0e-6

        #------define space and objects------
        cell_z = 0
        cell = mp.Vector3(cell_x, cell_y, cell_z) #space
        source_1 = mp.Vector3(source_1[0],source_1[1])
        resolution = 20

        fiber_epsilon = 2.09

        fiber_core = mp.Cylinder(
            center=source_1,  
            radius=0.8,               
            material=mp.Medium(epsilon=fiber_epsilon)
        )

        geometry = [
            arc_mirro(
                distance=wall_1_to_source,
                opening_deg=wall_1_open_deg,
                source_pos=(source_1.x,source_1.y)
                ),
            fiber_core
            
            ]

        # ------define source------
        sources = [
            mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.2), 
                            component=mp.Ez,
                            center=source_1)
        ]

        # ------define simulation-----
        sim = mp.Simulation(cell_size=cell,
                            boundary_layers=[mp.PML(1.0)],
                            geometry=geometry,
                            sources=sources,
                            resolution=resolution)

        #-------flux record region-----
        flux_rigion_width = 2*(source_1.x-flux_rigion_center_x)*np.tan(np.pi/12.0)
        flux_gather_points = 21
        f0 = 1.0
        df = 0.1
        flux_region = mp.FluxRegion(center=mp.Vector3(flux_rigion_center_x, 0), size=mp.Vector3(0, flux_rigion_width))
        flux_obj = sim.add_flux(
            f0,         # center frequency
            df,         # width of spectrem
            flux_gather_points, # gather points
            flux_region  # region
    )

        #-------simulation run--------
        sim.run(until=50) 

        #--------gather flux-------
        #export frequency of each point 
        freqs = mp.get_flux_freqs(flux_obj)
        # export flux inpoints as lists
        flux_vals = mp.get_fluxes(flux_obj)
        # average
        phi_avg = sum(flux_vals) / len(flux_vals)
        # center point
        phi_center = flux_vals[flux_gather_points//2]
        # bandwidth
        BW_ratio = phi_avg / phi_center
        print(f"[info]avg flux={phi_avg:.3e},  center flux={phi_center:.3e},  BW={BW_ratio:.2f}")

        #-------calculate Q-----
        try:
            Q_val = compute_Q_from_spectrum(freqs=freqs, spectrum=flux_vals)
            if (Q_val is None) or (np.isnan(Q_val)) or (np.isinf(Q_val)):
                Q_val = 0.0
        except:
            Q_val = 0.0
        #-------online snapshot: eps & Ez------
        eps_snapshot = sim.get_array(
            center=mp.Vector3(),
            size=cell,
            component=mp.Dielectric
        )
        ez_snapshot = sim.get_array(
            center=mp.Vector3(),
            size=cell,
            component=mp.Ez
        )

        V_eff = compute_mode_volume_from_arrays(
            eps=eps_snapshot,
            ez=ez_snapshot,
            cell_x=cell_x,
            cell_y=cell_y,
            pml_thickness=1.0,  # 和 mp.PML(1.0) 对应
            thickness_z=1.0
        )
        return {
            "phi_avg":    float(phi_avg),
            "Q":          float(Q_val),
            "V_eff":      float(V_eff),
        }

    finally:
        eps_snapshot = None
        ez_snapshot = None

        if sim is not None:
            try:
                sim.reset_meep()
            except Exception:
                pass
        sim = None

        gc.collect()
#==========================================
from geom_2D import conic_mirror

def conic_stucture(
    #----parameter for control----
    distance=2.0,                 # source -> apex along -u0 
    y_total=10.0,
    x_total=3.0,  
    e=1.0, 
    flux_region_center_x = -2.0,
    cell_x = 64,
    cell_y = 32,
    source_1 = (-15.0, 0.0),
    resolution = 20,
    sim_time =50,
) :
    #------size in real world---
    wave_length = 1000.0e-9
    fiber_ridus = 10.0e-6

    #------define space and objects------
    cell_z = 0
    cell = mp.Vector3(cell_x, cell_y, cell_z)  # space
    source_1 = mp.Vector3(source_1[0], source_1[1])
    resolution = 20

    fiber_epsilon = 2.09

    fiber_core = mp.Cylinder(
        center=source_1,  
        radius=0.8,               
        material=mp.Medium(epsilon=fiber_epsilon)
    )

    geometry = [
        conic_mirror(
            distance=distance,
            y_max=y_total/2.0,
            x_edge=x_total,
            e=e,
            source_pos=(source_1.x,source_1.y)
            ),
        fiber_core
        
        ]

    # ------define source------
    sources = [
        mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.2), 
                        component=mp.Ez,
                        center=source_1)
    ]

    # ------define simulation-----
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=[mp.PML(1.0)],
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution)

    #-------flux record region-----
    flux_rigion_width = 2*(flux_region_center_x - source_1.x)*np.tan(np.pi/12.0)
    flux_gather_points = 21
    f0 = 1.0
    df = 0.1
    flux_region = mp.FluxRegion(center=mp.Vector3(flux_region_center_x, 0), size=mp.Vector3(0, flux_rigion_width))
    flux_obj = sim.add_flux(
        f0,         # center frequency
        df,         # width of spectrem
        flux_gather_points, # gather points
        flux_region  # region
)

    #-------simulation run--------
    sim.run(until=sim_time) 

    #--------gather flux-------
    # export flux inpoints as lists
    flux_vals = mp.get_fluxes(flux_obj)
    # average
    phi_avg = sum(flux_vals) / len(flux_vals)
    # center point
    phi_center = flux_vals[flux_gather_points//2]
    # bandwidth
    BW_ratio = phi_avg / phi_center
    print(f"[info]avg flux={phi_avg:.3e},  center flux={phi_center:.3e},  BW={BW_ratio:.2f}")

    return phi_avg
#============================================
import numpy as np
import meep as mp


def bare_structure(
    # ----parameter for control----
    flux_region_center_x = -2.0,
    cell_x = 64,
    cell_y = 32,
    source_1 = (-15.0, 0.0),
    resolution = 20,
    sim_time = 50
):
    # ------size in real world---
    wave_length = 1000.0e-9
    fiber_ridus = 10.0e-6

    # ------define space and objects------
    cell_z = 0
    cell = mp.Vector3(cell_x, cell_y, cell_z)  # space
    source_1 = mp.Vector3(source_1[0], source_1[1])
    resolution = resolution

    fiber_epsilon = 2.09

    fiber_core = mp.Cylinder(
        center=source_1,
        radius=0.8,
        material=mp.Medium(epsilon=fiber_epsilon),
    )

    # baseline: only fiber + source (no reflector / no extra geometry)
    geometry = [fiber_core]

    # ------define source------
    sources = [
        mp.Source(
            mp.GaussianSource(frequency=1.0, fwidth=0.2),
            component=mp.Ez,
            center=source_1,
        )
    ]

    # ------define simulation-----
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(1.0)],
        geometry=geometry,
        sources=sources,
        resolution=resolution,
    )

    # -------flux record region-----
    flux_region_width = 2 * (flux_region_center_x - source_1.x) * np.tan(np.pi / 12.0)
    flux_gather_points = 21
    f0 = 1.0
    df = 0.1

    flux_region = mp.FluxRegion(
        center=mp.Vector3(flux_region_center_x, 0),
        size=mp.Vector3(0, flux_region_width),
    )

    flux_obj = sim.add_flux(
        f0,                 # center frequency
        df,                 # width of spectrem
        flux_gather_points, # gather points
        flux_region,        # region
    )

    # -------simulation run--------
    sim.run(until=int(sim_time))

    # --------gather flux-------
    flux_vals = mp.get_fluxes(flux_obj)
    phi_avg = sum(flux_vals) / len(flux_vals)
    phi_center = flux_vals[flux_gather_points // 2]
    BW_ratio = phi_avg / phi_center
    print(f"[info]avg flux={phi_avg:.3e},  center flux={phi_center:.3e},  BW={BW_ratio:.2f}")

    return phi_avg

#========================================================================================
import numpy as np
import meep as mp


def flat_mirror(
    # ----parameter for control----
    flux_region_center_x=-2.0,
    cell_x=64,
    cell_y=32,
    source_1=(-15.0, 0.0),
    resolution=20,
    # ----new parameter----
    mirror_distance=5.0,  # mirror position: x = source_1.x - mirror_distance
):
    # ------size in real world---
    wave_length = 1000.0e-9
    fiber_ridus = 10.0e-6

    # ------define space and objects------
    cell_z = 0
    cell = mp.Vector3(cell_x, cell_y, cell_z)  # space
    source_1 = mp.Vector3(source_1[0], source_1[1])
    resolution = resolution

    fiber_epsilon = 2.09

    fiber_core = mp.Cylinder(
        center=source_1,
        radius=0.8,
        material=mp.Medium(epsilon=fiber_epsilon),
    )

    # ------add a large plane mirror (parallel to flux plane, spans full y)------
    # Mirror is a slab normal to x, placed on negative x side of the source.
    # To guarantee it always spans the whole y-range even if cell_y changes:
    # use size.y = cell_y (or slightly larger) in Meep units.
    mirror_eps = 1e6
    mirror_thickness = 0.4  # adjust if needed (Meep units)

    mirror_x = source_1.x - float(mirror_distance)

    plane_mirror = mp.Block(
        center=mp.Vector3(mirror_x, 0.0, 0.0),
        size=mp.Vector3(mirror_thickness, cell_y, mp.inf),
        material=mp.Medium(epsilon=mirror_eps),
    )

    # baseline: fiber + source + plane mirror
    geometry = [fiber_core, plane_mirror]

    # ------define source------
    sources = [
        mp.Source(
            mp.GaussianSource(frequency=1.0, fwidth=0.2),
            component=mp.Ez,
            center=source_1,
        )
    ]

    # ------define simulation-----
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(1.0)],
        geometry=geometry,
        sources=sources,
        resolution=resolution,
    )

    # -------flux record region-----
    flux_region_width = 2 * (flux_region_center_x - source_1.x) * np.tan(np.pi / 12.0)
    flux_gather_points = 21
    f0 = 1.0
    df = 0.1

    flux_region = mp.FluxRegion(
        center=mp.Vector3(flux_region_center_x, 0),
        size=mp.Vector3(0, flux_region_width),
    )

    flux_obj = sim.add_flux(
        f0,                 # center frequency
        df,                 # width of spectrum
        flux_gather_points, # gather points
        flux_region,        # region
    )

    # -------simulation run--------
    sim.run(until=50)

    # --------gather flux-------
    flux_vals = mp.get_fluxes(flux_obj)
    phi_avg = sum(flux_vals) / len(flux_vals)
    phi_center = flux_vals[flux_gather_points // 2]
    BW_ratio = phi_avg / phi_center
    print(f"[info]avg flux={phi_avg:.3e},  center flux={phi_center:.3e},  BW={BW_ratio:.2f}")

    return phi_avg
#=====================================================
import numpy as np
import meep as mp


def fp_mirror_pair(
    # ----parameter for control----
    flux_region_center_x=-2.0,
    cell_x=64,
    cell_y=32,
    source_1=(-15.0, 0.0),
    resolution=20,
    # ----new parameter (keep same name/meaning as flat_mirror)----
    mirror_distance=5.0,  # mirror positions: x = source_1.x ± mirror_distance
    sim_time=100
):
    # ------size in real world (kept, even if unused)---
    wave_length = 1000.0e-9
    fiber_ridus = 10.0e-6

    # ------define space and objects------
    cell_z = 0
    cell = mp.Vector3(cell_x, cell_y, cell_z)
    source_1 = mp.Vector3(source_1[0], source_1[1])
    resolution = resolution

    fiber_epsilon = 2.09
    fiber_core = mp.Cylinder(
        center=source_1,
        radius=0.8,
        material=mp.Medium(epsilon=fiber_epsilon),
    )

    # ------FP mirrors: finite-height pair------
    # Mirror slab normal to x.
    mirror_thickness = 0.4  # Meep units; keep same as before unless you want thinner
    eps_left = 1e6          # highly reflective (approx PEC-like for Ez in 2D)
    eps_right = 10.0        # "semi-transparent" output coupler (tunable)

    # Finite mirror height in y:
    # Pick a size that covers the main radiating cone but not the full cell.
    # Here: at least a few wavelengths in Meep units + a bit of margin, capped by cell_y.
    # You can tune this later.
    mirror_y_size = min(cell_y * 0.8, 18.0)  # heuristic: 60% of cell_y, max 12
    mirror_y_size = max(mirror_y_size, 6.0)  # avoid too small

    x_left = source_1.x - float(mirror_distance)
    x_right = source_1.x + float(mirror_distance)

    left_mirror = mp.Block(
        center=mp.Vector3(x_left, 0.0, 0.0),
        size=mp.Vector3(mirror_thickness, mirror_y_size, mp.inf),
        material=mp.Medium(epsilon=eps_left),
    )

    right_mirror = mp.Block(
        center=mp.Vector3(x_right, 0.0, 0.0),
        size=mp.Vector3(mirror_thickness, mirror_y_size, mp.inf),
        material=mp.Medium(epsilon=eps_right),
    )

    geometry = [fiber_core, left_mirror, right_mirror]

    # ------define source------
    sources = [
        mp.Source(
            mp.GaussianSource(frequency=1.0, fwidth=0.2),
            component=mp.Ez,
            center=source_1,
        )
    ]

    # ------define simulation-----
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(1.0)],
        geometry=geometry,
        sources=sources,
        resolution=resolution,
    )

    # -------flux record region-----
    flux_region_width = 2 * (flux_region_center_x - source_1.x) * np.tan(np.pi / 12.0)
    flux_gather_points = 21
    f0 = 1.0
    df = 0.1

    flux_region = mp.FluxRegion(
        center=mp.Vector3(flux_region_center_x, 0),
        size=mp.Vector3(0, flux_region_width),
    )

    flux_obj = sim.add_flux(
        f0,
        df,
        flux_gather_points,
        flux_region,
    )

    # -------simulation run--------
    sim.run(until=int(sim_time))

    # --------gather flux-------
    flux_vals = mp.get_fluxes(flux_obj)
    phi_avg = sum(flux_vals) / len(flux_vals)
    phi_center = flux_vals[flux_gather_points // 2]
    BW_ratio = phi_avg / phi_center
    print(f"[info]avg flux={phi_avg:.3e},  center flux={phi_center:.3e},  BW={BW_ratio:.2f}")

    return phi_avg
#=======================================
import numpy as np
import meep as mp


def flat_mirror_finite(
    # ----parameter for control----
    flux_region_center_x=-2.0,
    cell_x=64,
    cell_y=32,
    source_1=(-15.0, 0.0),
    resolution=20,
    # ----new parameter----
    mirror_distance=5.0,  # mirror position: x = source_1.x - mirror_distance
    sim_time = 100
):
    # ------size in real world (kept, even if unused)---
    wave_length = 1000.0e-9
    fiber_ridus = 10.0e-6

    # ------define space and objects------
    cell_z = 0
    cell = mp.Vector3(cell_x, cell_y, cell_z)  # space
    source_1 = mp.Vector3(source_1[0], source_1[1])
    resolution = resolution

    fiber_epsilon = 2.09
    fiber_core = mp.Cylinder(
        center=source_1,
        radius=0.8,
        material=mp.Medium(epsilon=fiber_epsilon),
    )

    # ------single finite-height plane mirror (left side)------
    mirror_thickness = 0.4
    mirror_eps = 1e6

    # same finite size rule as the previous FP script
    mirror_y_size = min(cell_y * 0.8, 18.0)
    mirror_y_size = max(mirror_y_size, 6.0)

    mirror_x = source_1.x - float(mirror_distance)

    plane_mirror = mp.Block(
        center=mp.Vector3(mirror_x, 0.0, 0.0),
        size=mp.Vector3(mirror_thickness, mirror_y_size, mp.inf),
        material=mp.Medium(epsilon=mirror_eps),
    )

    geometry = [fiber_core, plane_mirror]

    # ------define source------
    sources = [
        mp.Source(
            mp.GaussianSource(frequency=1.0, fwidth=0.2),
            component=mp.Ez,
            center=source_1,
        )
    ]

    # ------define simulation-----
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(1.0)],
        geometry=geometry,
        sources=sources,
        resolution=resolution,
    )

    # -------flux record region-----
    flux_region_width = 2 * (flux_region_center_x - source_1.x) * np.tan(np.pi / 12.0)
    flux_gather_points = 21
    f0 = 1.0
    df = 0.1

    flux_region = mp.FluxRegion(
        center=mp.Vector3(flux_region_center_x, 0),
        size=mp.Vector3(0, flux_region_width),
    )

    flux_obj = sim.add_flux(
        f0,                 # center frequency
        df,                 # width of spectrum
        flux_gather_points, # gather points
        flux_region,        # region
    )

    # -------simulation run--------
    sim.run(until=int(sim_time))

    # --------gather flux-------
    flux_vals = mp.get_fluxes(flux_obj)
    phi_avg = sum(flux_vals) / len(flux_vals)
    phi_center = flux_vals[flux_gather_points // 2]
    BW_ratio = phi_avg / phi_center
    print(f"[info]avg flux={phi_avg:.3e},  center flux={phi_center:.3e},  BW={BW_ratio:.2f}")

    return phi_avg
#===============================================================
import numpy as np
import meep as mp

from geom_2D import conic_mirror


def _clip_polygon_x_leq(vertices_xy, x_cut):
    """
    Clip a 2D polygon by half-plane x <= x_cut using Sutherland–Hodgman.
    vertices_xy: list of (x,y) in order (closed or open is OK).
    Returns list of (x,y) after clipping.
    """
    def inside(p):
        return p[0] <= x_cut + 1e-12

    def intersect(p1, p2):
        # line segment p1->p2 intersection with x=x_cut
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        if abs(dx) < 1e-15:
            # Segment parallel to cut line; no proper intersection.
            return (x_cut, y1)
        t = (x_cut - x1) / dx
        t = max(0.0, min(1.0, t))
        y = y1 + t * (y2 - y1)
        return (x_cut, y)

    if not vertices_xy or len(vertices_xy) < 3:
        return []

    output = []
    prev = vertices_xy[-1]
    prev_in = inside(prev)

    for curr in vertices_xy:
        curr_in = inside(curr)

        if curr_in:
            if not prev_in:
                output.append(intersect(prev, curr))
            output.append(curr)
        else:
            if prev_in:
                output.append(intersect(prev, curr))

        prev, prev_in = curr, curr_in

    # Remove near-duplicate consecutive points
    cleaned = []
    for p in output:
        if not cleaned:
            cleaned.append(p)
        else:
            if (abs(p[0] - cleaned[-1][0]) > 1e-12) or (abs(p[1] - cleaned[-1][1]) > 1e-12):
                cleaned.append(p)

    # If polygon collapsed
    if len(cleaned) < 3:
        return []

    return cleaned


def _mirror_vertices_about_x(vertices_xy, x0):
    """Mirror 2D points about vertical line x = x0."""
    return [(2.0 * x0 - x, y) for (x, y) in vertices_xy]


def conic_cave(
    # ---- geometry control ----
    distance=2.0,
    y_total=10.0,
    x_total=3.0,
    e=1.0,

    # ---- optical/material control ----
    eps_left=1e6,        # left half: "perfect reflector" approximation
    eps_right=20.0,      # right half: user-controlled (semi-transparent)

    # ---- simulation control ----
    flux_region_center_x=-2.0,
    cell_x=64,
    cell_y=32,
    source_1=(-15.0, 0.0),
    resolution=20,
    sim_time=50,
):
    # ------ define space and objects ------
    cell = mp.Vector3(cell_x, cell_y, 0)
    source_1 = mp.Vector3(source_1[0], source_1[1])

    fiber_epsilon = 2.09
    fiber_core = mp.Cylinder(
        center=source_1,
        radius=0.8,
        material=mp.Medium(epsilon=fiber_epsilon),
    )

    # ------ build conic mirror (left), then clip at x = source_1.x ------
    # IMPORTANT: this assumes conic_mirror supports return_meta=True and returns (geom, meta)
    mirror_obj, meta = conic_mirror(
        distance=distance,
        y_max=y_total / 2.0,
        x_edge=x_total,
        e=e,
        source_pos=(source_1.x, source_1.y),
        eps_mirror=eps_left,
        return_meta=True,
    )

    if meta is None or ("vertices" not in meta):
        raise RuntimeError(
            "conic_mirror(return_meta=True) did not provide meta['vertices']. "
            "Please update conic_mirror to return polygon vertices in meta."
        )

    # meta['vertices'] expected as list of (x,y) or list of mp.Vector3
    raw_v = meta["vertices"]
    vertices_xy = []
    for v in raw_v:
        if isinstance(v, mp.Vector3):
            vertices_xy.append((v.x, v.y))
        else:
            vertices_xy.append((float(v[0]), float(v[1])))

    # clip to ensure mirror does NOT extend to x > source_1.x (solid angle < 180°)
    x_cut = source_1.x
    left_xy = _clip_polygon_x_leq(vertices_xy, x_cut)
    if len(left_xy) < 3:
        raise RuntimeError(
            "After clipping at x=source_x, the left mirror polygon collapsed. "
            "Try increasing y_total/x_total or check conic_mirror vertex order."
        )

    # rebuild left prism from clipped vertices
    left_vertices = [mp.Vector3(x, y) for (x, y) in left_xy]
    left_mirror = mp.Prism(
        vertices=left_vertices,
        height=mp.inf,
        material=mp.Medium(epsilon=eps_left),
    )

    # ------ mirror the clipped left half to create the right half ------
    right_xy = _mirror_vertices_about_x(left_xy, x_cut)
    right_vertices = [mp.Vector3(x, y) for (x, y) in right_xy]
    right_mirror = mp.Prism(
        vertices=right_vertices,
        height=mp.inf,
        material=mp.Medium(epsilon=eps_right),
    )

    geometry = [left_mirror, right_mirror, fiber_core]

    # ------ define source ------
    sources = [
        mp.Source(
            mp.GaussianSource(frequency=1.0, fwidth=0.2),
            component=mp.Ez,
            center=source_1,
        )
    ]

    # ------ define simulation ------
    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[mp.PML(1.0)],
        geometry=geometry,
        sources=sources,
        resolution=resolution,
    )

    # ------ flux region ------
    flux_rigion_width = 2 * (flux_region_center_x - source_1.x) * np.tan(np.pi / 12.0)
    flux_gather_points = 21
    f0 = 1.0
    df = 0.1

    flux_region = mp.FluxRegion(
        center=mp.Vector3(flux_region_center_x, 0),
        size=mp.Vector3(0, flux_rigion_width),
    )
    flux_obj = sim.add_flux(f0, df, flux_gather_points, flux_region)

    # ------ run ------
    sim.run(until=sim_time)

    # ------ gather flux ------
    flux_vals = mp.get_fluxes(flux_obj)
    phi_avg = sum(flux_vals) / len(flux_vals)
    phi_center = flux_vals[flux_gather_points // 2]
    BW_ratio = phi_avg / phi_center if phi_center != 0 else np.nan
    print(f"[info] avg flux={phi_avg:.3e}, center flux={phi_center:.3e}, BW={BW_ratio:.2f}")

    return phi_avg
