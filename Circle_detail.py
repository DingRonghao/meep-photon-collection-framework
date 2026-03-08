#-----import------
import meep as mp
import os
import time as pytime
import math
import matplotlib.pyplot as plt
import numpy as np
from rate import compute_Q_from_spectrum# import Q calculator
from geom_2D import arc_mirro# import geometry object wrapper function
from utils import make_output_dir # import timestamp wrapper function
output_dir = make_output_dir("my_simulation") # define & creat output file
output_dir_h5 = f"{output_dir}/h5files"#define & creat output subfile
os.makedirs(output_dir_h5, exist_ok=True)
#------size in real world---
wave_length = 1000.0e-9
fiber_ridus = 10.0e-6
S_meep = 2.4e26
#----parameter for control----
wall_1_to_source = 1.3322382
wall_1_open_deg = 293.0557767
flux_rigion_center_x = -3.1909772

wall_1_thick = 1.02
wall_1_smoothness = 2.27
wall_1_epsilon = 4.66

#------define space and objects------
cell_x = 32
cell_y = 16
cell_z = 0
cell = mp.Vector3(cell_x, cell_y, cell_z) #space
source_1 = mp.Vector3(-10.0,0.0)
resolution = 20

t0 = pytime.perf_counter()

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

# -------define source--------
sources = [
    mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.2), 
                     component=mp.Ez,
                     center=source_1)
]

t1 = pytime.perf_counter()
# --------define simulation--------
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=[mp.PML(1.0)],
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)


sim.use_output_directory(output_dir_h5) # restore sim files by "h5files" subfile

#---------flux record region------
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

t2 = pytime.perf_counter()
#----------simulation run----------
sim.run(mp.at_beginning(mp.output_epsilon),
    mp.at_every(1, mp.output_efield_z),
    mp.at_every(1, mp.output_hfield_x),
    mp.at_every(1, mp.output_hfield_y),
    until=50) 
t3 = pytime.perf_counter()

#--------gather flux-------
#export frequency of each point 
freqs = mp.get_flux_freqs(flux_obj)
# export flux in points as a list
flux_vals = mp.get_fluxes(flux_obj)
# make avrage
phi_avg = sum(flux_vals) / len(flux_vals)
# pick the flux in center point
phi_center = flux_vals[flux_gather_points//2]
# spectreum width
BW_ratio = phi_avg / phi_center
print(f"[info]avg flux={phi_avg:.3e},  center flux={phi_center:.3e},  bandwidth={BW_ratio:.2f}")

#-----anime generate----
from anime_tool import make_ez_gif
gif_path = make_ez_gif(
    eps_vmin = 1.0,eps_vmax = 3.0,
    cell_x=cell_x, cell_y=cell_y,
    resolution = resolution,
    flux_x = flux_rigion_center_x,
    flux_ymin = -flux_rigion_width/2,
    flux_ymax = flux_rigion_width/2 
)
print("[info]GIF finished:", gif_path)
#------plot flux heatmap---
from anime_tool import make_flux_heatmap
flux_heatmap_path = make_flux_heatmap(
    eps_vmin = 1.0,eps_vmax = 3.0,
    cell_x=cell_x, cell_y=cell_y,
    flux_x = flux_rigion_center_x,
    flux_ymin = -flux_rigion_width/2,
    flux_ymax = flux_rigion_width/2 
)
#---plot flux spectrum---
plt.plot(freqs, flux_vals, '-o')
plt.xlabel(f"Frequency(1/μm)")
plt.ylabel('Flux(FDU)') #Flux Density Unit(1FDU=2.4*10^26W/m^2)
plt.title('Collected Flux Spectrum')
plt.savefig(os.path.join(output_dir, "flux.png"))
#-------calculate Q-----
Q_val = compute_Q_from_spectrum(freqs=freqs, spectrum=flux_vals)
print(f"[info]Q value is {Q_val}")
t4 = pytime.perf_counter()

#----time countting---
print(
    "[info]timings: "
    f"geometry_build={t1 - t0:.3f}s, "
    f"simulation_construct={t2 - t1:.3f}s, "
    f"run={t3 - t2:.3f}s, "
    f"plot={t4 - t3:.3f}s, "
    f"TOTAL={t4 - t0:.3f}s"
)


