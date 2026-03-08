import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
import os
import glob
import inspect

def make_ez_gif(
        eps_vmin = 1.0,eps_vmax = 3.0,
        cell_x=320, cell_y=160,# [μm] according to the sim setting
        resolution = 20.0,# [pixcel/μm] according to the sim setting
        flux_x = - 3,
        flux_ymin = -3,
        flux_ymax = 3, 
):
    
    caller_path = inspect.stack()[1].filename       
    source_filename = os.path.splitext(os.path.basename(caller_path))[0]  

    def _find_latest_run(parent_dir="output", pattern="my_simulation_*"):
        cands = [p for p in glob.glob(os.path.join(parent_dir, pattern)) if os.path.isdir(p)]
        if not cands:
            raise FileNotFoundError(f"in {parent_dir} not found {pattern}")
        return max(cands, key=os.path.getmtime)
    
    folder = _find_latest_run(parent_dir="output", pattern="my_simulation_*")
    print("[info]anime tool find recent files:", folder)
    h5_dir = os.path.join(folder, "h5files")
    if not os.path.isdir(h5_dir):
        raise FileNotFoundError(f"[warnning]not find h5 files: {h5_dir}")

    prefix = "ez"
    
    n_frames = len(glob.glob(os.path.join(h5_dir, f"{source_filename}-{prefix}-*.00.h5")))
    if n_frames == 0:
        raise FileNotFoundError("no frames found")
    print(f"[info]gif detected frames: {n_frames}")


    with h5py.File(os.path.join(h5_dir, f"{source_filename}-{prefix}-000001.00.h5"), "r") as f:
        ez = f[prefix][:]
        shape = ez.shape
        

    with h5py.File(os.path.join(h5_dir, f"{source_filename}-eps-000000.00.h5"), "r") as f:
        eps = f["eps"][:]


    fig, ax = plt.subplots()


    norm = colors.Normalize(vmin=eps_vmin, vmax=eps_vmax,clip=True)
    eps_im = ax.imshow(eps.T, cmap="gray_r",norm=norm, alpha=0.8,
                    extent=[-cell_x/2, cell_x/2, -cell_y/2, cell_y/2],
                    origin="lower",aspect="equal", zorder=1) 

    plt.colorbar(eps_im, label="ε") 

    ax.vlines( flux_x,flux_ymin, flux_ymax, colors='lime', linestyles='--', linewidth=2, zorder=9) 
    ax.text(flux_x + 1,flux_ymin + 1, "Flux Region", color='green', fontsize=10, zorder=11)

    im = ax.imshow(np.zeros(shape).T, vmin=-0.5, vmax=0.5, cmap='seismic', animated=True,
                extent=[-cell_x/2, cell_x/2, -cell_y/2, cell_y/2],  
                origin="lower",aspect="equal",zorder=0)
    plt.colorbar(im, label="Re(Ez)")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")

    def update(frame):
        filename = os.path.join(h5_dir, f"{source_filename}-{prefix}-{frame+1:06d}.00.h5")
        with h5py.File(filename, "r") as f:
            ez = f[prefix][:]
            
        im.set_data(ez.T)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=True)
    print("[info]epsilon min/max:", eps.min(), eps.max()) 

    gif_path = os.path.join(folder, "ez_animation.gif")
    ani.save(gif_path, writer="pillow", fps=10)
    plt.close(fig)
    return gif_path

#======================================================================

def make_flux_heatmap(
        eps_vmin=1.0, eps_vmax=3.0,
        cell_x=16.0, cell_y=8.0,   # μm
        avg_ratio=0.7,            
        vmin=None, vmax=None,       
        flux_x = - 3,
        flux_ymin = -3,
        flux_ymax = 3 
):
    
    caller_path = inspect.stack()[1].filename
    source_filename = os.path.splitext(os.path.basename(caller_path))[0]

    def _find_latest_run(parent_dir="output", pattern="my_simulation_*"):
        cands = [p for p in glob.glob(os.path.join(parent_dir, pattern)) if os.path.isdir(p)]
        if not cands:
            raise FileNotFoundError(f"in {parent_dir} not found {pattern}")
        return max(cands, key=os.path.getmtime)

    folder = _find_latest_run(parent_dir="output", pattern="my_simulation_*")
    print("[info] heatmap confirm path:", folder)
    h5_dir = os.path.join(folder, "h5files")
    if not os.path.isdir(h5_dir):
        raise FileNotFoundError(f"[error]not find h5 files: {h5_dir}")

    # —— TM: Ez/Hx/Hy ——
    def frames_of(prefix):
        return sorted(glob.glob(os.path.join(h5_dir, f"{source_filename}-{prefix}-*.00.h5")))
    ez_list = frames_of("ez")
    hx_list = frames_of("hx")
    hy_list = frames_of("hy")
    if not ez_list or not hx_list or not hy_list:
        raise FileNotFoundError("[error]no engugh ez/hx/hy h5 files（for TM mode）")

    n = min(len(ez_list), len(hx_list), len(hy_list))
    ez_list, hx_list, hy_list = ez_list[:n], hx_list[:n], hy_list[:n]
    print(f"[info] heatmap find frames: {n}")

    with h5py.File(ez_list[0], "r") as f:
        ez0 = f["ez"][:]
    shape = ez0.shape

   
    k0 = max(0, int(n * avg_ratio))
    sel = range(k0, n)
    if len(sel) == 0:
        sel = range(n) 
    print(f"[info] heatmap use frames: {k0}..{n-1}（all frames: {len(sel)} ）")


    # —— （TM: S = 1/2 * Re( Ez * ẑ × H* ) = 1/2 * Re( -Ez*Hy, +Ez*Hx ) ）——
    Sx_acc = np.zeros(shape, dtype=np.float64)
    Sy_acc = np.zeros(shape, dtype=np.float64)

    for i in sel:
        with h5py.File(ez_list[i], "r") as f: ez = f["ez"][:]
        with h5py.File(hx_list[i], "r") as f: hx = f["hx"][:]
        with h5py.File(hy_list[i], "r") as f: hy = f["hy"][:]

        if np.iscomplexobj(ez) or np.iscomplexobj(hx) or np.iscomplexobj(hy):
            Sx_acc += 0.5 * np.real(-ez * np.conj(hy))
            Sy_acc += 0.5 * np.real( ez * np.conj(hx))
        else:
            Sx_acc += -0.5 * (ez * hy)
            Sy_acc +=  0.5 * (ez * hx)

    Sx = Sx_acc / len(sel)
    Sy = Sy_acc / len(sel)
    Smag = np.sqrt(Sx**2 + Sy**2)

    eps = None
    eps_path = os.path.join(h5_dir, f"{source_filename}-eps-000000.00.h5")
    if os.path.exists(eps_path):
        with h5py.File(eps_path, "r") as f:
            eps = f["eps"][:]
        print("[info] epsilon min/max:", float(eps.min()), float(eps.max()))

    extent = [-cell_x/2, cell_x/2, -cell_y/2, cell_y/2]

    def save_heat(img, name, cmap="inferno", add_eps=False):
        fig, ax = plt.subplots()
        if add_eps and eps is not None:

            norm = colors.Normalize(vmin=eps_vmin, vmax=eps_vmax,clip=True)
            eps_im = ax.imshow(eps.T, cmap="gray_r",norm=norm, alpha=0.65,
                extent=[-cell_x/2, cell_x/2, -cell_y/2, cell_y/2],
                origin="lower",aspect="equal", zorder=1) 
            plt.colorbar(eps_im, label="ε",location="left") 


        if name in ("Sx", "Sy") and (vmin is None and vmax is None):
            A = float(np.nanmax(np.abs(img)))
            vmin_use, vmax_use = -A, +A
        else:
            vmin_use, vmax_use = vmin, vmax

        im = ax.imshow(img.T, origin="lower", extent=extent, aspect="equal",
               cmap=("seismic" if name in ("Sx","Sy") else "jet"),
               vmin=vmin_use, vmax=vmax_use, alpha=1.0,zorder=0)
        
        ax.vlines( flux_x,flux_ymin, flux_ymax, colors='lime', linestyles='--', linewidth=2, zorder=9) 
        ax.text(flux_x + 1,flux_ymin + 1, "Flux Region", color='green', fontsize=10, zorder=11)

        plt.colorbar(im, label=f"{name}(FDU)")
        ax.set_xlabel("x (μm)")
        ax.set_ylabel("y (μm)")
        out = os.path.join(folder, f"flux_{name}.png")
        plt.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return out

    out_abs = save_heat(Smag, "S_abs",add_eps=True) #|S|
    out_sx  = save_heat(Sx,   "Sx",    cmap="seismic",add_eps=True) # Sx
    out_sy  = save_heat(Sy,   "Sy",    cmap="seismic",add_eps=True) # Sy
    return {"S_abs": out_abs, "Sx": out_sx, "Sy": out_sy}
