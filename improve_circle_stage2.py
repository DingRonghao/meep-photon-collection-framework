from scipy.optimize import differential_evolution
from sim_obj import cave_1_mirro,bare_structure,flat_mirror
import numpy as np
#----control panle----
robust_test_samples = 30

maxiter=5#max=20
popsize=5#max=20

scale_ratial=0.2
Input_in_stage1=[
    6.9623,#wall_1_to_source 
    193.5481,#wall_1_open_deg
]
R = scale_ratial*Input_in_stage1[0]
D = Input_in_stage1[1]

R_max,R_min = 1.15*R,0.85*R
D_max,D_min = 1.15*D,0.85*D

# Bounds
BOUNDS = [
    (R_min,R_max),    #wall_1_to_source 
    (D_min,D_max),  # wall_1_open_deg
]
#------make noise(gauss +-5%)---
def sample_noise():
    noise_rate=0.05
    noise = np.array([
        np.random.normal(0.0, noise_rate*R),# R
        np.random.normal(0.0, noise_rate*D),# D
    ])
    return noise

#=====================function for run optimizition=========================
phi_avg_BARE = None
lows  = np.array([b[0] for b in BOUNDS], dtype=float)
highs = np.array([b[1] for b in BOUNDS], dtype=float)
k_noise = 0#Initial noise candidates
def objective(x):
    vals=[]
    x = np.asarray(x, dtype=float)

    # ---- Step 2: run Meep sim with noise ----
    output_object = cave_1_mirro(
        wall_1_to_source=float(x[0]),
        wall_1_open_deg=float(x[1]),
        flux_rigion_center_x = -2.0*scale_ratial,
        cell_x = int(64*scale_ratial),
        cell_y = int(64*scale_ratial),
        source_1 = (-15.0*scale_ratial, 0.0),
    )

    phi_avg = output_object["phi_avg"]
    vals.append(phi_avg / phi_avg_BARE)

    #===================noise=================
    for n in range(k_noise):
        #------Step 1: add noise---
        noise = sample_noise()
        x_noisy = x.copy()
        x_noisy[:2] += noise[:2]

        # ---- Step 2: run Meep sim with noise ----
        output_object = cave_1_mirro(
            wall_1_to_source=float(x_noisy[0]),
            wall_1_open_deg=float(x_noisy[1]),
            flux_rigion_center_x = -2.0*scale_ratial,
            cell_x = int(64*scale_ratial),
            cell_y = int(64*scale_ratial),
            source_1 = (-15.0*scale_ratial, 0.0),
        )

        phi_avg = output_object["phi_avg"]

        vals.append(phi_avg/phi_avg_BARE)

    return -float(np.mean(vals))
#===================function for export best result without noise===============================
def eval_nominal(x):
    x = np.asarray(x, dtype=float)

    output_1 = cave_1_mirro(
        wall_1_to_source=float(x[0]),
        wall_1_open_deg=float(x[1]),
        flux_rigion_center_x = -2.0*scale_ratial,
        cell_x = int(64*scale_ratial),
        cell_y = int(64*scale_ratial),
        source_1 = (-15.0*scale_ratial, 0.0),
    )
    phi_avg = output_1["phi_avg"]
    
    Amp_rate = phi_avg/phi_avg_BARE 
    return Amp_rate,None
#===================function for test robustness of the best result========= 
def eval_robust(x, K=10):
    x = np.asarray(x, dtype=float)
    values = []

    for _ in range(K):
        noise = sample_noise()
        x_noisy = x.copy()
        x_noisy[0] += noise[0]
        x_noisy[1] += noise[1]

        output_2 = cave_1_mirro(
            wall_1_to_source=float(x_noisy[0]),
            wall_1_open_deg=float(x_noisy[1]),
            flux_rigion_center_x = -2.0*scale_ratial,
            cell_x = int(64*scale_ratial),
            cell_y = int(64*scale_ratial),
            source_1 = (-15.0*scale_ratial, 0.0),
        )
        phi_avg = output_2["phi_avg"]
        Amp_rate = phi_avg/phi_avg_BARE 
        values.append(float(Amp_rate))


    values = np.array(values)
    mean_rate = values.mean()
    std_rate = values.std()
    return mean_rate, std_rate

#===========optimizition settings===========================
#----function for minimalizie the object function output----
def run_de_minimal():
    return differential_evolution(
        func=objective,
        bounds=BOUNDS,
        maxiter=maxiter,#max=25
        popsize=popsize,#max=20
        callback=callback,
        polish=False,
        disp=True,
        updating='immediate',     
        workers=1,       #use cores
        seed=1
    )
#==========optimizition schedule=============
gen = 0
schedule = [
    (0, 0),
    (int(0.7 * maxiter), 1),
    (int(0.9 * maxiter), 2),
]

def callback(xk, convergence):
    global gen, k_noise
    gen += 1
    k = schedule[0][1]
    for th, kk in schedule:
        if gen >= th:
            k = kk
    k_noise = k

    return False

#========the main frame of optimizition========
if __name__ == "__main__":#run only if it is a dependent footnote
    
    np.random.seed(12345)
    
    phi_avg_BARE = bare_structure(
        flux_region_center_x = -2.0*scale_ratial,
        cell_x = int(64*scale_ratial),
        cell_y = int(64*scale_ratial),
        source_1 = (-15.0*scale_ratial, 0.0),
        resolution = 20,   
    )
    if (phi_avg_BARE is None) or (not np.isfinite(phi_avg_BARE)) or (phi_avg_BARE <= 0):
        raise ValueError(f"Bad phi_avg_BARE: {phi_avg_BARE}")
    print("baseline done, starting DE...")
    res = run_de_minimal()
    print("DE finished.")
    x_best, rate_best = res.x, -res.fun #restore the best input and output
    phi_avg_flat = flat_mirror(
        flux_region_center_x = -2.0*scale_ratial,
        cell_x = int(64*scale_ratial),
        cell_y = int(64*scale_ratial),
        source_1 = (-15.0*scale_ratial, 0.0),
        resolution = 20,   
        mirror_distance=x_best[0]        
    )
    flat_amp_rate = phi_avg_flat/phi_avg_BARE
    rate_nominal, nom_err = eval_nominal(x_best)
    mean_rate, std_rate = eval_robust(x_best, K=robust_test_samples)
    names = ["wall_1_to_source", "wall_1_open_deg"]
    print("\n=== DE (minimal) RESULT ===")
    for n, v in zip(names, x_best):
        print(f"{n:>20s} = {float(v):.4f}")
    print(f"{'best score':>20s} = {rate_best:.2f}")
    if np.isfinite(rate_nominal):
        print("raw best rate =", f"{rate_nominal:.2%}")
    else:
        print("noise eliminated best rate = FAILED:", nom_err)
    print("compare with flat mirror:=",f"{rate_nominal/flat_amp_rate:.2%}")
    print("\n=======Robustness test=======")

    print(f"WITH noise rate AVERAGE = {mean_rate:.2%}")
    print(f"WITH noise rate standard deviation = {std_rate:.2%}")

    print("===========================\n")