from scipy.optimize import differential_evolution
from sim_obj import cave_1_mirro,bare_structure
import numpy as np
#----control panle----
wight_phi=5.0
wight_purcell = 1.0
Veff_threshold = 0.2
robust_test_samples = 30

maxiter=5#max=20
popsize=4#max=20
#------make noise(gauss +-5%)---
def sample_noise():
    noise = np.array([
        np.random.normal(0.0, 0.3),# wall_1_to_source 
        np.random.normal(0.0, 3.0),# wall_1_open_deg 
        np.random.normal(0.0, 0.0)# flux_rigion_center_x
    ])
    return noise
#===========optimizition settings===========================
# Bounds
BOUNDS = [
    (6.5,  10.0),    # wall_1_to_source    
    (40.0, 300.0),  # wall_1_open_deg   
    (-2.0, -2.0)    # flux_rigion_center_x
]
#=====================function for run optimizition=========================
phi_avg_BARE = None
k_noise = 0#Initial noise candidates
def objective(x):
    vals=[]
    x = np.asarray(x, dtype=float)

    # ---- Step 2: run Meep sim with noise ----
    output_object = cave_1_mirro(
        wall_1_to_source=float(x[0]),
        wall_1_open_deg=float(x[1]),
        flux_rigion_center_x=float(x[2])
    )

    phi_avg = output_object["phi_avg"]
    vals.append(phi_avg / phi_avg_BARE)

    #===================noise=================
    for n in range(k_noise):
        #------Step 1: add noise---
        noise = sample_noise()
        x_noisy = x.copy()
        x_noisy[:4] += noise[:4]

        # ---- Step 2: run Meep sim with noise ----
        output_object = cave_1_mirro(
            wall_1_to_source=float(x_noisy[0]),
            wall_1_open_deg=float(x_noisy[1]),
            flux_rigion_center_x=float(x_noisy[2])
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
        flux_rigion_center_x=float(x[2]),
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
        x_noisy[2] += noise[2]

        output_2 = cave_1_mirro(
            wall_1_to_source=float(x_noisy[0]),
            wall_1_open_deg=float(x_noisy[1]),
            flux_rigion_center_x=float(x_noisy[2]),
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
        flux_region_center_x = -2.0,
        cell_x = 64,
        cell_y = 32,
        source_1 = (-15.0, 0.0),
        resolution = 20,    
    )
    if (phi_avg_BARE is None) or (not np.isfinite(phi_avg_BARE)) or (phi_avg_BARE <= 0):
        raise ValueError(f"Bad phi_avg_BARE: {phi_avg_BARE}")
    print("baseline done, starting DE...")
    res = run_de_minimal()
    print("DE finished.")
    x_best, rate_best = res.x, -res.fun #restore the best input and output
    rate_nominal, nom_err = eval_nominal(x_best)
    mean_rate, std_rate = eval_robust(x_best, K=robust_test_samples)
    names = ["distance","y_total","x_total","e"]
    print("\n=== DE (minimal) RESULT ===")
    for n, v in zip(names, x_best):
        print(f"{n:>20s} = {float(v):.4f}")
    print(f"{'best score':>20s} = {rate_best:.2f}")
    if np.isfinite(rate_nominal):
        print("raw best rate =", f"{rate_nominal:.2%}")
    else:
        print("noise eliminated best rate = FAILED:", nom_err)
    print("\n=======Robustness test=======")

    print(f"WITH noise rate AVERAGE = {mean_rate:.2%}")
    print(f"WITH noise rate standard deviation = {std_rate:.2%}")

    print("===========================\n")