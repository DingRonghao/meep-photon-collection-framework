from scipy.optimize import differential_evolution
from sim_obj import conic_stucture,bare_structure
import numpy as np
#----control panle----
robust_test_samples = 30

maxiter=5#max=20
popsize=5#max=20

scale_ratial=0.2
Input_in_stage1=[
    6.5981,#distance
    10.5441,#y_total
    11.2259,#x_total
    1.0210#e
]
D = scale_ratial*Input_in_stage1[0]
Y = scale_ratial*Input_in_stage1[1]
X = scale_ratial*Input_in_stage1[2]
E = Input_in_stage1[3]
D_max,D_min = 1.15*D,0.85*D
Y_max,Y_min = 1.15*Y,0.85*Y
X_max,X_min = 1.15*X,0.85*X
E_max,E_min = 1.15*E,max(0.85*E,1.0001)
# Bounds
BOUNDS = [
    (D_min,D_max),    # distance
    (Y_min,Y_max),  # y_total   
    (X_min,X_max),   # x_total
    (E_min,E_max),      # e
]
#------make noise(gauss +-5%)---
def sample_noise():
    noise_rate=0.05
    noise = np.array([
        np.random.normal(0.0, noise_rate*D),# distance
        np.random.normal(0.0, noise_rate*Y),# y_total 
        np.random.normal(0.0, noise_rate*X),# x_total
        np.random.normal(0.0, noise_rate*E),# e
    ])
    return noise

#=====================function for run optimizition=========================
phi_avg_BARE = None
PENALTY = 1e6
lows  = np.array([b[0] for b in BOUNDS], dtype=float)
highs = np.array([b[1] for b in BOUNDS], dtype=float)
k_noise = 0#Initial noise candidates
def objective(x):
    fails = 0 #Invalid candidates
    x = np.asarray(x, dtype=float)
    vals=[]
    # bare sanity
    if (phi_avg_BARE is None) or (not np.isfinite(phi_avg_BARE)) or (phi_avg_BARE <= 0):
        return PENALTY
    #================no noise==============
    # ---- Step 1: run Meep sim without noise ----
    try:
        phi_avg = conic_stucture(
            distance=float(x[0]),
            y_total =float(x[1]),
            x_total =float(x[2]),
            e       =float(x[3]),
            flux_region_center_x = -2.0*scale_ratial,
            cell_x = int(64*scale_ratial),
            cell_y = int(64*scale_ratial),
            source_1 = (-15.0*scale_ratial, 0.0),
        )
    except Exception as e:
        return PENALTY

    if (phi_avg is None) or (not np.isfinite(phi_avg)):
        return PENALTY
    if k_noise == 0:
        return - phi_avg / phi_avg_BARE
    vals.append(phi_avg / phi_avg_BARE)
    #===================noise=================
    for n in range(k_noise):
        #------Step 1: add noise---
        noise = sample_noise()
        x_noisy = x.copy()
        x_noisy[:4] += noise[:4]
        x_noisy = np.clip(x_noisy, lows, highs)

        # ---- Step 2: run Meep sim with noise ----
        try:
            phi_avg = conic_stucture(
                distance=float(x_noisy[0]),
                y_total =float(x_noisy[1]),
                x_total =float(x_noisy[2]),
                e       =float(x_noisy[3]),
                flux_region_center_x = -2.0*scale_ratial,
                cell_x = int(64*scale_ratial),
                cell_y = int(64*scale_ratial),
                source_1 = (-15.0*scale_ratial, 0.0),
            )
        except Exception as e:
            fails +=1
            continue

        if (phi_avg is None) or (not np.isfinite(phi_avg)):
            fails +=1
            continue
        vals.append(phi_avg/phi_avg_BARE)

    pass_rate = 1-fails/k_noise
    if pass_rate <= 0.5:
        return PENALTY#eliminate pass rate <=50%

    return -float(np.mean(vals)*pass_rate)

#===================function for export best result without noise===============================
def eval_nominal(x):
    x = np.asarray(x, dtype=float)

    try:
        phi_avg = conic_stucture(
            distance=float(x[0]),
            y_total =float(x[1]),
            x_total =float(x[2]),
            e       =float(x[3]),
            flux_region_center_x = -2.0*scale_ratial,
            cell_x = int(64*scale_ratial),
            cell_y = int(64*scale_ratial),
            source_1 = (-15.0*scale_ratial, 0.0),
        )
    except Exception as e:
        return np.nan,str(e)
    
    if (phi_avg is None) or (not np.isfinite(phi_avg)):
        return np.nan, "phi_avg is not finite"

    Amp_rate = phi_avg / phi_avg_BARE
    if (not np.isfinite(Amp_rate)) or (Amp_rate <= 0):
        return np.nan, "rate invalid"
    
    # ----compute rate----
    Amp_rate = phi_avg/phi_avg_BARE 
    return Amp_rate,None

#===================function for test robustness of the best result========= 
def eval_robust(x, K=10):
    x = np.asarray(x, dtype=float)
    values = []
    fail_msgs = []
    n_fail = 0

    # randomly try different noise for K times
    for _ in range(K):
        try:
            noise = sample_noise()
            x_noisy = x.copy()
            x_noisy[0] += noise[0]
            x_noisy[1] += noise[1]
            x_noisy[2] += noise[2]
            x_noisy[3] += noise[3]

            phi_avg = conic_stucture(
                distance=float(x_noisy[0]),
                y_total =float(x_noisy[1]),
                x_total =float(x_noisy[2]),
                e       =float(x_noisy[3]),
                flux_region_center_x = -2.0*scale_ratial,
                cell_x = int(64*scale_ratial),
                cell_y = int(64*scale_ratial),
                source_1 = (-15.0*scale_ratial, 0.0),
            )
        except Exception as e:
            n_fail += 1
            if len(fail_msgs) < 3:
                fail_msgs.append(str(e))
            continue

        if (phi_avg is None) or (not np.isfinite(phi_avg)):
            n_fail += 1
            if len(fail_msgs) < 3:
                fail_msgs.append("phi_avg not finite")
            continue
            # ----compute rate----
        Amp_rate = phi_avg/phi_avg_BARE 
        if (not np.isfinite(Amp_rate)) or (Amp_rate <= 0):
            n_fail += 1
            if len(fail_msgs) < 3:
                fail_msgs.append("rate invalid")
            continue 
        values.append(float(Amp_rate))

    n_ok =len(values)
    fail_prob = n_fail/K
    if n_ok == 0:
        return np.nan, np.nan, n_ok, fail_prob, fail_msgs
    
    values = np.array(values)
    mean_rate = values.mean()
    std_rate = values.std()
    return mean_rate, std_rate,n_ok,fail_prob,fail_msgs

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
        seed=42
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

    res = run_de_minimal()
    x_best, rate_best = res.x, -res.fun #restore the best input and output
    rate_nominal, nom_err = eval_nominal(x_best)
    mean_rate, std_rate, n_ok, fail_prob, fail_msgs = eval_robust(x_best, K=robust_test_samples)
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
    if n_ok > 0:
        print(f"WITH noise rate AVERAGE = {mean_rate:.2%}")
        print(f"WITH noise rate standard deviation = {std_rate:.2%}")
    else:
        print("WITH noise rate AVERAGE = FAILED (no valid samples)")
    print(f"robust valid = {n_ok}/{robust_test_samples}, fail_prob = {fail_prob:.2%}")
    if fail_msgs:
        print("example failures:")
        for m in fail_msgs:
            print("  -", m)
    print("===========================\n")