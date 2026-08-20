# generate_sweep.py
import numpy as np
import os

rng = np.random.default_rng(seed=0)  # master seed for reproducible sweep generation

N_SAMPLES = 500
n_seeds = 3

tau_plus_range   = (5e-3, 30e-3)
tau_minus_range  = (5e-3, 30e-3)
Aplus_range      = (-2.0, 2.0)
Aminus_range     = (-2.0, 2.0)
alpha_pre_range  = (-0.1, 0.1)
alpha_post_range = (-0.1, 0.1)

def sample(range_, n):
    return rng.uniform(range_[0], range_[1], n)

tau_plus_vals   = sample(tau_plus_range, N_SAMPLES)
tau_minus_vals  = sample(tau_minus_range, N_SAMPLES)
Aplus_vals      = sample(Aplus_range, N_SAMPLES)
Aminus_vals     = sample(Aminus_range, N_SAMPLES)
alpha_pre_vals  = sample(alpha_pre_range, N_SAMPLES)
alpha_post_vals = sample(alpha_post_range, N_SAMPLES)

os.makedirs("sweep_params", exist_ok=True)
with open("sweep_params/param_list.txt", "w") as f:
    idx = 0
    for i in range(N_SAMPLES):
        for seed in range(n_seeds):
            f.write(f"{tau_plus_vals[i]} {tau_minus_vals[i]} {Aplus_vals[i]} "
                     f"{Aminus_vals[i]} {alpha_pre_vals[i]} {alpha_post_vals[i]} "
                     f"{seed} run_{idx:05d}\n")
            idx += 1

print(f"Generated {N_SAMPLES * n_seeds} parameter combinations "
      f"({N_SAMPLES} unique param sets x {n_seeds} seeds).")