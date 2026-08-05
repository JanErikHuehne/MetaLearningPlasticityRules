"""
STDP parameter sweep -- single-simulation worker script for SLURM array jobs.

Each SLURM array task calls this script once. It reads its own row from
sweep_params.csv (indexed by SLURM_ARRAY_TASK_ID), runs one simulation to
convergence (or to the max_steps ceiling), and writes results/sim_XXXXX.npz.

Usage (normally invoked via run_sweep.sh, not directly):
    python stdp_sweep.py
"""
import os
import time
from pathlib import Path

import brian2 as b2
from brian2.units import *
import numpy as np
import pandas as pd
from collections import deque


task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
b2.prefs.codegen.runtime.cython.cache_dir = f"/tmp/brian2_cache_{task_id}"
b2.prefs.codegen.target = 'cython'

RESULTS_DIR = Path("/nas/ge64qic/shared/Luis2026")
PARAMS_FILE = Path("sweep_params.csv")

#FIxed Parameters:
gl = 10 * nS
er = -80 * mV
el = -60 * mV
vt = -50 * mV
memc = 200 * pfarad
gmax = 2
lr = 1e-2
tau_gaba = 10.0 * ms
tau_ampa = 5.0 * ms

r1e, r1i = 15, 3
r2e, r2i = 15, 3
r3e, r3i = 15, 3
input_num_e = 100
input_num_i = 20

#Variable Parameters: 
param_names = ['Aplus', 'Aminus', 'tau_plus', 'tau_minus', 'alpha_pre', 'alpha_post']

#Convergence Settings 
CHUNK_DURATION = 1 * second     #Split simulation up into small steps 
MAX_STEPS = 12000                 #Max. simulation time      
EPSILON = 1e-3                  #Early Stopping/Convergence Criterion


def build_network(params):
    # This uses numpy backend in Brian, try using CPython -> benchmarken 
    b2.start_scope()

    #Import from our csv file 
    Aplus = params['Aplus']
    Aminus = params['Aminus']
    tau_plus = params['tau_plus'] * ms
    tau_minus = params['tau_minus'] * ms
    alpha_pre = params['alpha_pre']
    alpha_post = params['alpha_post']

    #Diff. Equation
    eqs = """
        dv/dt = (-gl*(v - el) - (g_ampa*v + g_gaba*(v - er))) / memc : volt (unless refractory)
        dg_ampa/dt = -g_ampa / tau_ampa : siemens
        dg_gaba/dt = -g_gaba / tau_gaba : siemens
    """
    neurons = b2.NeuronGroup(3, eqs, threshold="v>vt", reset='v=el', method='euler', refractory=5*ms)
    neurons.v = el
    n1, n2, n3 = neurons[0], neurons[1], neurons[2]

    synapse_model = """
        w : 1
        dtrace_pre_/dt = -trace_pre_ / tau_plus : 1 (event-driven)
        dtrace_post_/dt = -trace_post_ / tau_minus : 1 (event-driven)
    """
    connections = b2.Synapses(
        neurons, model=synapse_model,
        on_pre="""
            g_ampa += w * nS
            trace_pre_ += 1.0
            w = clip(w + lr * (alpha_pre - Aminus * trace_post_), 0, gmax)
        """,
        on_post="""
            trace_post_ += 1
            w = clip(w + lr * (alpha_post + Aplus * trace_pre_), 0, gmax)
        """,
        namespace=dict(Aplus=Aplus, Aminus=Aminus, alpha_pre=alpha_pre,
                        alpha_post=alpha_post, lr=lr, gmax=gmax,
                        tau_plus=tau_plus, tau_minus=tau_minus),
    )
    connections.connect(condition="i != j")

    #Random Initial Weights in [0.1, 0.2]
    connections.w = np.random.uniform(low=0.1, high=0.2, size=len(connections.w))

    #Poisson Input Neuron 1 
    n1_exc = b2.PoissonGroup(input_num_e, rates=r1e*Hz)
    n1_exc_syn = b2.Synapses(n1_exc, n1, on_pre="g_ampa += 1.5*nS"); n1_exc_syn.connect(p=0.1)
    n1_inh = b2.PoissonGroup(input_num_i, rates=r1i*Hz)
    n1_inh_syn = b2.Synapses(n1_inh, n1, on_pre="g_gaba += 3*nS"); n1_inh_syn.connect(p=0.1)

    #Poisson Input Neuron 2 
    n2_exc = b2.PoissonGroup(input_num_e, rates=r2e*Hz)
    n2_exc_syn = b2.Synapses(n2_exc, n2, on_pre="g_ampa += 1.5*nS"); n2_exc_syn.connect(p=0.1)
    n2_inh = b2.PoissonGroup(input_num_i, rates=r2i*Hz)
    n2_inh_syn = b2.Synapses(n2_inh, n2, on_pre="g_gaba += 3*nS"); n2_inh_syn.connect(p=0.1)

    #Poisson Input Neuron 3 
    n3_exc = b2.PoissonGroup(input_num_e, rates=r3e*Hz)
    n3_exc_syn = b2.Synapses(n3_exc, n3, on_pre="g_ampa += 1.5*nS"); n3_exc_syn.connect(p=0.1)
    n3_inh = b2.PoissonGroup(input_num_i, rates=r3i*Hz)
    n3_inh_syn = b2.Synapses(n3_inh, n3, on_pre="g_gaba += 3*nS"); n3_inh_syn.connect(p=0.1)

    #Collect everything to the network 
    network = b2.Network(
        neurons, connections,
        n1_exc, n1_exc_syn, n1_inh, n1_inh_syn,
        n2_exc, n2_exc_syn, n2_inh, n2_inh_syn,
        n3_exc, n3_exc_syn, n3_inh, n3_inh_syn,
    )
    return network, connections



def run_until_converged(network, connections, chunk_duration=CHUNK_DURATION,
                          epsilon_mad=5e-4, max_steps=MAX_STEPS, window_size=30):
    history = deque(maxlen=window_size)                                 #Sets/Manages length of the sliding window 
    curr_w = np.array(connections.w)                    
    history.append(curr_w)                                              #Add initial weights to history 

    for step in range(max_steps):
        network.run(chunk_duration)                                     #Run for 1s                          
        curr_w = np.array(connections.w)                               
        history.append(curr_w)                                          #Add current weights to history  

        if len(history) == window_size:
            window_arr = np.array(history)                              # (30, 6)
            median = np.median(window_arr, axis=0)                      # (6,) ->  per synapse, calculate median over window
            mad = np.median(np.abs(window_arr - median), axis=0)        # (6,) ->  per synapse, calculate Median Absolute Deviation 
            if np.max(mad) < epsilon_mad:                               # Stop if largest MAD is below the epsilon 
                return curr_w, True, step + 1                           # Return the final weights and the steps until convergence 
    return curr_w, False, max_steps


def run_single_simulation(sim_id, params, outdir=RESULTS_DIR, overwrite=False):
    sim_label = f"{sim_id:05d}" if isinstance(sim_id, int) else str(sim_id)
    outfile = outdir / f"sim_{sim_label}.npz"                                   #Builds the filename
    if outfile.exists() and not overwrite:                              
        print(f"sim {sim_label} already done, skipping")
        return None                                                             #If filename already exists go to next 

    t0 = time.time()                                                            #time the whole run
    network, connections = build_network(params)                                #build the network 
    final_w, converged, n_steps = run_until_converged(network, connections)     #run the simulation
    elapsed = time.time() - t0                                                  #measure time it took 

    np.savez(                                                                   #Bundle everything into the npz file
        outfile,
        **{k: params[k] for k in param_names},
        final_weights=final_w,
        converged=converged,
        n_steps=n_steps,
        elapsed_seconds=elapsed,
    )                                                                            
    print(f"sim {sim_label}: converged={converged}, n_steps={n_steps}, elapsed={elapsed:.1f}s")
    return dict(sim_id=sim_id, converged=converged, n_steps=n_steps, elapsed=elapsed)


if __name__ == "__main__":
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    df_params = pd.read_csv(PARAMS_FILE)
    row = df_params.iloc[task_id]
    run_single_simulation(task_id, row)