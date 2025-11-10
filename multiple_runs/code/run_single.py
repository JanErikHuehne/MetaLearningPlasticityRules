import brian2 as b2
import numpy as np
from brian2.units import *
from matplotlib import pyplot as plt
import os 
import secrets
import string
import json 
from pathlib import Path
"""def random_filename(length=12, ext=None, alphabet_only=True):
    if alphabet_only:
        alphabet = string.ascii_letters
    else:
        alphabet = string.ascii_letters + string.digits
    name = ''.join(secrets.choice(alphabet) for _ in range(length)).lower()
    return f"{name}{ext if ext else ''}"
nas_path = str(os.environ['nas_path'])"""


path = os.environ['SAMPLE']

with open(path, 'rb') as f:
    parameters = json.load(f)['parameters']
for n in range(20):
    b2.prefs.codegen.target = "numpy"
    b2.start_scope()
    # set random seed
    b2.seed(42)


    NE = 800
    NI = NE // 4
    epsilon = 0.1  # fixed connection sparsity

    K_ee = int(epsilon * NE)
    K_ie = int(epsilon * NE)
    K_ei = int(epsilon * NI)
    K_ii = int(epsilon * NI)

    scale_ee = 1.0 / np.sqrt(K_ee)
    scale_ie = 1.0 / np.sqrt(K_ie)
    scale_ei = 1.0 / np.sqrt(K_ei)
    scale_ii = 1.0 / np.sqrt(K_ii)


    # ----------------------------
    # Neuron Parameters
    # ----------------------------
    gl = 10 * nS
    er = -80 * mV
    el = -60 * mV
    vt = -50 * mV
    tau_gaba = 10.0 * ms
    tau_ampa = 5.0 * ms
    memc = 200 * pfarad
    gmax = 2
    lr = 1e-3

    eqs = """
            dv/dt = (-gl*(v - el) - (g_ampa*v + g_gaba*(v - er))) / memc : volt (unless refractory)
            dg_ampa/dt = -g_ampa / tau_ampa : siemens
            dg_gaba/dt = -g_gaba / tau_gaba : siemens
        """

    neurons = b2.NeuronGroup(
        NE + NI, eqs, threshold="v>vt", reset="v=el", method="euler", refractory=5 * ms
    )
    neurons.v = el
    Pe = neurons[:NE]
    Pi = neurons[NE:]


    # EE plasticity parameters
    ee_alpha_pre = parameters[0]
    ee_alpha_post = parameters[1]
    ee_Aplus = parameters[2]
    ee_tauplus_stdp = parameters[3] * ms
    ee_tauminus_stdp =  parameters[4] * ms
    factor_ee =  parameters[5]
    ee_Aminus = -1.0

    # Synapse model with an `active` parameter±
    synapse_model = """
            w : 1  # Weight
            syn_status : integer  # Active or inactive status (1 or 0)
            dee_trace_pre_plus/dt = -ee_trace_pre_plus / ee_tauplus_stdp : 1 (event-driven)
            dee_trace_pre_minus/dt = -ee_trace_pre_minus / ee_tauminus_stdp : 1 (event-driven)
            dee_trace_post_plus/dt = -ee_trace_post_plus / ee_tauplus_stdp : 1 (event-driven)
            dee_trace_post_minus/dt = -ee_trace_post_minus / ee_tauminus_stdp : 1 (event-driven)
        """

    # Define EE synapses with sparsity and `active` parameter
    con_ee = b2.Synapses(
        Pe,
        Pe,
        model=synapse_model,
        on_pre="""
                                g_ampa += w * nS * syn_status  # Only contribute if active
                                ee_trace_pre_plus += 1.0
                                ee_trace_pre_minus += 1.0
                                w = clip(w + lr * (ee_alpha_pre + factor_ee * ee_Aplus * ee_trace_post_plus + factor_ee * ee_Aminus * ee_trace_post_minus), 0, gmax)
                            """,
        on_post="""
                                ee_trace_post_plus += 1
                                ee_trace_post_minus += 1
                                w = clip(w + lr * (ee_alpha_post + ee_Aplus * ee_trace_pre_plus + ee_Aminus * ee_trace_pre_minus), 0, gmax)
                            """,
    )

    con_ee.connect(condition="i != j")  # Fully connect, except self-connections
    con_ee.w = np.random.uniform(low=0.1, high=0.2, size=len(con_ee.w)) 
    con_ee.syn_status = np.random.choice(
        [0.0, 1.0], size=len(con_ee.w), p=[1 - epsilon, epsilon]
    )

    # Introduce initial sparsity
    # EI and II synapses
    con_ei = b2.Synapses(Pe, Pi, model="w : 1", on_pre="g_ampa += w * nS")
    con_ii = b2.Synapses(Pi, Pi, model="w : 1", on_pre="g_gaba += w * nS")
    con_ei.connect(p=epsilon, condition="i != j")
    con_ii.connect(p=epsilon, condition="i != j")
    con_ei.w = np.random.uniform(low=0.1, high=0.2, size=len(con_ei.w)) 
    con_ii.w = np.random.uniform(low=0.1, high=0.2, size=len(con_ii.w)) 

    # IE Plasticity parameters and model
    ie_alpha_pre = parameters[6]
    ie_alpha_post = parameters[7]
    ie_Aplus = parameters[8]
    ie_tauplus_stdp = parameters[9] * ms
    ie_tauminus_stdp = parameters[10] * ms
    factor_ie = parameters[11]
    ie_Aminus = -1.0


    synapse_model = """
            w : 1  # Weight
            syn_status : integer  # Active or inactive status (1 or 0)
            die_trace_pre_plus/dt = -ie_trace_pre_plus / ie_tauplus_stdp : 1 (event-driven)
            die_trace_pre_minus/dt = -ie_trace_pre_minus / ie_tauminus_stdp : 1 (event-driven)
            die_trace_post_plus/dt = -ie_trace_post_plus / ie_tauplus_stdp : 1 (event-driven)
            die_trace_post_minus/dt = -ie_trace_post_minus / ie_tauminus_stdp : 1 (event-driven)
        """
    con_ie = b2.Synapses(
        Pi,
        Pe,
        model=synapse_model,
        on_pre="""
                                        g_gaba += w*nS * syn_status
                                        ie_trace_pre_plus += 1.0
                                        ie_trace_pre_minus += 1.0
                                        w = clip(w + lr * (ie_alpha_pre +factor_ie*ie_Aplus * ie_trace_post_plus + factor_ie*ie_Aminus * ie_trace_post_minus), 0, gmax)
                                        """,
        on_post="""
                                        ie_trace_post_plus += 1
                                        ie_trace_post_minus += 1
                                        w = clip(w + lr * (ie_alpha_post + ie_Aplus * ie_trace_pre_plus + ie_Aminus * ie_trace_pre_minus), 0, gmax)
                                        """,
    )
    con_ie.connect(condition="i != j")  # Fully connect, except self-connections
    con_ie.w = np.random.uniform(low=0.1, high=0.2, size=len(con_ie.w)) 
    con_ie.syn_status = np.random.choice(
        [0.0, 1.0], size=len(con_ie.w), p=[1 - epsilon, epsilon]
    )

    # ----------------------------
    # Input (Scaled to NE) — EXC
    # ----------------------------
    input_num = int(NE * 1.25)
    desired_in_degree = 30

    p_in = desired_in_degree / input_num

    P_ext_E = b2.PoissonGroup(input_num, rates=20 * Hz)
    S_input_E = b2.Synapses(P_ext_E, Pe, on_pre="g_ampa += 0.5 * nS")
    S_input_E.connect(p=p_in)


    # ----------------------------
    # Input (Scaled to NI) — INH
    # ----------------------------
    input_num_I = int(NI * 1.25)
    desired_in_degree_I = 30
    p_in_I = desired_in_degree_I / input_num_I

    P_ext_I = b2.PoissonGroup(input_num_I, rates=20 * Hz)
    S_input_I = b2.Synapses(P_ext_I, Pi, on_pre="g_ampa += 0.5 * nS")
    S_input_I.connect(p=p_in_I)

    # Threshold and pruning set
    threshold = 0.005
    # con_ee.add_variable('prune_count', 0, is_scalar_type=True)  # Initialize pruning count for each synapse
    # con_ie.add_variable('prune_count', 0, is_scalar_type=True)
    ee_total_connected = np.sum(con_ee.syn_status)
    ie_total_connected = np.sum(con_ie.syn_status)
    # print("EE TOTAL ", ee_total_connected,"IE TOTAL ", ie_total_connected)
    pruned = [0, 0]


    @b2.network_operation(dt=50 * ms)
    def structural_plasticity():
        epsilon = 0.01
        for k, synapse in enumerate([con_ee, con_ie]):
            weights = synapse.w[:]
            syn_status = synapse.syn_status[:]
            active_synapses = syn_status == 1
            to_remove_indices = np.nonzero((weights < threshold) & (active_synapses))[0]
            pruned[k] += len(to_remove_indices)
            # we calculate the weight of each neuron to receive a new incoming or outgoing connection
            if len(to_remove_indices):

                if k == 0:
                    outgoing_counts = np.bincount(synapse.i[active_synapses], minlength=NE)
                    incoming_counts = np.bincount(synapse.j[syn_status == 1], minlength=NE)
                elif k == 1:
                    outgoing_counts = np.bincount(synapse.i[active_synapses], minlength=NI)
                    incoming_counts = np.bincount(synapse.j[active_synapses], minlength=NE)
                inactive_indices = np.nonzero(syn_status == 0)[0]
                # Selection probabilities for each inactive synapse
                # We get the pre and post indices of inactive synapses
                pre_indices = synapse.i[inactive_indices]
                post_indices = synapse.j[inactive_indices]

                pre_weights = 1 / (outgoing_counts[pre_indices] + epsilon)
                post_weights = 1 / (incoming_counts[post_indices] + epsilon)

                # Combined probability
                probabilities = pre_weights * post_weights
                probabilities /= np.sum(probabilities)  # Normalize

                new_indices = np.random.choice(
                    inactive_indices, len(to_remove_indices), replace=False, p=probabilities
                )
                # Prune + generate
                synapse.syn_status[to_remove_indices] = 0
                synapse.syn_status[new_indices] = 1
                if k == 0:
                    synapse.w[new_indices] = np.random.uniform(low=0.1, high=0.2, size=1)[0] 
                    synapse.w[new_indices] = np.random.uniform(low=0.1, high=0.2, size=1)[0] 
   

    # Monitors
    allMPe = b2.SpikeMonitor(Pe)
    allMPi = b2.SpikeMonitor(Pi)
    MPe = b2.SpikeMonitor(Pe)
    MPi = b2.SpikeMonitor(Pi)


    # Create a Brian2 Network object and add components
    network = b2.Network(b2.collect())
    network.add(structural_plasticity)
    # Run the simulation, total time 10 min 

    

    #npy_dir = save_dir / "npy"
    #npy_dir.mkdir(exist_ok=True)
    
    valid = True
    block_num = 0
    ee_pruned_trace = []
    ie_pruned_trace = []
    big_weights_ee = []
    big_weights_ie = []
    pop_rate_e = []
    pop_rate_i = []
    
    # First Minute of Simulation Time
    valid = True
    from tqdm import tqdm 
    for tt in tqdm(range(120)):
        
        pruned = [0, 0]
        duration = 0.5 * second 

        network.run(duration)
       
    
        active_ee = np.array(con_ee.syn_status)
        ids = np.argwhere(active_ee == 1)[:,0]
        wee = np.array(con_ee.w)[active_ee == 1]
        big_ee = np.where(wee > 0.9*gmax, 1, 0).sum() / len(wee)
        
        active_ie = np.array(con_ie.syn_status)
        ids = np.argwhere(active_ie == 1)[:,0]
        wie = np.array(con_ie.w)[active_ie == 1]
        big_ie = np.where(wie > 0.9*gmax, 1, 0).sum() / len(wie)
        big_weights_ee.append(big_ee)
        big_weights_ie.append(big_ie)
        
        ee_total = np.sum(np.array(con_ee.syn_status))
        ie_total = np.sum(np.array(con_ie.syn_status))
        ee_prune_rate = pruned[0] / float(duration)
        ie_prune_rate = pruned[1] / float(duration)
        ee_pruned_trace.append(ee_prune_rate)
        ie_pruned_trace.append(ie_prune_rate)
        # Monitor firing rates  
        Pe_rate = len(MPe.t) / (len(Pe) * duration)
        Pi_rate = len(MPi.t) / (len(Pi) * duration)

        pop_rate_e.append(float(Pe_rate))
        pop_rate_i.append(float(Pi_rate))
        
        # We only termine the simulation here is firing rates go out of bound (no firing or too much firing)
        min_Pe_rate = 0.5 * Hz
        max_Pe_rate = 100 * Hz
        min_Pi_rate = 0.5 * Hz
        max_Pi_rate = 100 * Hz
        #if (
        #    not (min_Pe_rate <= Pe_rate <= max_Pe_rate)
        #    or not (min_Pi_rate <= Pi_rate <= max_Pi_rate)
        #    ):
        #    print(f'Rates out of bounds rate_e:{Pe_rate} rate_i:{Pi_rate}')
        #    valid = False
             
        
        network.remove(MPe)
        network.remove(MPi)
        MPe = b2.SpikeMonitor(Pe)
        MPi = b2.SpikeMonitor(Pi)
        network.add(MPe)
        network.add(MPi)
    # We disable this, so its not activated and we run the whole thing 
    if False:
        continue
    else:
        minute_one_json = {
            
            
            #'weights_ee' : all_active_ee,
            #'weights_ie' : all_active_ie,
            'pruning_rate_ee' : ee_pruned_trace,
            'pruning_rate_ie' : ie_pruned_trace,
            'big_weights_ee' : big_weights_ee,
            'big_weights_ie' : big_weights_ie,
            'rates_e' : pop_rate_e,
            'rates_i' : pop_rate_i,
            'all_timepoints' : [0.5 * (tt+1) for tt in range(120)]
        }
        # Log-simulation of 9 min with sparser saving intervals
        duration = 30 * second   
       
        
        ee_pruned_trace = []
        ie_pruned_trace = []
        big_weights_ee = []
        big_weights_ie = []
        pop_rate_e = []
        pop_rate_i = []
        all_active_ee = {}
        all_active_ie = {}
        for tt in tqdm(range(18)):
            pruned = [0, 0]
            network.run(duration)
            
            active_ee = np.array(con_ee.syn_status)
            ids = np.argwhere(active_ee == 1)[:,0]
            wee = np.array(con_ee.w)[active_ee == 1]
            big_ee = np.where(wee > 0.9*gmax, 1, 0).sum() / len(wee)
            
            active_ie = np.array(con_ie.syn_status)
            ids = np.argwhere(active_ie == 1)[:,0]
            wie = np.array(con_ie.w)[active_ie == 1]
            big_ie = np.where(wie > 0.9*gmax, 1, 0).sum() / len(wie)
            big_weights_ee.append(big_ee)
            big_weights_ie.append(big_ie)

            
            ee_total = np.sum(np.array(con_ee.syn_status))
            ie_total = np.sum(np.array(con_ie.syn_status))
            ee_prune_rate = pruned[0] / float(duration)
            ie_prune_rate = pruned[1] / float(duration)
            ee_pruned_trace.append(ee_prune_rate)
            ie_pruned_trace.append(ie_prune_rate)
        
            if tt in [0,17]:
 
                # We save the weights 2 times, after the inital 1 min and after 10 min
                active_ee = np.array(con_ee.syn_status)
                ids = np.argwhere(active_ee == 1)[:,0]
                wee = np.array(con_ee.w)[active_ee == 1]
                for id, ww in zip(ids, wee):
                        if not all_active_ee.get(str(int(id))):
                            all_active_ee[str(int(id))] = [[float(ww), tt]]
                        else: 
                            all_active_ee[str(int(id))].append([float(ww), tt])
                            
                active_ie = np.array(con_ie.syn_status)
                ids = np.argwhere(active_ie == 1)[:,0]
                wie = np.array(con_ie.w)[active_ie == 1]
                for id, ww in zip(ids, wie):
                        if not all_active_ie.get(str(int(id))):
                            all_active_ie[str(int(id))] = [[float(ww), tt]]
                        else: 
                            all_active_ie[str(int(id))].append([float(ww), tt])
                
            # Monitor firing rates  
            Pe_rate = len(MPe.t) / (len(Pe) * duration)
            Pi_rate = len(MPi.t) / (len(Pi) * duration)

            pop_rate_e.append(float(Pe_rate))
            pop_rate_i.append(float(Pi_rate))
                
            # We only termine the simulation here is firing rates go out of bound (no firing or too much firing)
            # we are not more strict with those
            min_Pe_rate = 5 * Hz
            max_Pe_rate = 50 * Hz
            min_Pi_rate = 5 * Hz
            max_Pi_rate = 50 * Hz
            if (
                    not (min_Pe_rate <= Pe_rate <= max_Pe_rate)
                    or not (min_Pi_rate <= Pi_rate <= max_Pi_rate)
                ):
                print(f'Rates out of bounds rate_e:{Pe_rate} rate_i:{Pi_rate}')
                #valid = False
                #break
            network.remove(MPe)
            network.remove(MPi)
            MPe = b2.SpikeMonitor(Pe)
            MPi = b2.SpikeMonitor(Pi)
            network.add(MPe)
            network.add(MPi)
        # We disable this again so we run the whole thing
        if False:
            continue

        else:
            
       
            long_run_json = {'weights_ee' : all_active_ee,
                            'weights_ie' : all_active_ie,
                            'pruning_rate_ee' : ee_pruned_trace,
                            'pruning_rate_ie' : ie_pruned_trace,
                            'big_weights_ee' : big_weights_ee,
                            'big_weights_ie' : big_weights_ie,
                            'rates_e' : pop_rate_e,
                            'rates_i' : pop_rate_i,
                            'all_timepoints' : [0.5 * (tt+1)+60 for tt in range(120)]
                            }
            spikes = {"exc": {'unit' : allMPe.i, 'times' :  allMPe.t}, "inh": {'unit' : allMPi.i, 'times' :  allMPi.t}}
            total_data = {'parameters' : parameters, 'minute_one' : minute_one_json, 'long_run' : long_run_json, 'spikes' : spikes }
            # We save the json file 
            #rd_dir1 = random_filename(length=2)
            #rd_dir2 = random_filename(length=1)
            path = Path(path)
            save_path = path.parent / (str(path.stem) + f"_{n}" + ".json")
            #save_path = Path(nas_path) / rd_dir1 / rd_dir2 
            #save_path.mkdir(parents=True, exist_ok=True)
            #save_path = save_path / (random_filename() + ".json")
            with open(str(save_path), 'w') as f:
                json.dump(total_data, f)