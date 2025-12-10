import numpy as np 
from brian2.units import *
import brian2 as b2 
import json 



b2.BrianLogger.log_level_error()

def standard_ei_wrapper(weight_dict):
    seed = weight_dict.get('seed') if  weight_dict.get('seed') else 42
    b2.start_scope()
    b2.seed(seed)
    
    """
    We define a mock model 
    """
    NE = 800
    NI = NE // 4
    
    neurons = b2.NeuronGroup(NE + NI, model='')
    # Neuron model
    Pe = neurons[:NE]
    Pi = neurons[NE:]
    con_ei = b2.Synapses(Pe, Pi, model='w : 1')
    con_ei.connect(p=0.1, condition='i != j')
   
    #con_ei.w = np.random.uniform(low=1.0,high=5.0, size=len(con_ei.w))
    matrix = np.zeros((NE, NI))
    iis = np.array(con_ei.i)
    jjs = np.array(con_ei.j)

    matrix[iis, jjs] = 1
    
    return matrix

def standard_ii_wrapper(weight_dict):
    seed = weight_dict.get('seed') if  weight_dict.get('seed') else 42
    b2.start_scope()
    b2.seed(seed)
    
    """
    We define a mock model 
    """
    NE = 800
    NI = NE // 4
    
    neurons = b2.NeuronGroup(NE + NI, model='')
    # Neuron model
    Pe = neurons[:NE]
    Pi = neurons[NE:]
    con_ii = b2.Synapses(Pi, Pi, model='w : 1')
    con_ii.connect(p=0.1, condition='i != j')
   
    #con_ei.w = np.random.uniform(low=1.0,high=5.0, size=len(con_ei.w))
    matrix = np.zeros((NI, NI))
    iis = np.array(con_ii.i)
    jjs = np.array(con_ii.j)

    matrix[iis, jjs] = 1
    
    return matrix

def standard_ei_wrapper(weight_dict):
    seed = weight_dict.get('seed') if weight_dict.get('seed') else 42 
    b2.start_scope()
    b2.seed(seed)
    
    """
    We define a mock model 
    """
    NE = 800
    NI = NE // 4
    
    neurons = b2.NeuronGroup(NE + NI, model='')
    # Neuron model
    Pe = neurons[:NE]
    Pi = neurons[NE:]
    con_ei = b2.Synapses(Pe, Pi, model='w : 1')
    con_ii = b2.Synapses(Pi, Pi, model='w : 1')
    con_ei.connect(p=0.1, condition='i != j')
   
    #con_ei.w = np.random.uniform(low=1.0,high=5.0, size=len(con_ei.w))
    matrix = np.zeros((NE, NI))
    iis = np.array(con_ei.i)
    jjs = np.array(con_ei.j)

    matrix[iis, jjs] = 1
    
    return matrix



def standard_ee_wrapper(weight_dict):
    dic = weight_dict['long_run']['weights_ee']
    return extract_weight_matrix(weight_dict=dic, shape=(800,800))

def standard_ie_wrapper(weight_dict):
    dic = weight_dict['long_run']['weights_ie']
    return extract_weight_matrix(weight_dict=dic, shape=(200,800), self_connected=False)
    
def extract_weight_matrix(weight_dict, shape=(800,800),  self_connected=True):
    """
    
    """
    # We build a mock network in brian2
    
    n1 = b2.NeuronGroup(shape[0], model='')
    if self_connected:
        con = b2.Synapses(n1,n1) 
        con.connect(condition="i != j")
    else:
        n2 = b2.NeuronGroup(shape[1], model='')
        con = b2.Synapses(n1,n2) 
        con.connect(condition="i != j")
    
    # We initalize the weight matrix with 0s
    matrix = np.zeros(shape)
    # We extract the is and js from the mock matrix
    iis = np.array(con.i)
    jjs = np.array(con.j)
    # We now iterate over the weight_dict searching for entries at the 
    # last time point, if they exist we backtrack the i and j index and set
    # the weight accordingly 
    for key, value in weight_dict.items():
        for v in list(value): 
            # only if this is the last timepoint we will ad it to the matrix
            if v[-1] == 17:
                matrix[iis[int(key)], jjs[int(key)]] = v[0]
    print(np.sum(matrix > 0))
    # We return the connectivity matrix
    return matrix 
  
    
   


#with open("/nas/ge84yes/projects/2024OptimizeSTDP/2025_SUMMER_RUNS/wf/v/jjhcxxvfugif_0.json", 'r') as f:
#   dic =json.load(f)

    
#ei_matrix  =standard_ei_wrapper(dic)
#print(ei_matrix.shape)
#ii_matrix  =standard_ii_wrapper(dic)
#print(ii_matrix.shape)
#print(np.mean(ie_matrix.sum(axis=1)), np.std(ie_matrix.sum(axis=1)))
#print(np.mean(ie_matrix.sum(axis=0)), np.std(ie_matrix.sum(axis=0)))

#ee_matrix  =standard_ee_wrapper(dic)
#print(np.mean(ee_matrix.sum(axis=1)), np.std(ee_matrix.sum(axis=1)))
#print(np.mean(ee_matrix.sum(axis=0)), np.std(ee_matrix.sum(axis=0)))
    