import numpy as np 
from brian2.units import *
import brian2 as b2 
import json 



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
#    dic =json.load(f)

    
#ie_matrix  =standard_ie_wrapper(dic)
#print(np.mean(ie_matrix.sum(axis=1)), np.std(ie_matrix.sum(axis=1)))
#print(np.mean(ie_matrix.sum(axis=0)), np.std(ie_matrix.sum(axis=0)))

#ee_matrix  =standard_ee_wrapper(dic)
#print(np.mean(ee_matrix.sum(axis=1)), np.std(ee_matrix.sum(axis=1)))
#print(np.mean(ee_matrix.sum(axis=0)), np.std(ee_matrix.sum(axis=0)))
    