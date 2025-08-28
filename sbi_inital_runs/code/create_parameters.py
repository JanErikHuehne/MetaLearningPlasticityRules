"""
Here we create a small batch of simulations to run with the simulation function 
"""
import json
import numpy as np 
from math import ceil 

class CPrior():
    def __init__(self, low, high):
        self.low = low
        self.high = high 
        
    def sample(self, n):
        return np.random.uniform(self.low, self.high, (n,))
    
class DPrior():
    def __init__(self, intervals):
        self.num_intervals = len(intervals)
        self.intervals = intervals 
        
    def sample(self, n):
        inter = np.random.choice(a=list(range(self.num_intervals)), size=(n))
        samples = []
        for int_ in inter:
            
            samples.append(np.random.uniform(low=self.intervals[int_][0], high=self.intervals[int_][1], size=(1,))[0] )
        return np.array(samples)
    

def sample_parameters(classes=2, n_samples=30, batch_jobs=True, batch_size=30):
    class_names = ['ee', 'ie']
    prior_names = ['alpha_pre', 'alpha_post', 'Aplus', 'tauplus_stdp', 'tauminus_stdp', 'factor']
    priors = [CPrior(-1.0, 1.0), CPrior(-1.0, 1.0), CPrior(0.2, 5.0), CPrior(0.005, 0.030),CPrior(0.005, 0.030), DPrior([[-2,-0.5], [0.5, 2]]) ]
  
    
    all_batches = []
    total_batches = ceil(n_samples / batch_size) if batch_jobs else 1
    for _ in range(total_batches):
        #current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)
        batch_params = []

        for _ in range(classes):
            for prior in priors:
                s = prior.sample(batch_size if batch_jobs else n_samples)  # shape: (current_batch_size,)
                batch_params.append(s)
        # shape: (current_batch_size, classes * len(priors))
        batch_array = np.vstack(batch_params).T
        all_batches.append(batch_array)
    all_batches =  np.array(all_batches) if batch_jobs else  np.vstack(all_batches)
    
    if batch_jobs:
        for k,batch in enumerate(all_batches):
            batch_file = f"{k}.json"
            batch_list = []
            for sample in batch:
                sample_dict = {}
                for i, c in enumerate(class_names):
                    for ii, p in enumerate(prior_names):
                        name = f"{c}_{p}"
                        sample_dict[name]  = sample[len(prior_names) * i + ii]
                batch_list.append(sample_dict)
            
            with open(batch_file, 'w') as f:
                json.dump(batch_list, f)
                        
        
        
    return all_batches
    
