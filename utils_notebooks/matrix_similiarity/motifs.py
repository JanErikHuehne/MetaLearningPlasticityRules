"""
This file contains helper classes to 
extract triplet motifs from and E-I style 
network 
"""

import json 
from utils import standard_ee_wrapper


def random_matrix_generator(row_sums, col_sums, num=10):
    """
    This function generates multiple binary matrices
    satisfying the given row and column sums. 
    """
    import numpy as np 
    import pulp 
    n = len(row_sums)  # Matrix size
    solver = pulp.PULP_CBC_CMD(msg=False)  # Use CBC solver
    
    found_matrices = []
    while len(found_matrices) < num:
        # Define ILP problem
        prob = pulp.LpProblem("BinaryMatrixGeneration", pulp.LpMinimize)
        # Define binary decision variables
        X = [[pulp.LpVariable(f"X_{i}_{j}", 0, 1, pulp.LpBinary) for j in range(n)] for i in range(n)]
         # Row sum constraints
        for i in range(n):
            prob += (pulp.lpSum(X[i][j] for j in range(n)) == row_sums[i]), f"Row_{i}_sum"
        
        # Column sum constraints
        for j in range(n):
            prob += (pulp.lpSum(X[i][j] for i in range(n)) == col_sums[j]), f"Col_{j}_sum"
        # Diagonal constraints (forcing X[i][i] = 0)
        for i in range(n):
            prob += (X[i][i] == 0), f"Diagonal_{i}_zero"
        # Define a small random cost for each X[i][j]
        # This will enable different solutions for each run 
        random_objective = pulp.lpSum(np.random.uniform(0, 1) * X[i][j] for i in range(n) for j in range(n))
        prob += random_objective  # Encourage different solutions
        prob.solve(solver)
        
        
        # Check if a solution was found
        # if not return the found matrices 
        if pulp.LpStatus[prob.status] != "Optimal":
            print('Did not find a solution in this run')
            continue 
        # Extract the solution matrix
        new_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                new_matrix[i, j] = int(pulp.value(X[i][j]))
        new_solution = True
        for found in found_matrices:
            if np.array_equal(found, new_matrix):
                new_solution = False
        if new_solution:
            found_matrices.append(new_matrix)
    return found_matrices
def eee_triplets(trial_json, gpu_acc=True):
    if gpu_acc: 
        try:
            import cupy as np
        except Exception as e: 
            print('---- motifs - eee_triplets() --- function called with gpu_acc activated but unable to load gpu library cupy')
    else:
        import numpy as np 

    connectivity_matrix = np.array(standard_ee_wrapper(trial_json))
    print(connectivity_matrix.shape)
    print(type(connectivity_matrix))
    

#with open("/nas/ge84yes/projects/2024OptimizeSTDP/2025_SUMMER_RUNS/wf/v/jjhcxxvfugif_0.json", 'r') as f:
#    dic =json.load(f)

    
#eee_triplets(dic)
#matrix = standard_ee_wrapper(dic)
#print(matrix.shape)
#matrix = (matrix > 0).astype(int)
#print(matrix.shape)
#print(len(random_matrix_generator(matrix.sum(axis=1), matrix.sum(axis=0), )))
#print(np.mean(ie_matrix.sum(axis=1)), np.std(ie_matrix.sum(axis=1)))
#print(np.mean(ie_matrix.sum(axis=0)), np.std(ie_matrix.sum(axis=0)))

#ee_matrix  =standard_ee_wrapper(dic)
#print(np.mean(ee_matrix.sum(axis=1)), np.std(ee_matrix.sum(axis=1)))
#print(np.mean(ee_matrix.sum(axis=0)), np.std(ee_matrix.sum(axis=0)))