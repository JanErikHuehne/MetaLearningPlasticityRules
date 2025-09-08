import numpy as np 
import brian2 as b2 
from brian2.units import * 
import pulp 
import cupy as cp
from tqdm import tqdm 
import itertools
def mock_network():
    
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




    # Synapse model with an `active` parameter±
    synapse_model = """
            w : 1  # Weight
            syn_status : integer  # Active or inactive status (1 or 0)
        """

    # Define EE synapses with sparsity and `active` parameter
    con_ee = b2.Synapses(
        Pe,
        Pe,
        model=synapse_model,
    )

    con_ee.connect(condition="i != j")  # Fully connect, except self-connections
    con_ee.w = np.random.uniform(low=0.1, high=0.2, size=len(con_ee.w)) 
    con_ee.syn_status = np.random.choice(
        [0.0, 1.0], size=len(con_ee.w), p=[1 - epsilon, epsilon]
    )
    synapse_model = """
            w : 1  # Weight
            syn_status : integer  # Active or inactive status (1 or 0)
                    """
    con_ie = b2.Synapses(
        Pi,
        Pe,
        model=synapse_model,
    )
    con_ie.connect(condition="i != j")  # Fully connect, except self-connections
    con_ie.w = np.random.uniform(low=0.1, high=0.2, size=len(con_ie.w)) 
    con_ie.syn_status = np.random.choice(
        [0.0, 1.0], size=len(con_ie.w), p=[1 - epsilon, epsilon]
    )
    
    return con_ee, con_ie


def generate_multiple_matrices_with_diagonal_constraint(row_sums, col_sums, num_matrices=5):
    """
    Generates multiple binary matrices satisfying given row and column sums,
    while ensuring the diagonal entries remain zero.
    """
    n = len(row_sums)  # Matrix size
    solver = pulp.PULP_CBC_CMD(msg=False)  # Use CBC solver

    found_matrices = []
    
    while len(found_matrices) < num_matrices:
        print(f'Finding new alternative matrix {len(found_matrices)}/{ num_matrices}')
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

        # Exclude previously found solutions (ensuring unique names)
        """
        for idx, matrix in enumerate(found_matrices):
            prob += (pulp.lpSum(X[i][j] for i in range(n) for j in range(n) if matrix[i][j] == 1) 
                     <= np.sum(matrix) - 1), f"Exclude_Solution_{idx}"
        """ 
        # Define a small random cost for each X[i][j]
        random_objective = pulp.lpSum(np.random.uniform(0, 1) * X[i][j] for i in range(n) for j in range(n))
        prob += random_objective  # Encourage different solutions
        # Solve the problem
        prob.solve(solver)

        # Check if a solution was found
        if pulp.LpStatus[prob.status] != "Optimal":
            print(f"Only {len(found_matrices)} solutions found.")
            break

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



def get_canonical_form_batch(matrices):
    """Compute canonical forms for all triplets in parallel on GPU."""
    
    weights = cp.array([[1, 2, 4], [8, 16, 32], [64, 128, 256]])  # GPU-based weight matrix

    # Generate all 3! permutations (GPU-accelerated)
    perm_indices = cp.array(list(itertools.permutations([0, 1, 2])))  # Shape: (6, 3)

    batch_size = matrices.shape[0]  # Number of triplets being processed in parallel

    # Step 1: Expand matrices to match 6 permutations per batch
    matrices_expanded = cp.repeat(matrices[:, None, :, :], repeats=6, axis=1)  # Shape: (batch, 6, 3, 3)
    for i in range(6):  # Loop over 6 permutations
        matrices_expanded[:, i, :, :] = matrices_expanded[:, i, perm_indices[i], :][:, :, perm_indices[i]]
    # Step 3: Compute lexicographic order weight for all permutations in parallel
    perm_weights = cp.sum(weights[None, None, :, :] * matrices_expanded, axis=(2, 3))  # Shape: (batch, 6)
    # Step 4: Find the index of the minimum weight for each matrix in parallel
    min_idx = cp.argmin(perm_weights, axis=1)  # Shape: (batch,)

    # Step 5: Select the corresponding canonical form
    canonical_forms = matrices_expanded[cp.arange(batch_size), min_idx]  # Shape: (batch, 3, 3)
    return canonical_forms  # Returns all canonical forms in parallel


def classify_triplet_batch(canonical_forms):
    canonical_motifs = cp.stack([
        cp.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        cp.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
        cp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        cp.array([[0, 0, 1], [1, 0, 0], [0, 0, 0]]),
        cp.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]]),
        cp.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]]),
        cp.array([[0, 1, 0], [1, 0, 0], [1, 0, 0]]),
        cp.array([[0, 1, 1], [1, 0, 0], [0, 0, 0]]),
        cp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        cp.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]),
        cp.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        cp.array([[0, 1, 1], [0, 0, 1], [1, 0, 0]]),
        cp.array([[0, 1, 1], [1, 0, 1], [0, 0, 0]]),
        cp.array([[0, 0, 1], [1, 0, 1], [1, 0, 0]]),
        cp.array([[0, 1, 1], [1, 0, 1], [1, 0, 0]]),
        cp.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    ])  # Shape: (num_motifs, 3, 3)
    # Compare all canonical forms against all motifs in parallel
    matches = (canonical_forms[:, None, :, :] == canonical_motifs[None, :, :, :]).all(axis=(2, 3))  # Shape: (batch, num_motifs)
    # Find the index of the first matching motif
    motif_indices = cp.argmax(matches, axis=1)  # Returns motif index for each matrix
    return motif_indices  # Return motif indices in parallel

   
def parallel_triplet_processing(connectivity_matrix, batch_size=1000000):
    nodes = cp.arange(connectivity_matrix.shape[0])
    all_triplets = list(itertools.combinations(nodes.tolist(), 3))

    results = []
    for i in range(0, len(all_triplets), batch_size):
        batch = cp.array(all_triplets[i:i+batch_size])
        triplet_indices = batch[:, :, None]
        submatrices = connectivity_matrix[triplet_indices, triplet_indices.transpose(0, 2, 1)]
        canonical_forms = get_canonical_form_batch(submatrices)
        motif_indices = classify_triplet_batch(canonical_forms)
        results.append(motif_indices)

    return cp.concatenate(results).tolist()

def get_triplets(np_matrix):
    
    connectivity_matrix = cp.array(np_matrix, dtype=cp.int8)
    print(f"Loaded connectivity matrix with shaoe {connectivity_matrix.shape}")
    # Compute row and column sums (CuPy operations)
    sum_rows = cp.sum(connectivity_matrix, axis=1)
    sum_cols = cp.sum(connectivity_matrix, axis=0)
    
    
    alternative_matrices = cp.array(generate_multiple_matrices_with_diagonal_constraint(
        cp.asnumpy(sum_rows), cp.asnumpy(sum_cols), num_matrices=50
    ))  # Convert NumPy output to CuPy
    print(f"Generated alternative matrices")
    print(f"Computing triplets in connectivity matrix ...")
    run_counts = cp.bincount(cp.array(parallel_triplet_processing(connectivity_matrix)), minlength=16)
    print(f"Done")
    print(f"Computing triplets in alternative matrices ...")
    all_counts = []
    for equivalent_matrix in tqdm(alternative_matrices):
            counts = cp.bincount(cp.array(parallel_triplet_processing(equivalent_matrix)), minlength=16)
            all_counts.append(cp.asarray(counts))  # Ensure each count array is a CuPy array

    print(f"Done")
    all_counts = cp.vstack(all_counts)  # Stack them properly into a single CuPy array
    z_mean = cp.mean(all_counts, axis=0)
    z_std = cp.std(all_counts, axis=0)
    
    # Compute Z-Scores (fully on GPU)
    bar_values = (run_counts - z_mean) / z_std

    # Convert to NumPy for Matplotlib
    z_scores = cp.asnumpy(bar_values)
    
    return z_mean, z_std, z_scores