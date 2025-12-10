"""
Helper classes to extract triplet motifs from an E–I style network.
"""

import json
import itertools
import logging
import time
from utils import standard_ee_wrapper, standard_ei_wrapper,standard_ie_wrapper, standard_ii_wrapper


# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #

logger = logging.getLogger("motifs")
logger.setLevel(logging.DEBUG)

# default handler if user does not configure logging externally
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(levelname)s] %(name)s: %(message)s",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# --------------------------------------------------------------------------- #
# Random Matrix Generator
# --------------------------------------------------------------------------- #

def random_matrix_generator(row_sums, col_sums, num=10):
    """
    Generates multiple binary matrices satisfying specified row/column sums.
    Provides extensive diagnostic logging for each ILP solve.
    """
    import numpy as np
    import pulp

    g = logging.getLogger("motifs.random_matrix")

    n = len(row_sums)
    g.info(f"Starting random matrix generation: size={n}×{n}, target={num}")

    solver = pulp.PULP_CBC_CMD(msg=False)
    found = []

    attempt = 0
    t0 = time.time()

    while len(found) < num:
        attempt += 1
        g.debug(f"ILP attempt {attempt}: beginning solve")

        prob = pulp.LpProblem("BinaryMatrixGeneration", pulp.LpMinimize)
        X = [[pulp.LpVariable(f"X_{i}_{j}", 0, 1, pulp.LpBinary)
              for j in range(n)] for i in range(n)]

        # Row constraints
        for i in range(n):
            prob += (pulp.lpSum(X[i][j] for j in range(n)) == row_sums[i])
        # Column constraints
        for j in range(n):
            prob += (pulp.lpSum(X[i][j] for i in range(n)) == col_sums[j])
        # Diagonal zeros
        for i in range(n):
            prob += (X[i][i] == 0)

        # Random perturbation to diversify solutions
        random_obj = pulp.lpSum(np.random.uniform(0, 1) * X[i][j]
                                for i in range(n) for j in range(n))
        prob += random_obj

        # Solve ILP
        solve_start = time.time()
        prob.solve(solver)
        solve_time = time.time() - solve_start

        status = pulp.LpStatus[prob.status]
        g.debug(f"ILP attempt {attempt}: status={status}, solve_time={solve_time:.3f}s")

        if status != "Optimal":
            g.warning(f"Attempt {attempt}: no optimal solution")
            continue

        new_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                new_matrix[i, j] = int(pulp.value(X[i][j]))

        # Deduplication
        duplicate = any(np.array_equal(new_matrix, f) for f in found)
        if duplicate:
            g.debug(f"Attempt {attempt}: solution duplicated, discarding")
        else:
            g.info(f"Found new unique matrix ({len(found)+1}/{num})")
            found.append(new_matrix)

    g.info(f"Completed generation: {len(found)} matrices in {time.time() - t0:.2f}s")
    return found





def ei_triplets(ee_conn_matrix, ei_conn_matrix, ie_conn_matrix, ii_conn_matrix,
                 gpu_acc=True, batch_size=50000, savepath=None, triplet_type='eei'):
    """
    Extracts canonical triplet motifs with detailed logging of GPU status,
    batch progress, canonicalization steps, and classification timing.
    """
    g = logging.getLogger("motifs.ei_triplets")
    assert triplet_type in ['eei', 'iie'], "Error unrecognized triplet type"
    # --------------------------------------------------------------- #
    # Backend selection
    # --------------------------------------------------------------- #
    if gpu_acc:
        try:
            import cupy as np
            g.info("GPU acceleration enabled (CuPy backend)")
        except Exception:
            g.warning("CuPy unavailable; falling back to NumPy")
            import numpy as np
    else:
        import numpy as np
        g.info("Using npU (NumPy backend)")
    
    
    ee_conn = np.array(ee_conn_matrix)
    ee_conn = (ee_conn > 0).astype(int)
    N1 = ee_conn.shape[0]

    ie_conn = np.array(ie_conn_matrix)
    ie_conn = (ie_conn > 0).astype(int)
    N2 = ie_conn.shape[0]
    
    ei_conn = np.array(ei_conn_matrix)
    ei_conn = (ei_conn > 0).astype(int)
   
    ii_conn = np.array(ii_conn_matrix)
    ii_conn = (ii_conn > 0).astype(int) 
    g.info(f"Input connectivity: {N1} (E) {N2} (I) nodes, {batch_size} batch size, triplet type {triplet_type}")

    # --------------------------------------------------------------- #
    # Canonical form computation
    # --------------------------------------------------------------- #

    def get_canonical_form_batch(submat):
        B = submat.shape[0]
        weights = np.array([[1, 2, 4],
                            [8, 16, 32],
                            [64, 128, 256]])

        perm_indices = np.array([[1,0,2], [0,1,2]])  # Shape: (2, 2)

        g.debug(f"Canonicalizing batch of {B} matrices")

        matrices_expanded = np.repeat(submat[:, None, :, :], repeats=2, axis=1)  # Shape: (batch, 2, 3, 3)
        for i in range(2):  # Loop over 6 permutations
            matrices_expanded[:, i, :, :] = matrices_expanded[:, i, perm_indices[i], :][:, :, perm_indices[i]]
        perm_weights = np.sum(weights[None, None, :, :] * matrices_expanded, axis=(2, 3))  # Shape: (batch, 2)

        # Find the index of the minimum weight for each matrix in parallel
        min_idx = np.argmin(perm_weights, axis=1)  # Shape: (batch,)
        
        # Select the corresponding canonical form
        canonical_forms = matrices_expanded[np.arange(B), min_idx]  # Shape: (batch, 3, 3)
        return canonical_forms  # Returns all canonical forms in parallel
    
    def classify_triplet_batch(canonical_forms):
        canonical_motifs = np.stack([
            
             np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), # 0 connections

            np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]), # 1 connection
            np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
            
            
            np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]), # 2 connections
            np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
            np.array([[0, 0, 1], [1, 0, 0], [0, 0, 0]]), 
            np.array([[0, 1,0], [0, 0, 0], [1, 0, 0]]),
            np.array([[0, 0,0], [0, 0, 1], [1, 0, 0]]),
            np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]]),
            np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]]),
            np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]]), 
            np.array([[0, 0,0], [0, 0, 0], [1, 1, 0]]),
            
            np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0]]), # 3 connections
            np.array([[0, 0, 1], [1, 0, 0], [1, 0, 0]]), 
            np.array([[0, 1, 1], [1, 0, 0], [0, 0, 0]]), 
            np.array([[0, 1, 1], [0, 0, 0], [1, 0, 0]]), 
            np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]), 
            np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]), 
            np.array([[0, 0, 0], [1, 0, 1], [1, 0, 0]]),
            np.array([[0, 1, 0], [0, 0, 0], [1, 1, 0]]),
            np.array([[0, 0, 1], [0, 0, 0], [1, 1, 0]]),
            np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0]]),
            
            
            np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]), # 4 connections
            np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]]),
            np.array([[0, 1, 0], [1, 0, 1], [1, 0, 0]]), 
            np.array([[0, 0, 1], [1, 0, 0], [1, 1, 0]]),
            np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0]]), 
            np.array([[0, 1, 0], [1, 0, 0], [1, 1, 0]]), 
            np.array([[0, 0, 1], [1, 0, 1], [1, 0, 0]]),
            np.array([[0, 1, 1], [1, 0, 1], [0, 0, 0]]), 
            np.array([[0, 1, 1], [0, 0, 0], [1, 1, 0]]), 
            

            np.array([[0, 1, 1], [0, 0, 1], [1, 1, 0]]), # 5 connections
            np.array([[0, 1, 1], [1, 0, 0], [1, 1, 0]]), 
            np.array([[0, 1, 1], [1, 0, 1], [1, 0, 0]]),
            
            np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) # 6 connections
        ])  # Shape: (num_motifs, 3, 3)
        # Compare all canonical forms against all motifs in parallel
        matches = (canonical_forms[:, None, :, :] == canonical_motifs[None, :, :, :]).all(axis=(2, 3))  # Shape: (batch, num_motifs)
               # Compare against catalogue
        matches = (canonical_forms[:, None, :, :] ==
                   canonical_motifs[None, :, :, :]).all(axis=(2, 3))

        # Identify rows with no motif match
        has_match = matches.any(axis=1)

        if not has_match.all():
            # Extract failing canonical forms
            bad_idx = np.where(~has_match)[0]

            # If np == cupy, move to host for printing
            try:
                bad_idx_cpu = bad_idx.get()
            except AttributeError:
                bad_idx_cpu = bad_idx

            for idx in bad_idx_cpu:
                # Fetch the offending 3×3 matrix
                M = canonical_forms[idx]
                try:
                    M_cpu = M.get()     # cupy → numpy
                except AttributeError:
                    M_cpu = M

                print("Unmatched canonical motif at index", idx)
                print(M_cpu)

            
            raise RuntimeError("Some canonical forms did not match any motif.")

        # Otherwise normal path
        motif_indices = matches.argmax(axis=1)

        # Return directly; this is a NumPy array if np=numpy, or CuPy array if np=cupy
        return motif_indices
        
        return np.array(result)

    def parallel_triplets(conn_flex_flex, conn_flex_fixed, conn_fixed_flex):
        n1 = conn_flex_flex.shape[0]
        n2 = conn_flex_fixed.shape[1]   # or conn_fixed_flex.shape[0]
        flex = np.arange(n1)
        fixed = np.arange(n2)
        
        # unordered flex pairs
        pairs = np.array(list(itertools.combinations(flex.tolist(), 2)))   # (P, 2)
        P = pairs.shape[0]
        # combine with fixed units
        triplets = np.concatenate([
            np.repeat(pairs, n2, axis=0),         # (P*n2, 2)
            np.tile(fixed, (P, 1)).reshape(-1, 1) # (P*n2, 1)
        ], axis=1)
        triplet_indices = triplets[:, :, None]  # Reshape to (num_triplets, 3, 1)
        T = triplet_indices.shape[0]

        g.info(f"Enumerating {T} triplets")
        
        results = []
        t_start = time.time()

        for k, start in enumerate(range(0, T, batch_size)):
            end = min(start + batch_size, T)
            batch = triplets[start:end]
            B = batch.shape[0]
           

            i = batch[:, 0]
            j = batch[:, 1]
            kf = batch[:, 2]               # fixed index
            
            g.debug(f"Processing batch {k+1}: size={B}, triplets {start}–{end}")

            # Build the 3×3 connectivity matrices for each triplet
            subm = np.zeros((B, 3, 3), dtype=conn_flex_flex.dtype)

            # --- flex–flex block ---
            subm[:, 0, 0] = conn_flex_flex[i, i]
            subm[:, 0, 1] = conn_flex_flex[i, j]
            subm[:, 1, 0] = conn_flex_flex[j, i]
            subm[:, 1, 1] = conn_flex_flex[j, j]

            # --- flex–fixed block ---
            subm[:, 0, 2] = conn_flex_fixed[i, kf]
            subm[:, 1, 2] = conn_flex_fixed[j, kf]

            # --- fixed–flex block ---
            subm[:, 2, 0] = conn_fixed_flex[kf, i]
            subm[:, 2, 1] = conn_fixed_flex[kf, j]

            # --- fixed–fixed ---
            #  diagonal 0
            subm[:, 2, 2] = 0
            canon = get_canonical_form_batch(subm)
            
            results.append(canon)
            if gpu_acc:
                np.get_default_memory_pool().free_all_blocks()
                g.debug("GPU memory pool cleared")
        g.info(f"Finished canonicalization in {time.time() - t_start:.2f}s")
        return np.concatenate(results, axis=0)
    
    if triplet_type == 'eei':
        canonical_forms = parallel_triplets(ee_conn, ei_conn, ie_conn)
    elif triplet_type == 'iie':
        canonical_forms = parallel_triplets(ii_conn, ie_conn, ei_conn)
    g.info("Beginning classification")

    t_class = time.time()
    motifs = classify_triplet_batch(canonical_forms)
    g.info(f"Classification complete in {time.time() - t_class:.2f}s")

    motif_counts = np.bincount(motifs, minlength=36)
    g.info(f"Motif histogram: {motif_counts.tolist()}")

    #if savepath:
    #    np.save(savepath, motif_counts)
    #    g.info(f"Saved motifs to {savepath}")

    #return motif_counts

    
        
   
def eee_triplets(connectivity_matrix, gpu_acc=True, batch_size=50000, savepath=None):
    """
    Extracts canonical triplet motifs with detailed logging of GPU status,
    batch progress, canonicalization steps, and classification timing.
    """

    g = logging.getLogger("motifs.eee_triplets")

    # --------------------------------------------------------------- #
    # Backend selection
    # --------------------------------------------------------------- #
    if gpu_acc:
        try:
            import cupy as np
            g.info("GPU acceleration enabled (CuPy backend)")
        except Exception:
            g.warning("CuPy unavailable; falling back to NumPy")
            import numpy as np
    else:
        import numpy as np
        g.info("Using npU (NumPy backend)")

    conn = np.array(connectivity_matrix)
    conn = (conn > 0).astype(int)
    N = conn.shape[0]

    g.info(f"Input connectivity: {N} nodes, {batch_size} batch size")

    # --------------------------------------------------------------- #
    # Canonical form computation
    # --------------------------------------------------------------- #

    def get_canonical_form_batch(submat):
        B = submat.shape[0]
        weights = np.array([[1, 2, 4],
                            [8, 16, 32],
                            [64, 128, 256]])

        perm_indices = np.array(list(itertools.permutations([0, 1, 2])))

        g.debug(f"Canonicalizing batch of {B} matrices")

        matrices_expanded = np.repeat(submat[:, None, :, :], repeats=6, axis=1)

        # Apply permutations
        for p_idx in range(6):
            matrices_expanded[:, p_idx] = matrices_expanded[:, p_idx,
                                                            perm_indices[p_idx]][:,
                                                            :, perm_indices[p_idx]]

        scores = np.sum(weights[None, None, :, :] * matrices_expanded, axis=(2, 3))
        min_idx = np.argmin(scores, axis=1)

        return matrices_expanded[np.arange(B), min_idx]

    # --------------------------------------------------------------- #
    # Motif classification
    # --------------------------------------------------------------- #

    def classify_triplet_batch(canonical_forms):
        g.debug(f"Classifying {canonical_forms.shape[0]} canonical matrices")

        canonical_motifs = np.stack([
            np.array([[0,0,0],[0,0,0],[0,0,0]]),
            np.array([[0,1,0],[0,0,0],[0,0,0]]),
            np.array([[0,1,0],[1,0,0],[0,0,0]]),
            np.array([[0,0,1],[1,0,0],[0,0,0]]),
            np.array([[0,0,1],[0,0,1],[0,0,0]]),
            np.array([[0,1,1],[0,0,0],[0,0,0]]),
            np.array([[0,1,0],[1,0,0],[1,0,0]]),
            np.array([[0,1,1],[1,0,0],[0,0,0]]),
            np.array([[0,1,0],[0,0,1],[1,0,0]]),
            np.array([[0,1,1],[0,0,1],[0,0,0]]),
            np.array([[0,1,1],[1,0,0],[1,0,0]]),
            np.array([[0,1,1],[0,0,1],[1,0,0]]),
            np.array([[0,1,1],[1,0,1],[0,0,0]]),
            np.array([[0,0,1],[1,0,1],[1,0,0]]),
            np.array([[0,1,1],[1,0,1],[1,0,0]]),
            np.array([[0,1,1],[1,0,1],[1,1,0]])
        ])

        matches = (canonical_forms[:, None, :, :] ==
                   canonical_motifs[None, :, :, :]).all(axis=(2, 3))

        return np.argmax(matches, axis=1)

    # --------------------------------------------------------------- #
    # Batch extraction
    # --------------------------------------------------------------- #

    def parallel_triplets(conn_matrix):
        nodes = np.arange(conn_matrix.shape[0])
        triples = np.array(list(itertools.combinations(nodes.tolist(), 3)))
        T = triples.shape[0]

        g.info(f"Enumerating {T} triplets")

        results = []
        t_start = time.time()

        for k, start in enumerate(range(0, T, batch_size)):
            end = min(start + batch_size, T)
            batch = triples[start:end]
            B = batch.shape[0]

            g.debug(f"Processing batch {k+1}: size={B}, triplets {start}–{end}")

            idx = batch[:, :, None]
            subm = conn_matrix[idx, idx.transpose(0, 2, 1)]

            canon = get_canonical_form_batch(subm)
            
            results.append(canon)

            if gpu_acc:
                np.get_default_memory_pool().free_all_blocks()
                g.debug("GPU memory pool cleared")

        g.info(f"Finished canonicalization in {time.time() - t_start:.2f}s")
        return np.concatenate(results, axis=0)

    canonical_forms = parallel_triplets(conn)
   
    g.info("Beginning classification")

    t_class = time.time()
    motifs = classify_triplet_batch(canonical_forms)
    g.info(f"Classification complete in {time.time() - t_class:.2f}s")

    motif_counts = np.bincount(motifs, minlength=16)
    g.info(f"Motif histogram: {motif_counts.tolist()}")

    if savepath:
        np.save(savepath, motif_counts)
        g.info(f"Saved motifs to {savepath}")

    return motif_counts


#with open("/nas/ge84yes/projects/2024OptimizeSTDP/2025_SUMMER_RUNS/wf/v/jjhcxxvfugif_0.json", 'r') as f:
#    dic =json.load(f)
#
#    
#eee_triplets(dic)
#import numpy as np 
#ee_matrix = np.random.random((800,800)).astype(float)
#ee_matrix[ee_matrix < 0.7] = 0
#for i in range(800):
#    ee_matrix[i,i] = 0
#ei_matrix = np.random.random((800,200)).astype(float)
#ei_matrix[ei_matrix < 0.7] = 0
#ie_matrix = np.random.random((200,800)).astype(float)
#ie_matrix[ie_matrix < 0.7] = 0
#ii_matrix = np.random.random((200,200)).astype(float)
#ii_matrix[ii_matrix < 0.7] = 0
#for i in range(200):
#    ii_matrix[i,i] =0
#ee_matrix = standard_ee_wrapper(dic)
#ei_matrix = standard_ei_wrapper(dic)
#ie_matrix = standard_ie_wrapper(dic)
#ii_matrix = standard_ii_wrapper(dic)

#ei_triplets(ee_matrix, ei_matrix, ie_matrix, ii_matrix, batch_size=10000, triplet_type='iie')
#print(matrix.shape)
#eee_triplets(matrix)
#matrix = (matrix > 0).astype(int)
#print(matrix.shape)
#print(len(random_matrix_generator(matrix.sum(axis=1), matrix.sum(axis=0), )))
#print(np.mean(ie_matrix.sum(axis=1)), np.std(ie_matrix.sum(axis=1)))
#print(np.mean(ie_matrix.sum(axis=0)), np.std(ie_matrix.sum(axis=0)))

#ee_matrix  =standard_ee_wrapper(dic)
#print(np.mean(ee_matrix.sum(axis=1)), np.std(ee_matrix.sum(axis=1)))
#print(np.mean(ee_matrix.sum(axis=0)), np.std(ee_matrix.sum(axis=0)))