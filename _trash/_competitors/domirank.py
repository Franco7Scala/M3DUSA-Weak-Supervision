########## Here are the associated DomiRank functions #############
import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import scipy.sparse.csgraph
import networkx as nx
import multiprocessing as mp


########## Here are the general functions needed for efficient dismantling and testing of networks #############

def get_largest_component(G, strong=False):
    '''
    Gets the largest component of a graph, either from scipy.sparse or from networkX.Graph datatype.
    '''
    if isinstance(G, nx.Graph):  # Check using isinstance for better compatibility
        if nx.is_directed(G) and not strong:
            GMask = max(nx.weakly_connected_components(G), key=len)
        elif nx.is_directed(G) and strong:
            GMask = max(nx.strongly_connected_components(G), key=len)
        else:
            GMask = max(nx.connected_components(G), key=len)
        G = G.subgraph(GMask)
    else:
        raise TypeError('You must input a networkx.Graph Data-Type')
    return G


def relabel_nodes(G, yield_map=False):
    '''Relabels the nodes to be from 0, ... len(G).'''
    if yield_map:
        nodes = dict(zip(range(len(G)), G.nodes()))
        G = nx.relabel_nodes(G, dict(zip(G.nodes(), range(len(G)))))
        return G, nodes
    else:
        G = nx.relabel_nodes(G, dict(zip(G.nodes(), range(len(G)))))
        return G


def get_component_size(G, strong=False):
    '''
    Gets the size of the largest component.
    Handles both NetworkX graphs and SciPy sparse matrices.
    '''
    if isinstance(G, nx.Graph):
        if nx.is_directed(G) and not strong:
            GMask = max(nx.weakly_connected_components(G), key=len)
        elif nx.is_directed(G) and strong:
            GMask = max(nx.strongly_connected_components(G), key=len)
        else:
            GMask = max(nx.connected_components(G), key=len)
        return len(GMask)
    elif sp.sparse.issparse(G):  # FIXED: Use issparse instead of internal _arrays check
        if not strong:
            connection_type = 'weak'
        else:
            connection_type = 'strong'
        # Note: connected_components returns (n_components, labels)
        noComponent, lenComponent = sp.sparse.csgraph.connected_components(G, directed=True, connection=connection_type, return_labels=True)
        return np.bincount(lenComponent).max()
    else:
        raise TypeError(f'Input must be networkx.Graph or scipy.sparse matrix. Got {type(G)}')


def get_link_size(G):
    if isinstance(G, nx.Graph):
        links = len(G.edges())
    elif sp.sparse.issparse(G):  # FIXED: Use issparse
        links = G.sum()
    else:
        raise TypeError('You must input a networkx.Graph Data-Type or Sparse Matrix')
    return links


def remove_node(G, removedNode):
    '''
    Removes the node from the graph.
    For sparse matrices, it zeros out the rows/cols (preserving shape to maintain indexing).
    '''
    if isinstance(G, nx.Graph):
        if isinstance(removedNode, int):
            G.remove_node(removedNode)
        else:
            G.remove_nodes_from(removedNode)  # More efficient networkx method
        return G
    elif sp.sparse.issparse(G):  # FIXED: Use issparse
        # Create a diagonal matrix with 1s everywhere except the removed nodes
        # We need to construct the diagonal manually to be efficient
        n = G.shape[0]
        # Start with all ones
        diag_data = np.ones(n)
        # Set removed indices to 0
        diag_data[removedNode] = 0

        # Create a dia_matrix or csr_matrix directly
        diag = sp.sparse.diags(diag_data, format='csr')

        # Apply mask: Row zeroing (diag @ G) and Column zeroing (Result @ diag)
        G = diag @ G
        return G @ diag


def generate_attack(centrality, node_map=False):
    '''Generates an attack strategy based on centrality values.'''
    if node_map is False:
        node_map = range(len(centrality))
    else:
        node_map = list(node_map.values())
    zipped = dict(zip(node_map, centrality))
    attackStrategy = sorted(zipped, reverse=True, key=zipped.get)
    return attackStrategy


def network_attack_sampled(G, attackStrategy, sampling=0):
    '''Attack a network in a sampled manner.'''
    if isinstance(G, nx.Graph):
        GAdj = nx.to_scipy_sparse_array(G)
    else:
        GAdj = G.copy()

    if sampling == 0:
        sampling = int(max(1, GAdj.shape[0] / 100))  # Ensure sampling is at least 1

    N = GAdj.shape[0]
    initialComponent = get_component_size(GAdj)
    initialLinks = get_link_size(GAdj)

    # Pre-calculate size to avoid errors if logic slightly misses
    num_steps = int(N / sampling)
    componentEvolution = np.zeros(num_steps)
    linksEvolution = np.zeros(num_steps)

    j = 0
    for i in range(N - 1):
        if i % sampling == 0 and j < num_steps:
            if i == 0:
                componentEvolution[j] = get_component_size(GAdj) / initialComponent
                linksEvolution[j] = get_link_size(GAdj) / initialLinks
                j += 1
            else:
                # Remove nodes in batch based on sampling size
                nodes_to_remove = attackStrategy[i - sampling:i]
                GAdj = remove_node(GAdj, nodes_to_remove)
                componentEvolution[j] = get_component_size(GAdj) / initialComponent
                linksEvolution[j] = get_link_size(GAdj) / initialLinks
                j += 1
    return componentEvolution, linksEvolution


######## Beginning of domirank stuff! ####################

def domirank(G, analytical=True, sigma=-1, dt=0.1, epsilon=1e-5, maxIter=1000, checkStep=10):
    '''
    Calculates DomiRank Centrality.
    '''
    if isinstance(G, nx.Graph):
        G = nx.to_scipy_sparse_array(G)
    else:
        G = G.copy()

    if analytical == False:
        if sigma == -1:
            sigma, _ = optimal_sigma(G, analytical=False, dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)

        pGAdj = sigma * G.astype(np.float32)
        Psi = np.ones(pGAdj.shape[0], dtype=np.float32) / pGAdj.shape[0]
        maxVals = np.zeros(int(maxIter / checkStep), dtype=np.float32)
        dt = np.float32(dt)
        j = 0
        boundary = epsilon * pGAdj.shape[0] * dt

        for i in range(maxIter):
            tempVal = ((pGAdj @ (1 - Psi)) - Psi) * dt
            Psi += tempVal.real

            if i % checkStep == 0:
                if np.abs(tempVal).sum() < boundary:
                    break
                if j < len(maxVals):
                    maxVals[j] = tempVal.max()
                    # Divergence check
                    if j > 0:
                        if maxVals[j] > maxVals[j - 1] and maxVals[j - 1] > maxVals[j - 2]:
                            return False, Psi
                    j += 1

        return True, Psi
    else:
        if sigma == -1:
            sigma, _ = optimal_sigma(G, analytical=True, dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)

            # Solve linear system: (sigma*A + I) * Psi = sigma * A * 1
        # The equation is derived from setting dPsi/dt = 0
        # sigma * A * (1 - Psi) - Psi = 0  =>  sigma*A*1 - sigma*A*Psi - Psi = 0
        # => (sigma*A + I) * Psi = sigma * A * 1

        Identity = sp.sparse.eye(G.shape[0], format='csc')
        A = sigma * G
        b = A.sum(axis=-1)  # sigma * A * 1 (vector of row sums)

        # Use spsolve for exact solution
        try:
            Psi = sp.sparse.linalg.spsolve(A + Identity, b)
            return True, Psi
        except RuntimeError:
            # Matrix might be singular if sigma is exactly the eigenvalue
            return False, np.zeros(G.shape[0])


def find_eigenvalue(G, minVal=0, maxVal=1, maxDepth=100, dt=0.1, epsilon=1e-5, maxIter=100, checkStep=10):
    '''
    Finds the largest negative eigenvalue of an adjacency matrix using the DomiRank algorithm's divergence property.
    '''
    # Initial guess scaling
    x = (minVal + maxVal) / G.sum(axis=-1).max()
    minValStored = 0

    for i in range(maxDepth):
        if maxVal - minVal < epsilon:
            break

        converged, _ = domirank(G, analytical=False, sigma=x, dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)

        if converged:
            minVal = x
            minValStored = minVal
            x = (minVal + maxVal) / 2
        else:
            maxVal = x  # If it diverged, x is too "large" (too close to singularity or past it)
            x = (minVal + maxVal) / 2

        if minVal == 0:
            print(f'Current Interval : [-inf, -{1 / maxVal}]')
        else:
            print(f'Current Interval : [-{1 / minVal}, -{1 / maxVal}]')

    finalVal = (maxVal + minVal) / 2
    if finalVal == 0: return -1  # Prevent division by zero
    return -1 / finalVal


############## This section is for finding the optimal sigma #######################

def process_iteration_wrapper(args):
    '''Helper function to unpack arguments for multiprocessing pool'''
    return process_iteration(*args)


def process_iteration(analytical, sigma, spArray, maxIter, checkStep, dt, epsilon, sampling):
    '''
    Modified to RETURN values instead of using a Queue.
    This is safer and prevents deadlocks/ordering issues.
    '''
    tf, domiDist = domirank(spArray, analytical=analytical, sigma=sigma, dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)
    domiAttack = generate_attack(domiDist)
    ourTempAttack, __ = network_attack_sampled(spArray, domiAttack, sampling=sampling)
    finalErrors = ourTempAttack.sum()
    return finalErrors


def optimal_sigma(spArray, analytical=True, endVal=0, startval=0.000001, iterationNo=10, dt=0.1, epsilon=1e-5, maxIter=100, checkStep=10, maxDepth=100, sampling=0):
    '''
    Finds the optimal sigma by searching the space.
    Rewritten to use multiprocessing.Pool for correct ordering of results.
    '''
    if endVal == 0:
        print("Calculating eigenvalue bounds...")
        endVal = find_eigenvalue(spArray, maxDepth=maxDepth, dt=dt, epsilon=epsilon, maxIter=maxIter, checkStep=checkStep)

    # Calculate range
    # Note: using abs(endVal) to ensure correct range direction if endVal is negative
    endval_inverted = -0.9999 / endVal
    step_size = (endval_inverted - startval) / iterationNo
    tempRange = np.arange(startval, endval_inverted + step_size, step_size)

    # Cap the range length if it slightly exceeded iterationNo due to float precision
    if len(tempRange) > iterationNo + 1:
        tempRange = tempRange[:iterationNo + 1]

    print(f"Searching optimal sigma in range [{startval} ... {endval_inverted}] with {len(tempRange)} steps.")

    # Prepare arguments for multiprocessing
    # We must tuples of arguments for starmap
    tasks = []
    for sigma in tempRange:
        tasks.append((analytical, sigma, spArray, maxIter, checkStep, dt, epsilon, sampling))

    # Use Pool to execute. This guarantees that results[i] corresponds to tempRange[i]
    # Queue in the original code did NOT guarantee this (FIFO based on completion time).
    with mp.Pool(mp.cpu_count()) as pool:
        finalErrors = pool.starmap(process_iteration, tasks)

    finalErrors = np.array(finalErrors)

    # Find index of minimum error
    min_idx = np.argmin(finalErrors)
    best_sigma = tempRange[min_idx]

    print(f"Optimal Sigma found: {best_sigma}")
    return best_sigma, finalErrors