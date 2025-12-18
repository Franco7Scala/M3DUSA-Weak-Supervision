import random
from tqdm import tqdm
from network_diffusion import mln
from network_diffusion.simulator import Simulator

from src.diffusion_models.multilayer_ic import MultiLayerICModel
from src.utils_graph import precompute_neighbor_probs

"""
Computes influence of each node as if it's the ONLY seed.
"""
def compute_influence_scores(nodes, layer_graphs, layer_probs, num_steps=5, n_sim=200, seed=42):
    if seed is not None:
        random.seed(seed)

    network = mln.MultilayerNetwork(layer_graphs)
    # Precompute neighbor probabilities + nodes_in_layer
    neighbor_probs, nodes_in_layer = precompute_neighbor_probs(layer_graphs, layer_probs)

    # Collect all nodes across all layers
    if nodes is None:
        nodes = set()
        for lnodes in nodes_in_layer.values():
            nodes |= set(lnodes)

    influence = {}

    # Outer loop: iterate over nodes as single seed
    for node in tqdm(sorted(nodes), desc="Nodes processed"):
        total_spread = 0

        # Create one model for this seed (fast), reuse simulator across n_sim runs
        model = MultiLayerICModel(layer_probs=layer_probs, neighbor_probs=neighbor_probs, nodes_in_layer=nodes_in_layer,
                                  seed_nodes=[node])

        sim = Simulator(model=model, network=network)


        # Run n_sim monte-carlo runs; reuse model + simulator; reset model between runs
        for _ in tqdm(range(n_sim), desc=f"Simulations for node {node}", leave=False):
            # Reset model internal states to initial seeds
            model.reset(seed_nodes=[node])

            # perform propagation (Simulator will call determine_initial_states)
            sim.perform_propagation(n_epochs=num_steps)

            # count global infected nodes
            total_spread += len(model.active)

        influence[node] = round(total_spread / n_sim, 2)
        print(f"\n Node {node}: {influence[node]}")

    return influence
