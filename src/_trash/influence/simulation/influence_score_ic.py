import random
import os
import json
from tqdm import tqdm
from network_diffusion import mln
from network_diffusion.simulator import Simulator

from src._trash.models.diffusion_multilayer_ic import MultiLayerICModel
from src._trash.support.utils import save_influence_to_csv, save_influence_to_json
from src._trash.support.utils_graph import precompute_neighbor_probs

"""
Computes influence of each node as if it's the ONLY seed.
"""
def compute_influence_scores(layer_graphs, layer_probs, num_steps=5, n_sim=200, seed=42, descending_order=False, out_dir=None):
    if seed is not None:
        random.seed(seed)

    network = mln.MultilayerNetwork(layer_graphs)
    # Precompute neighbor probabilities + nodes_in_layer
    neighbor_probs, nodes_in_layer = precompute_neighbor_probs(layer_graphs, layer_probs)

    # Collect all nodes across all layers

    nodes = set()
    for lnodes in nodes_in_layer.values():
        nodes |= set(lnodes)
    nodes = sorted(nodes, reverse=descending_order)

    out_file = os.path.join(out_dir, f"influence_scores_{num_steps}steps.json")

    if os.path.exists(out_file):
        with open(out_file, "r") as f:
            influence = {int(k): v for k, v in json.load(f).items()}
        print(f"Resuming from checkpoint. {len(influence)} nodes already processed.")
        # EARLY EXIT if all nodes are already processed
        if set(influence.keys()) == set(nodes):
            print("All nodes already processed. Returning cached influence scores.")
            return influence
    else:
        influence = {}

    influence_checkpoint = {}

    # Outer loop: iterate over nodes as single seed
    for node in tqdm(nodes, desc="Nodes processed"):

        if node in influence:
            continue  # already processed

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

        score = round(total_spread / n_sim, 2)
        influence[node] = score
        influence_checkpoint[node] = score
        print(f"\n Node {node}: {score}")

        if node%20==0:
            save_influence_to_csv(influence_checkpoint, os.path.join(out_dir, f"influence_scores_{num_steps}steps.csv"))
            save_influence_to_json(influence_checkpoint, os.path.join(out_dir, f"influence_scores_{num_steps}steps.json"))
            influence_checkpoint = {}

    save_influence_to_csv(influence, os.path.join(out_dir, f"influence_scores_{num_steps}steps.csv"))
    save_influence_to_json(influence, os.path.join(out_dir, f"influence_scores_{num_steps}steps.json"))

    return influence
