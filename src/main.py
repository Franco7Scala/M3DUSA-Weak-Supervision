import os

from src.dataset_loader import load_dataset
from src.simulation.influence_score import compute_influence_scores
from src.utils import print_top_influencers, get_device, save_influence_to_csv, save_influence_to_json, get_base_dir
from src.utils_graph import build_metapath_graphs, compute_layer_probabilities


def main():

    dataset_name = "imdb"
    beta = 0.55 #0.55 for academic networks, 0.85 for social networks
    num_steps = 3 # number of diffusion epochs (5-20)
    n_sim = 3 # number of independent Monte-Carlo simulations per seed.  200+ for research-quality results, 20–50 for quick experiments
    reduction_factor = 1#4
    num_hops = 2
    results_dir = os.path.join(get_base_dir(), dataset_name, "influential_nodes", "results")
    subset_nodes = list(range(26, 50))  # TODO #None for all nodes

    device = get_device()
    data = load_dataset(dataset_name, reduction_factor=reduction_factor, k=num_hops, device=device)
    layer_graphs = build_metapath_graphs(data)
    layer_probs = compute_layer_probabilities(layer_graphs, beta)

    scores = compute_influence_scores(
        nodes = subset_nodes,
        layer_graphs=layer_graphs,
        layer_probs=layer_probs,
        num_steps=num_steps,
        n_sim=n_sim,
        seed=42
    )

    os.makedirs(results_dir, exist_ok=True)
    save_influence_to_csv(scores, os.path.join(results_dir, f"influence_scores_{num_steps}steps.csv"))
    save_influence_to_json(scores, os.path.join(results_dir, f"influence_scores_{num_steps}steps.json"))

    print("Top influencers:")
    print_top_influencers(scores, k=10)


if __name__ == "__main__":
    main()
