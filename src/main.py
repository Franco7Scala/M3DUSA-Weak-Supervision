import os

from src.dataset_loader import load_dataset
from src.IC_proxy import compute_ic_like_influence_scores
from src.influence_groups import compare_scores_and_return_groups
from src.simulation.influence_score_ic import compute_influence_scores
from src.utils import get_device, save_influence_to_json, get_base_dir
from src.utils_graph import build_metapath_graphs, compute_layer_probabilities


def main():

    dataset_name = "imdb"
    beta = 0.55 #0.55 for academic networks, 0.85 for social networks
    num_steps = 3 # number of diffusion epochs (5-20)
    n_sim = 3 # number of independent Monte-Carlo simulations per seed.  200+ for research-quality results, 20–50 for quick experiments
    reduction_factor = 1#4
    num_hops = 2
    results_dir = os.path.join(get_base_dir(), dataset_name, "influential_nodes", "results")
    os.makedirs(results_dir, exist_ok=True)

    device = get_device()
    data = load_dataset(dataset_name, reduction_factor=reduction_factor, k=num_hops, device=device)
    layer_graphs = build_metapath_graphs(data)
    layer_probs = compute_layer_probabilities(layer_graphs, beta)

    #IC model - ground truth
    ic_scores = compute_influence_scores(
        layer_graphs=layer_graphs,
        layer_probs=layer_probs,
        num_steps=num_steps,
        n_sim=n_sim,
        seed=42,
        out_dir = results_dir #salva internamente chackpoint
    )

    #IC-like scores
    proxy_scores = compute_ic_like_influence_scores(layer_graphs, layer_probs, normalize=False)
    save_influence_to_json(proxy_scores, os.path.join(results_dir, f"influence_scores_IC_proxy.json"))

    #confronto tra gli scores
    groups_ic, groups_proxy = compare_scores_and_return_groups(ic_scores, proxy_scores, results_dir) #qui dentro calcolo gruppi

    #TODO: da quali gruppi faccio sampling? penso groups_proxy

    #TODO: GAT training

    #TODO: AL


if __name__ == "__main__":
    main()
