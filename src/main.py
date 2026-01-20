import os
import torch
import warnings

from src.active_learning.node_sampler import sample_nodes
from src.models.gat.trainer import train, evaluate
from src.dataset.dataset_loader import load_dataset
from src.influence.proxy.IC_proxy import compute_ic_like_influence_scores
from src.influence.influence_groups import compare_scores_and_return_groups
from src.models.gat.hetero_gat import HeteroGAT
from src.models.gat.mixed_loss import MixedLoss
from src.influence.simulation.influence_score_ic import compute_influence_scores
from src.utils import get_device, save_influence_to_json, get_base_dir, to_categorical, build_combined_output, filter_combined_output, print_metrics
from src.utils_graph import build_metapath_graphs, compute_layer_probabilities, k_hop_subgraph


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    dataset_name = "imdb"

    # IC model parameters
    beta = 0.55 #0.55 for academic networks, 0.85 for social networks
    num_steps = 3 # number of diffusion epochs (5-20)
    n_sim = 3 # number of independent Monte-Carlo simulations per seed.  200+ for research-quality results, 20–50 for quick experiments
    reduction_factor = 1#4 - (8 per politifact)
    num_hops = 2
    results_dir = os.path.join(get_base_dir(), dataset_name, "influential_nodes", "results")

    # model parameters
    hidden_channels = 3
    out_channels = 3
    dropout = 0.3
    num_layers = 3

    # data parameters
    influence_levels = 3  # number of influence levels (e.g., low, medium, high)

    # training parameters
    learning_rate = 0.01
    weight_decay = 0.001
    training_epochs = 5
    percentage_training_set = 0.8
    percentage_labeled_set = 0.2
    al_cycles = 4

    ##############################################################################

    os.makedirs(results_dir, exist_ok=True)
    device = get_device()
    data = load_dataset(dataset_name, reduction_factor=reduction_factor, k=num_hops, device=device)
    layer_graphs = build_metapath_graphs(data)
    layer_probs = compute_layer_probabilities(layer_graphs, beta)

    # IC model - ground truth
    ic_scores = compute_influence_scores(layer_graphs=layer_graphs, layer_probs=layer_probs, num_steps=num_steps, n_sim=n_sim, seed=42, out_dir = results_dir)

    # IC-like scores
    proxy_scores = compute_ic_like_influence_scores(layer_graphs, layer_probs, normalize=False)
    save_influence_to_json(proxy_scores, os.path.join(results_dir, f"influence_scores_IC_proxy.json"))

    # comparison between scores
    groups_ic, groups_proxy = compare_scores_and_return_groups(ic_scores, proxy_scores, results_dir)

    # data to categorical for first output
    categorical_ic_scores = to_categorical(ic_scores, influence_levels)

    # merging and storing results in data data[data.target_type].y
    combined_output = build_combined_output(categorical_ic_scores, ic_scores, proxy_scores, device)

    # training procedure
    model = HeteroGAT(metadata=data.metadata(), target_type=data.target_type, hidden_channels=hidden_channels, out_channels=out_channels, dropout=dropout, num_layers=num_layers).to(device)
    criterion = MixedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    size = data[data.target_type].x.shape[0]
    n_samples_training_set = int(size * percentage_training_set)
    n_samples_labeled_set = int(size * percentage_labeled_set)
    seed_mask = torch.randperm(size)
    train_mask = torch.zeros(size, dtype=torch.int)
    train_mask[seed_mask[: n_samples_training_set]] = 1
    test_mask = torch.zeros(size, dtype=torch.int)
    test_mask[seed_mask[n_samples_training_set: ]] = 1

    # defining mask for labeled and unlabeled nodes in the training set
    train_size = len(torch.nonzero(train_mask).flatten())

    train_mask_labeled = torch.zeros(train_size, dtype=torch.int)
    indices = seed_mask[:n_samples_labeled_set]
    limit = train_mask_labeled.shape[0]
    valid_indices = indices[indices < limit]
    train_mask_labeled[valid_indices] = 1
    train_mask_labeled = train_mask_labeled.unsqueeze(1).to(device)

    train_mask_unlabeled = torch.zeros(train_size, dtype=torch.int)
    indices_unlabeled = seed_mask[n_samples_labeled_set: n_samples_training_set]
    limit_unlabeled = train_mask_unlabeled.shape[0]
    valid_indices_unlabeled = indices_unlabeled[indices_unlabeled < limit_unlabeled]
    train_mask_unlabeled[valid_indices_unlabeled] = 1
    train_mask_unlabeled = train_mask_unlabeled.unsqueeze(1).to(device)

    # calculating train and test graph and plugging labels in combined_output into them
    train_data, _ = k_hop_subgraph(data, torch.nonzero(train_mask).flatten(), 2)
    train_data[data.target_type].y = filter_combined_output(combined_output, torch.nonzero(train_mask).flatten())
    test_data, _ = k_hop_subgraph(data, torch.nonzero(test_mask).flatten(), 2)
    test_data[data.target_type].y = filter_combined_output(combined_output, torch.nonzero(test_mask).flatten())
    # AL cycles
    for cycle in range(al_cycles):
        print(f"Cycle {cycle+1}")
        train(model, train_data, optimizer, criterion, train_mask_labeled, n_epochs=training_epochs)
        report = evaluate(model, test_data)
        print_metrics(report)
        # TODO implement active learning sampling strategy to update sampling_mask
        # update "train_mask_labeled" with the selected nodes
        sample_nodes(train_data, train_mask_labeled, model)



    a, b, c = model(data.x_dict, data.edge_index_dict)



    loss = criterion((a[data.target_type], b[data.target_type], c[data.target_type]), combined_output)

    print(f"Loss:{loss.item()}")
