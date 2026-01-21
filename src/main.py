import sys
import os
import torch
import warnings

from src.active_learning.al_techniques.margin_al_technique import MarginALTechnique
from src.active_learning.al_techniques.entropy_al_technique import EntropyALTechnique
from src.active_learning.al_techniques.lcs_al_technique import LCSALTechnique
from src.active_learning.al_techniques.random_al_technique import RandomALTechnique
from src.active_learning.node_sampler import ActiveLearningSampler
from src.models.gat.trainer import train, evaluate
from src.dataset.dataset_loader import load_dataset
from src.influence.proxy.IC_proxy import compute_ic_like_influence_scores
from src.influence.influence_groups import compare_scores_and_return_groups
from src.models.gat.hetero_gat import HeteroGAT
from src.models.gat.mixed_loss import MixedLoss
from src.influence.simulation.influence_score_ic import compute_influence_scores
from src.support.arguments import parse_arguments
from src.support.utils import get_device, save_influence_to_json, get_base_dir, to_categorical, build_combined_output, filter_combined_output, print_metrics, Color, cprint
from src.support.utils_graph import build_metapath_graphs, compute_layer_probabilities, k_hop_subgraph


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parse_arguments()

    al_technique = getattr(sys.modules[__name__], args.al_technique)
    cprint(f"Running experiment on dataset: {args.dataset_name}", Color.EXPERIMENT_CONFIG_INFO)
    results_dir = os.path.join(get_base_dir(), args.dataset_name, "influential_nodes", "results")
    cprint(f"Saving experiment results in: '{results_dir}'", Color.EXPERIMENT_CONFIG_INFO)
    os.makedirs(results_dir, exist_ok=True)
    device = get_device()
    data = load_dataset(args.dataset_name, reduction_factor=args.reduction_factor, k=args.num_hops, device=device)
    layer_graphs = build_metapath_graphs(data)
    layer_probs = compute_layer_probabilities(layer_graphs, args.beta)

    # IC model - ground truth
    ic_scores = compute_influence_scores(layer_graphs=layer_graphs, layer_probs=layer_probs, num_steps=args.num_steps, n_sim=args.n_sim, seed=42, out_dir = results_dir)

    # IC-like scores
    proxy_scores = compute_ic_like_influence_scores(layer_graphs, layer_probs, normalize=False)
    save_influence_to_json(proxy_scores, os.path.join(results_dir, f"influence_scores_IC_proxy.json"))

    # comparison between scores
    groups_ic, groups_proxy = compare_scores_and_return_groups(ic_scores, proxy_scores, results_dir)

    # data to categorical for first output
    categorical_ic_scores = to_categorical(ic_scores, args.influence_levels)

    # merging and storing results in data data[data.target_type].y
    combined_output = build_combined_output(categorical_ic_scores, ic_scores, proxy_scores, device)

    # training procedure
    cprint("Building model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    model = HeteroGAT(metadata=data.metadata(), target_type=data.target_type, hidden_channels=args.hidden_channels, out_channels=args.out_channels, dropout=args.dropout, num_layers=args.num_layers).to(device)
    node_sampler = ActiveLearningSampler(al_technique(model), args.k)
    criterion = MixedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    cprint("Preparing data...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    size = data[data.target_type].x.shape[0]
    n_samples_training_set = int(size * args.percentage_training_set)
    n_samples_labeled_set = int(size * args.percentage_labeled_set)
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

    # calculating train and test graph and plugging labels in combined_output into them
    train_data, _ = k_hop_subgraph(data, torch.nonzero(train_mask).flatten(), 2)
    train_data[data.target_type].y = filter_combined_output(combined_output, torch.nonzero(train_mask).flatten())
    test_data, _ = k_hop_subgraph(data, torch.nonzero(test_mask).flatten(), 2)
    test_data[data.target_type].y = filter_combined_output(combined_output, torch.nonzero(test_mask).flatten())
    # AL cycles
    for cycle in range(args.al_cycles):
        print("+" * 50)
        cprint(f"Cycle {cycle+1}", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        cprint("Training model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        train(model, train_data, optimizer, criterion, train_mask_labeled, n_epochs=args.training_epochs)
        cprint("Evaluating model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        report = evaluate(model, test_data)
        print_metrics(report)
        # AL sampling - updating "train_mask_labeled" adding the new selected nodes
        if cycle < args.al_cycles - 1:
            cprint("Making AL selection...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
            train_mask_labeled = node_sampler.sample(train_data, train_mask_labeled)

    cprint("Completed!", Color.OTHER)
