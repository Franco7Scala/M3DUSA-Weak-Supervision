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
from src.support.utils import get_device, save_influence_to_json, get_base_dir, to_categorical, build_combined_output, filter_combined_output, print_metrics, Color, cprint, normalize, print_args, seed_everything, print_data_analysys, plot_ranking_comparison, plot_data_distribution, apply_pca
from src.support.utils_graph import build_metapath_graphs, compute_layer_probabilities, k_hop_subgraph


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parse_arguments()
    print_args(args)
    seed_everything(args.seed)

    al_technique = getattr(sys.modules[__name__], args.al_technique)
    cprint(f"Running experiment on dataset: {args.dataset_name}", Color.EXPERIMENT_CONFIG_INFO)
    results_dir = os.path.join(get_base_dir(), args.dataset_name, "influential_nodes", "results")
    cprint(f"Saving experiment results in: '{results_dir}'", Color.EXPERIMENT_CONFIG_INFO)
    os.makedirs(results_dir, exist_ok=True)
    device = get_device()
    data = load_dataset(args.dataset_name, reduction_factor=args.reduction_factor, k=args.num_hops, device=device)
    layer_graphs = build_metapath_graphs(data)
    layer_probs = compute_layer_probabilities(layer_graphs, args.beta)

    # influence scores computation
    ic_scores = compute_influence_scores(layer_graphs=layer_graphs, layer_probs=layer_probs, num_steps=args.num_steps, n_sim=args.n_sim, seed=args.seed, out_dir=results_dir)
    proxy_scores = compute_ic_like_influence_scores(layer_graphs, layer_probs, normalize=False)
    save_influence_to_json(proxy_scores, os.path.join(results_dir, f"influence_scores_IC_proxy.json"))

    # comparison between scores
    groups_ic, groups_proxy = compare_scores_and_return_groups(ic_scores, proxy_scores, results_dir)
    if args.show_plots:
        plot_ranking_comparison(torch.tensor(list(ic_scores.values())), torch.tensor(list(proxy_scores.values())), ("IC scores", "Proxy scores"))

    # training procedure
    cprint("Building model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    model = HeteroGAT(metadata=data.metadata(), target_type=data.target_type, hidden_channels=args.hidden_channels, out_channels=args.influence_levels, dropout=args.dropout, num_layers=args.num_layers).to(device)
    node_sampler = ActiveLearningSampler(al_technique(model), args.k)
    criterion = MixedLoss(weight_ic_classification=args.weight_ic_classification, weight_ic_regression=args.weight_ic_regression, weight_proxy_regression=args.weight_proxy_regression)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    cprint("Preparing data...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    size = data[data.target_type].x.shape[0]
    n_samples_training_set = int(size * args.percentage_training_set)
    n_samples_labeled_set = int(n_samples_training_set * args.percentage_labeled_set)
    seed_mask = torch.randperm(size)

    # building masks
    train_mask_labeled = torch.zeros(size, dtype=torch.int)
    train_mask_labeled[seed_mask[: n_samples_labeled_set]] = 1
    train_mask_labeled = train_mask_labeled.unsqueeze(1).to(device)
    train_mask_unlabeled = torch.zeros(size, dtype=torch.int)
    train_mask_unlabeled[seed_mask[n_samples_labeled_set: n_samples_training_set]] = 1
    train_mask_unlabeled = train_mask_unlabeled.unsqueeze(1).to(device)
    test_mask = torch.zeros(size, dtype=torch.int)
    test_mask[seed_mask[n_samples_training_set: ]] = 1
    test_mask = test_mask.unsqueeze(1).to(device)

    # normalization of influence scores and data analysis
    train_indices = train_mask_labeled.flatten().nonzero().cpu().flatten().tolist()
    ic_scores = normalize(ic_scores, train_indices, normalization_type="z-score")
    cprint(f"Analysis on IC data:", Color.EXPERIMENT_CONFIG_INFO)
    print_data_analysys(torch.tensor(list(ic_scores.values())))
    if args.show_plots:
        plot_data_distribution(torch.tensor(list(ic_scores.values())), "IC score")

    proxy_scores = normalize(proxy_scores, train_indices, normalization_type="qt")
    cprint(f"Analysis on proxy data:", Color.EXPERIMENT_CONFIG_INFO)
    print_data_analysys(torch.tensor(list(proxy_scores.values())))
    if args.show_plots:
        plot_data_distribution(torch.tensor(list(proxy_scores.values())), "Proxy score")

    categorical_ic_scores = to_categorical(ic_scores, args.influence_levels)
    combined_output = build_combined_output(categorical_ic_scores, ic_scores, proxy_scores, device)
    data[data.target_type].y = combined_output

    # AL cycles
    for cycle in range(args.al_cycles):
        print("+" * 50)
        cprint(f"Cycle {cycle+1}", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        cprint("Training model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        train(model, data, optimizer, criterion, train_mask_labeled, n_epochs=args.training_epochs)
        cprint("Evaluating model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        report = evaluate(model, data, train_mask_labeled)
        cprint("on training set...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        print_metrics(report)
        report = evaluate(model, data, test_mask)
        cprint("on test set...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        print_metrics(report)
        # AL sampling - updating "train_mask_labeled" adding the new selected nodes
        if cycle < args.al_cycles - 1:
            cprint("Making AL selection...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
            train_mask_labeled = node_sampler.sample(data, train_mask_labeled, train_mask_unlabeled)

    cprint(f"Saving model in '{results_dir}/hetero_gat_model_final.ckpt'...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    torch.save(model.state_dict(), f"{results_dir}/hetero_gat_model_final.ckpt")
    if args.show_plots:
        plot_ranking_comparison(data[data.target_type].y[1][test_mask.squeeze().bool()], model(data.x_dict, data.edge_index_dict)[1][data.target_type][test_mask.squeeze().bool()], ("IC scores real", "IC scores predicted"))
        plot_ranking_comparison(data[data.target_type].y[2][test_mask.squeeze().bool()], model(data.x_dict, data.edge_index_dict)[2][data.target_type][test_mask.squeeze().bool()], ("IC scores real", "IC scores predicted"))

    cprint("Completed!", Color.OTHER)
