import sys
import warnings

from src._trash.active_learning.node_sampler import ActiveLearningSampler
from src.mixed_loss.nc_loss import NCLoss
from src._trash.models.gat.trainer import train, evaluate
from src._trash.dataset.dataset_loader import load_dataset
from src._trash.influence.proxy.IC_proxy import compute_ic_like_influence_scores
from src._trash.influence.influence_groups import compare_scores_and_return_groups
from src._trash.models.gat.hetero_gat import HeteroGAT
from src.mixed_loss.mixed_loss import MixedLoss
from src.mixed_loss.mae_loss import MAELoss
from src._trash.support.arguments import parse_arguments
from src._trash.support.utils import *
from src._trash.support.utils_graph import build_metapath_graphs, compute_layer_probabilities


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
    #ic_scores = compute_influence_scores(layer_graphs=layer_graphs, layer_probs=layer_probs, num_steps=args.num_steps, n_sim=args.n_sim, seed=args.seed, descending_order=args.descending_order, out_dir=results_dir)
    ic_scores = compute_centrality_measure(data, "betweenness")
    proxy_scores = compute_ic_like_influence_scores(layer_graphs, layer_probs, normalize=False)
    save_influence_to_json(proxy_scores, os.path.join(results_dir, f"influence_scores_IC_proxy.json"))

    # comparison between scores
    groups_ic, groups_proxy = compare_scores_and_return_groups(ic_scores, proxy_scores, results_dir)
    if args.show_plots:
        plot_ranking_comparison(torch.tensor(list(ic_scores.values())), torch.tensor(list(proxy_scores.values())), ("IC scores", "Proxy scores"))

    cprint("Comparing rankings: IC vs proxy...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    print_ranking_comparison(ic_scores, proxy_scores)

    # training procedure
    cprint("Building model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    model = HeteroGAT(metadata=data.metadata(), target_type=data.target_type, hidden_channels=args.hidden_channels, out_channels=args.influence_levels, dropout=args.dropout, num_layers=args.num_layers).to(device)
    node_sampler = ActiveLearningSampler(al_technique(model), args.k)
    if args.loss == "mixed_loss":
        criterion = MixedLoss(weight_ic_classification=args.weight_ic_classification, weight_proxy_component=args.weight_ic_regression, weight_proxy_regression=args.weight_proxy_regression, weight_consistency=args.weight_consistency)

    elif args.loss == "mae":
        criterion = MAELoss()

    elif args.loss == "nc":
        criterion = NCLoss()

    else:
        raise ValueError(f"Loss function '{args.loss}' not recognized.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    cprint("Preparing data...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    size = data[data.target_type].x.shape[0]
    n_samples_training_set = int(size * args.percentage_training_set)
    n_samples_labeled_set = int(n_samples_training_set * args.percentage_labeled_set)
    seed_mask = torch.randperm(size)
    if args.stratified_sampling:
        seed_mask = stratified_sampling(proxy_scores, seed_mask, n_samples_labeled_set, args.influence_levels)

    # building masks
    train_mask_global = torch.zeros(size, dtype=torch.int)
    train_mask_global[seed_mask[: n_samples_training_set]] = 1

    train_mask = torch.ones(n_samples_training_set, dtype=torch.int)

    train_mask_labeled = torch.zeros(n_samples_training_set, dtype=torch.int)
    train_mask_labeled[seed_mask[(seed_mask < len(train_mask_labeled)) & (seed_mask >= 0)][:n_samples_labeled_set]] = 1
    train_mask_labeled = train_mask_labeled.unsqueeze(1).to(device)

    train_mask_unlabeled = torch.zeros(n_samples_training_set, dtype=torch.int)
    train_mask_unlabeled[seed_mask[(seed_mask < len(train_mask_unlabeled)) & (seed_mask >= 0)][n_samples_labeled_set: n_samples_training_set]] = 1
    train_mask_unlabeled = train_mask_unlabeled.unsqueeze(1).to(device)

    test_mask = torch.zeros(size, dtype=torch.int)
    test_mask[seed_mask[n_samples_training_set: ]] = 1
    test_mask = test_mask.unsqueeze(1).to(device)

    # normalization of influence scores and data analysis
    train_indices = train_mask_labeled.flatten().nonzero().cpu().flatten().tolist()
    test_indices = test_mask.flatten().nonzero().cpu().flatten().tolist()
    ic_scores = normalize(ic_scores, train_indices, normalization_type="min-max")
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

    train_data, train_data_mask = k_hop_subgraph(data, train_mask_global.flatten().nonzero().cpu().flatten(), 2, True)
    train_data[data.target_type].y = filter_combined_output(combined_output, train_data_mask[data.target_type])
    test_data, test_data_mask = k_hop_subgraph(data, test_mask.flatten().nonzero().cpu().flatten(), 2, False)
    test_data[data.target_type].y = filter_combined_output(combined_output, test_data_mask[data.target_type])

    # AL cycles
    for cycle in range(args.al_cycles):
        print("+" * 50)
        cprint(f"Cycle {cycle+1}", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        cprint("Training model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        train(model, train_data, optimizer, criterion, train_mask_labeled, n_epochs=args.training_epochs)
        cprint("Evaluating model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        report = evaluate(model, train_data, train_mask, -1)
        cprint("on training set...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        print_metrics(report)
        report = evaluate(model, test_data, torch.ones(test_data[data.target_type]["x"].shape[0]), -1)
        cprint("on test set...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
        print_metrics(report)
        # AL sampling - updating "train_mask_labeled" adding the new selected nodes
        if cycle < args.al_cycles - 1:
            cprint("Making AL selection...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
            node_sampler.sample(train_data, train_mask_labeled, train_mask_unlabeled)

    cprint(f"Saving model in '{results_dir}/hetero_gat_model_final.ckpt'...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    torch.save(model.state_dict(), f"{results_dir}/hetero_gat_model_final.ckpt")

    cprint("Final evaluation on test set with connectivity computation...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    report = evaluate(model, test_data, torch.ones(test_data[data.target_type]["x"].shape[0]), args.connectivity_evaluation_bound)

    # if args.connectivity_evaluation_bound > 0:
    #     all_sorted_indices = sorted(ic_scores.keys(), key=lambda x: ic_scores[x], reverse=True)
    #     ic_con_values = compute_incremental_con_measure(all_sorted_indices, args.connectivity_evaluation_bound, data, False)
    #     report["ic_con_measures"] = {"values": ic_con_values}
    #     all_sorted_indices = sorted(proxy_scores.keys(), key=lambda x: proxy_scores[x], reverse=True)
    #     proxy_con_values = compute_incremental_con_measure(all_sorted_indices, args.connectivity_evaluation_bound, data, False)
    #     report["proxy_con_measures"] = {"values": proxy_con_values}

    print_metrics(report)

    cprint("Comparing ranking with IC model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    mask = test_data_mask[data.target_type] #test_mask.squeeze().bool()
    out_model = model(test_data.x_dict, test_data.edge_index_dict)
    ic_test_values = {idx.item(): val.item() for idx, val in zip(mask.nonzero().flatten(), test_data[data.target_type].y[1])}
    model_test_values = {idx.item(): val.item() for idx, val in zip(mask.nonzero().flatten(), out_model[1][test_data.target_type])}
    print_ranking_comparison(ic_test_values, model_test_values)
    cprint("Comparing ranking with proxy score...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    proxy_test_values = {idx.item(): val.item() for idx, val in zip(mask.nonzero().flatten(), test_data[test_data.target_type].y[2])}
    model_test_values = {idx.item(): val.item() for idx, val in zip(mask.nonzero().flatten(), out_model[2][test_data.target_type])}
    print_ranking_comparison(proxy_test_values, model_test_values)

    # if args.show_plots:
    #     plot_ranking_comparison(data[data.target_type].y[1][test_mask.squeeze().bool()], model(data.x_dict, data.edge_index_dict)[1][data.target_type][test_mask.squeeze().bool()], ("IC scores real", "IC scores predicted"))
    #     plot_ranking_comparison(data[data.target_type].y[2][test_mask.squeeze().bool()], model(data.x_dict, data.edge_index_dict)[2][data.target_type][test_mask.squeeze().bool()], ("Proxy scores real", "Proxy scores predicted"))

    cprint("Completed!", Color.OTHER)
