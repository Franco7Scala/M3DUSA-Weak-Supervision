import warnings

from src.dataset.dataset_loader import load_dataset
from src.influence.proxy.IC_proxy import compute_ic_like_influence_scores
from src.models.gat.hetero_gat import HeteroGAT
from src.influence.simulation.influence_score_ic import compute_influence_scores
from src.support.arguments import parse_arguments
from src.support.utils import *
from src.support.utils_graph import build_metapath_graphs, compute_layer_probabilities
from torch_geometric.utils import to_networkx


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parse_arguments()
    print_args(args)
    seed_everything(args.seed)

    cprint(f"Running experiment on dataset: {args.dataset_name}", Color.EXPERIMENT_CONFIG_INFO)
    results_dir = os.path.join(get_base_dir(), args.dataset_name, "influential_nodes", "results")
    cprint(f"Saving experiment results in: '{results_dir}'", Color.EXPERIMENT_CONFIG_INFO)
    os.makedirs(results_dir, exist_ok=True)
    device = "cpu"
    data = load_dataset(args.dataset_name, reduction_factor=args.reduction_factor, k=args.num_hops, device=device)
    layer_graphs = build_metapath_graphs(data)
    layer_probs = compute_layer_probabilities(layer_graphs, args.beta)

    # influence scores computation
    ic_scores = compute_influence_scores(layer_graphs=layer_graphs, layer_probs=layer_probs, num_steps=args.num_steps, n_sim=args.n_sim, seed=args.seed, out_dir=results_dir)
    proxy_scores = compute_ic_like_influence_scores(layer_graphs, layer_probs, normalize=False)
    save_influence_to_json(proxy_scores, os.path.join(results_dir, f"influence_scores_IC_proxy.json"))

    # training procedure
    cprint("Building model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    model = HeteroGAT(metadata=data.metadata(), target_type=data.target_type, hidden_channels=args.hidden_channels, out_channels=args.influence_levels, dropout=args.dropout, num_layers=args.num_layers).to(device)

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
    ic_scores = normalize(ic_scores, train_indices, normalization_type="min-max")
    cprint(f"Analysis on IC data:", Color.EXPERIMENT_CONFIG_INFO)
    print_data_analysys(torch.tensor(list(ic_scores.values())))
    if args.show_plots:
        plot_data_distribution(torch.tensor(list(ic_scores.values())), "IC score")

    ordered_ic_scores = [ic_scores[k] for k in sorted(ic_scores.keys())]
    data[data.target_type].y = torch.tensor(ordered_ic_scores)

    cprint("Building NetworkX graph for centrality analysis...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    g_total = to_networkx(data.to_homogeneous(), to_undirected=True)
    cprint(f"Graph built successfully: {g_total.number_of_nodes()} nodes, {g_total.number_of_edges()} edges.", Color.EXPERIMENT_CONFIG_INFO)
    user_id_map = {}
    current_offset = 0
    cprint(f"Calculating index mapping for target type: '{data.target_type}'...", Color.EXPERIMENT_STATUS_LOW_PRIORITY)
    for node_type in data.node_types:
        num_nodes_in_type = data[node_type].num_nodes
        if node_type == data.target_type:
            user_id_map = {current_offset + i: i for i in range(num_nodes_in_type)}
            break

        current_offset += num_nodes_in_type

    test_indices = test_mask.squeeze().nonzero().flatten().tolist()
    cprint(f"Processing {args.centrality_measure}...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    #centrality_scores_dict = get_topk_centrality_nodes(g_total, user_id_map, centrality_measure=args.centrality_measure)

    #variante multilayer
    centrality_scores_dict = get_topk_centrality_nodes_multilayer(layer_graphs, layer_probs, user_id_map, centrality_measure=args.centrality_measure, normalization="l1")
    y_real_list = [ic_scores[i] for i in test_indices]
    y_pred_list = [centrality_scores_dict[i] for i in test_indices]
    cprint("Final evaluation on test set with connectivity computation...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    report = evaluate_centrality_measure(centrality_scores_dict, data, connectivity_bound=args.connectivity_evaluation_bound)
    print_metrics(report)
    cprint(f"Comparing {args.centrality_measure} with IC model...", Color.EXPERIMENT_STATUS_HIGH_PRIORITY)
    y_real = {test_indices[i]: y_real_list[i] for i in range(len(test_indices))}
    y_pred = {test_indices[i]: y_pred_list[i] for i in range(len(test_indices))}
    print_ranking_comparison(y_real, y_pred)

    if args.show_plots:
        y_real_tensor = torch.tensor(y_real_list, dtype=torch.float32)
        y_pred_tensor = torch.tensor(y_pred_list, dtype=torch.float32)
        plot_ranking_comparison(y_real_tensor, y_pred_tensor, ("IC Scores", f"Centrality {args.centrality_measure}"))

cprint("Completed!", Color.OTHER)
