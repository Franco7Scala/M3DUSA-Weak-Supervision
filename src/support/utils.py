import os
import torch
import csv
import json
import time
import numpy
import copy
import math
import random
import matplotlib.pyplot as plt
import networkx as nx

from enum import Enum
from sklearn.preprocessing import label_binarize, QuantileTransformer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from torch_geometric.data import HeteroData
from torch_geometric.utils import dropout_edge
from scipy.stats import spearmanr, kendalltau
from src.support.ranking_comparison import jaccard_index, precision_at_k, ndcg
from src.support.utils_graph import compute_incremental_con_measure


class Color(Enum):
    EXPERIMENT_CONFIG_INFO = 2
    EXPERIMENT_STATUS_HIGH_PRIORITY = 3
    EXPERIMENT_STATUS_LOW_PRIORITY = 4
    EXPERIMENT_OUTPUT = 6
    WARNING = 5
    OTHER = 7
    BLACK = 8


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_base_dir():
    return "/home/jovyan/projects/InfluentialNodes/dataset/"
    #return "/home/scala/projects/InfluentialNodes/dataset/"
    #return "/home/martirano/data"


def get_time_in_millis():
    return int(round(time.time() * 1000))


def print_top_influencers(scores, k=10):
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for node, score in sorted_scores[:k]:
        print(f"Node {node}: {score:.4f}")


def save_influence_to_csv(influence_dict, filename):
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["node", "expected_spread"])

        for node, score in sorted(influence_dict.items()):
            writer.writerow([node, score])


def save_influence_to_json(influence_dict, filename):
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    existing_data.update(influence_dict)
    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=4)


def compute_auc(y_true, y_pred):
    all_classes = numpy.arange(y_pred.shape[1])
    scores = roc_auc_score(y_true=label_binarize(y_true, classes=all_classes), y_score=y_pred, average=None, multi_class="ovo")
    valid_scores = scores[~numpy.isnan(scores)]
    return numpy.mean(valid_scores)


def merge_masks(masks):
    if len(masks) == 0:
        return []

    if len(masks) == 1:
        masks[0]

    merged_mask = copy.deepcopy(masks[0])
    for mask in masks[1:]:
        for key in merged_mask:
            merged_mask[key] = torch.cat((merged_mask[key], mask[key]), dim=0)

    return merged_mask


def to_categorical(scores, influence_levels):
    group_size = (len(scores) // influence_levels) + 1
    sorted_scores = sorted(list(scores.items()), key=lambda x: x[1])
    sorted_indexes = [couple[0] for couple in sorted_scores]

    groups = []
    for i in range(0, len(sorted_indexes), group_size):
        groups.append(sorted_indexes[i: i + group_size])

    result = {}
    for idx, group in enumerate(groups):
        for node_idx in group:
            result[node_idx] = torch.eye(influence_levels)[idx]

    return result


def build_combined_output(categorical_ic_scores, ic_scores, proxy_scores, device):
    return (torch.cat([categorical_ic_scores[idx_node].unsqueeze(0) for idx_node in ic_scores.keys()]).to(device),
            torch.tensor([ic_scores[idx_node] for idx_node in ic_scores.keys()]).to(device).reshape(-1, 1),
            torch.tensor([proxy_scores[idx_node] for idx_node in proxy_scores.keys()]).to(device).reshape(-1, 1))


def filter_combined_output(combined_output, indices):
    categorical_ic_scores, ic_scores, proxy_scores = combined_output
    return (categorical_ic_scores[indices], ic_scores[indices], proxy_scores[indices])


def fixed_randperm(n, k):
    fixed = torch.arange(k)
    suffix = torch.randperm(n - k) + k
    return torch.cat([fixed, suffix])


def print_metrics(data):
    for head, metrics in data.items():
        print(f"{head.replace('_', ' ').upper()}")
        print("-" * 50)
        for key, value in metrics.items():
            if isinstance(value, (numpy.ndarray, list)):
                formatted_list = ", ".join([f"{x:.4f}" for x in value])
                print(f"{key:<15}: [{formatted_list}]")

            elif isinstance(value, (int, float)):
                print(f"{key:<15}: {value:.4f}")

            else:
                print(f"{key:<15}: {value}")

        print("-" * 50)


def cprint(text, color=Color.BLACK):
    if color == Color.EXPERIMENT_CONFIG_INFO:
        code_color = "\033[94m"

    elif color == Color.EXPERIMENT_STATUS_HIGH_PRIORITY:
        code_color = "\033[32m"

    elif color == Color.EXPERIMENT_STATUS_LOW_PRIORITY:
        code_color = "\033[92m"

    elif color == Color.WARNING:
        code_color = "\033[91m"

    elif color == Color.EXPERIMENT_OUTPUT:
        code_color = "\033[95m"

    elif color == Color.OTHER:
        code_color = "\033[96m"

    else:
        code_color = "\033[0m"

    print(code_color + str(text) + "\033[0m")


def normalize(data: dict, train_indices: list, normalization_type="none") -> dict:
    keys = list(data.keys())
    values = numpy.array(list(data.values()))
    train_values = values[train_indices].reshape(-1, 1)

    if normalization_type == "qt":
        n_quantiles = min(len(train_values), 1000)
        qt = QuantileTransformer(output_distribution="normal", n_quantiles=n_quantiles)
        qt.fit(train_values)
        transformed_values = qt.transform(values.reshape(-1, 1)).flatten()
        return dict(zip(keys, transformed_values))

    if normalization_type == "log":
        epsilon = 1e-5
        return {key: math.log(value + epsilon) for key, value in data.items()}

    if normalization_type == "z-score":
        avg = numpy.mean(train_values)
        std_dev = numpy.std(train_values)
        if std_dev == 0:
            return data

        return {key: (value - avg) / std_dev for key, value in data.items()}

    if normalization_type == "min-max":
        min_val = numpy.min(train_values)
        max_val = numpy.max(train_values)
        value_range = max_val - min_val

        if value_range == 0:
            return data

        return {key: (value - min_val) / value_range for key, value in data.items()}

    raise Exception("Unsupported normalization type!")


def print_args(args):
    args_dict = vars(args)
    if not args_dict:
        print("No arguments provided.")
        return

    max_key_len = max(len(key) for key in args_dict)
    print("\n" + "=" * 30)
    print(f"{'CONFIGURATION':^{30}}")
    print("=" * 30)
    for key, value in sorted(args_dict.items()):
        print(f"{key:<{max_key_len}} : {value}")

    print("=" * 30 + "\n")


def print_data_analysys(data):
    print("Average:", data.mean().item())
    print("Standard deviation:", data.std().item())
    print("Minimum:", data.min().item())
    print("Maximum:", data.max().item())


def print_ranking_comparison(ranking_a, ranking_b):
    ranking_a_sorted_indexes = torch.tensor(sorted(ranking_a.keys(), key=lambda x: ranking_a[x], reverse=True)).flatten().cpu().tolist()
    ranking_b_sorted_indexes = torch.tensor(sorted(ranking_b.keys(), key=lambda x: ranking_b[x], reverse=True)).flatten().cpu().tolist()
    ranking_a = [ranking_a[x] for x in ranking_a.keys()]
    ranking_b = [ranking_b[x] for x in ranking_b.keys()]
    cprint(f"Spearman's rho: {spearmanr(ranking_a, ranking_b)[0]}", Color.EXPERIMENT_OUTPUT)
    cprint(f"Kendall's tau: {kendalltau(ranking_a, ranking_b)[0]}", Color.EXPERIMENT_OUTPUT)
    for value in [100, 250, 500]:
        cprint("-" * 30, Color.EXPERIMENT_OUTPUT)
        cprint(f"Jaccard Index @{value}: {jaccard_index(ranking_a_sorted_indexes, ranking_b_sorted_indexes, value)}", Color.EXPERIMENT_OUTPUT)
        cprint(f"Precision@{value}: {precision_at_k(ranking_a_sorted_indexes, ranking_b_sorted_indexes, value)}", Color.EXPERIMENT_OUTPUT)
        cprint(f"NDCG@{value}: {ndcg(ranking_a_sorted_indexes, ranking_b_sorted_indexes, value)}", Color.EXPERIMENT_OUTPUT)


def plot_data_distribution(values, label):
    values = values.flatten()
    indices = torch.argsort(values)
    sorted_values = values[indices]
    plt.figure(figsize=(12, 6))
    x = numpy.arange(len(values))
    width = 0.35
    plt.bar(x - width / 2, sorted_values.detach().cpu().numpy(), width, label="Values", color="skyblue")
    plt.ylabel(f"Values of {label}")
    plt.xlabel("Index")
    plt.xticks(x)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_ranking_comparison(real_scores, predicted_scores, labels):
    real_scores = real_scores.flatten()
    predicted_scores = predicted_scores.flatten()
    indices = torch.argsort(real_scores)
    real_scores_sorted = real_scores[indices]
    predicted_scores_reordered = predicted_scores[indices]
    plt.figure(figsize=(12, 6))
    x = numpy.arange(len(real_scores))
    width = 0.35
    plt.bar(x - width / 2, real_scores_sorted.detach().cpu().numpy(), width, label=labels[0], color="skyblue")
    plt.bar(x + width / 2, predicted_scores_reordered.detach().cpu().numpy(), width, label=labels[1], color="orange")
    plt.ylabel("Value")
    plt.xlabel("Index (sorted by real scores)")
    plt.xticks(x)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def seed_everything(seed):
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def hetero_dropout(data, p=0.5, training=True):
    for edge_type, store in data.edge_items():
        new_index, _ = dropout_edge(store.edge_index, p=p, training=training)
        data[edge_type].edge_index = new_index
    return data


def apply_pca(data, n_components=128):
    for node_type in data.x_dict:
        x_numpy = data.x_dict[node_type].cpu().numpy()
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(x_numpy)
        data[node_type].x = torch.from_numpy(x_pca).float().to(data.device)

    return data


def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True

    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False

    else:
        raise Exception("Boolean value expected.")


def stratified_sampling(proxy_scores, seed_mask, n_samples_in_head, n_levels):
    sorted_indexes = sorted(seed_mask.cpu().tolist(), key=lambda i: proxy_scores[list(proxy_scores.keys())[i]])
    level_size = len(sorted_indexes) // n_levels
    k_per_level = n_samples_in_head // n_levels

    selected_indexes = []
    for level in range(0, n_levels):
        selected_indexes.extend(random.sample(sorted_indexes[(level * level_size): ((level + 1) * level_size)], k_per_level))

    selected_indexes = torch.tensor(selected_indexes, device=seed_mask.device)
    mask = torch.isin(seed_mask, selected_indexes)
    remaining_indexes = seed_mask[~mask]
    return torch.cat([selected_indexes, remaining_indexes], dim=0)


def get_topk_centrality_nodes(g, user_id_map, centrality_measure):
    if centrality_measure == "degree":
        centrality = nx.degree_centrality(g)

    elif centrality_measure == "closeness":
        centrality = nx.closeness_centrality(g)

    elif centrality_measure == "betweenness":
        centrality = nx.betweenness_centrality(g)

    elif centrality_measure == "eigenvector":
        centrality = nx.eigenvector_centrality(g, max_iter=500)

    elif centrality_measure == "pagerank":
        centrality = nx.pagerank(g)

    elif centrality_measure == "katz":
        centrality = nx.katz_centrality(g)

    elif centrality_measure == "hits":
        _, centrality = nx.hits(g)

    else:
        raise ValueError(f"Unknown centrality measure: '{centrality_measure}'")

    user_nodes = {user_id: centrality[node] for node, user_id in user_id_map.items()}
    return user_nodes


def evaluate_centrality_measure(centrality_scores, data, mask=None, connectivity_bound=20):
    report = {}
    if mask is None:
        ground_truth = data[data.target_type].y.cpu().numpy()
        preds_raw = numpy.array([centrality_scores[idx] for idx in centrality_scores.keys()])

    else:
        mask_indices = mask.squeeze().nonzero().cpu().flatten().tolist()
        ground_truth = data[data.target_type].y[mask.squeeze().bool()].cpu().numpy()
        preds_raw = numpy.array([centrality_scores[idx] for idx in mask_indices])

    ground_truth_norm = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min())
    preds_norm = (preds_raw - preds_raw.min()) / (preds_raw.max() - preds_raw.min())

    def compute_metrics(true, pred):
        mae = mean_absolute_error(true, pred)
        mse = mean_squared_error(true, pred)
        rmse = numpy.sqrt(mse)
        r2 = r2_score(true, pred)
        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    report["regression_metrics"] = compute_metrics(ground_truth_norm, preds_norm)

    if connectivity_bound > 0:
        all_sorted_indices = sorted(centrality_scores.keys(), key=lambda x: centrality_scores[x], reverse=True)
        con_values = compute_incremental_con_measure(all_sorted_indices, connectivity_bound, data, False)
        report["con_measures"] = {"values": con_values}

    return report
