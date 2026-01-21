import os
import torch
import csv
import json
import time
import numpy
import copy
import torch_geometric.transforms as T

from enum import Enum
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData


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
    return '/home/jovyan/projects/InfluentialNodes/dataset/'
    #return "/home/scala/projects/InfluentialNodes/dataset/"
    #return '/home/martirano/data'


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
