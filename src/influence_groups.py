import os
import json
import numpy as np
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt

from src.utils import get_base_dir


def ic_percentile_thresholds(ic_scores, low_q=50, high_q=85):

    scores = np.array(list(ic_scores.values()), dtype=float)

    low_thr = np.percentile(scores, low_q)
    high_thr = np.percentile(scores, high_q)

    return low_thr, high_thr


""" data: dict[node_id -> score] """
def compute_influence_groups_from_dict(data, low_thr, high_thr):

    data = {int(k): float(v) for k, v in data.items()} # for numeric node ids

    #low_thr = np.percentile(scores, 33)
    #high_thr = np.percentile(scores, 66)

    groups = {
        "low_influence_nodes": set(),
        "medium_influence_nodes": set(),
        "high_influence_nodes": set()
    }

    for node, score in data.items():
        if score <= low_thr:
            groups["low_influence_nodes"].add(node)
        elif score <= high_thr:
            groups["medium_influence_nodes"].add(node)
        else:
            groups["high_influence_nodes"].add(node)

    return groups

# How many nodes are assigned to a different influence group? 0.0 perfect agreement, >0.4 poor agreement
def group_disagreement_percentage(groups1, groups2):
    all_nodes = set().union(*groups1.values())

    label1 = {}
    label2 = {}

    for g, nodes in groups1.items():
        for n in nodes:
            label1[n] = g

    for g, nodes in groups2.items():
        for n in nodes:
            label2[n] = g

    diff = sum(label1[n] != label2[n] for n in all_nodes)

    return diff / len(all_nodes)


def jaccard(a, b):
    return len(a & b) / len(a | b)

# 1.0 identical groups, >0.7 very similar, <0.5 weak similarity
def group_jaccard_similarity(groups1, groups2):
    return {
        g: jaccard(groups1[g], groups2[g])
        for g in groups1
    }

""" scores1, scores2: dict[node -> score] """
def ranking_comparison(scores1, scores2):

    common_nodes = sorted(set(scores1) & set(scores2))

    v1 = [scores1[n] for n in common_nodes]
    v2 = [scores2[n] for n in common_nodes]

    spearman = spearmanr(v1, v2).correlation
    kendall = kendalltau(v1, v2).correlation

    return spearman, kendall

def rank_normalize(scores: dict):
    """
    Convert scores to percentile ranks in [0, 1].
    """
    nodes = list(scores.keys())
    values = np.array(list(scores.values()))

    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values))

    ranks /= (len(values) - 1)

    return dict(zip(nodes, ranks))



def compute_and_plot_influence_groups(json_path, low_threshold, high_threshold):

    with open(json_path, "r") as f:
        data = json.load(f)

    node_ids = list(data.keys())
    scores = np.array(list(data.values()))

    groups = {
        "low_influence_nodes": [],
        "medium_influence_nodes": [],
        "high_influence_nodes": []
    }

    for node_id, score in data.items():
        if score <= low_threshold:
            groups["low_influence_nodes"].append(node_id)
        elif score <= high_threshold:
            groups["medium_influence_nodes"].append(node_id)
        else:
            groups["high_influence_nodes"].append(node_id)

    plt.figure(figsize=(10, 5))

    #plot_scatter
    """
    plt.plot(scores, marker='o', linestyle='', alpha=0.7, label="Influence score")

    plt.axhline(low_threshold, color='orange', linestyle='--', label=f"Low threshold ({low_threshold:.2f})")
    plt.axhline(high_threshold, color='red', linestyle='--', label=f"High threshold ({high_threshold:.2f})")

    plt.xlabel("Node index")
    plt.ylabel("Influence score")
    plt.title("Distribution of Influence Scores")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    """

    #plot histogram
    plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')

    plt.axvline(low_threshold, color='orange', linestyle='--',
                label=f"Low threshold ({low_threshold:.2f})")
    plt.axvline(high_threshold, color='red', linestyle='--',
                label=f"High threshold ({high_threshold:.2f})")

    plt.xlabel("Influence score")
    plt.ylabel("Number of nodes")
    plt.title("Histogram of Influence Scores")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    plt.show()


def compare_scores_and_return_groups(ic_scores, proxy_scores, output_dir):

    # convert keys
    ic_scores = {int(k): float(v) for k, v in ic_scores.items()}
    proxy_scores = {int(k): float(v) for k, v in proxy_scores.items()}

    output_file = os.path.join(output_dir, "comparison_ic_vs_proxy.txt")

    with open(output_file, "w") as f:

        # 1) ranking comparison
        rho, tau = ranking_comparison(ic_scores, proxy_scores)
        f.write(f"Spearman rho: {rho}\n")
        f.write(f"Kendall tau: {tau}\n\n")

        # normalization
        ic_scores_norm = rank_normalize(ic_scores)
        proxy_scores_norm = rank_normalize(proxy_scores)

        # 2) group comparison
        low_thr, high_thr = ic_percentile_thresholds(ic_scores_norm, 50, 85)
        groups_ic = compute_influence_groups_from_dict(ic_scores_norm, low_thr, high_thr)
        groups_proxy = compute_influence_groups_from_dict(proxy_scores_norm, low_thr, high_thr)

        disagreement = group_disagreement_percentage(groups_ic, groups_proxy)
        f.write(f"Disagreement %: {disagreement}\n\n")

        f.write("Jaccard similarity:\n")
        jaccard = group_jaccard_similarity(groups_ic, groups_proxy)
        f.write(f"{jaccard}\n")

        return groups_ic, groups_proxy



if __name__ == '__main__':
    dataset_name = "imdb"
    out_dir = os.path.join(get_base_dir(), dataset_name, "influential_nodes", "results")

    with open(os.path.join(out_dir, "influence_scores_3steps.json")) as f:
        ic_scores = json.load(f)

    with open(os.path.join(out_dir, "influence_scores_IC_proxy.json")) as f:
        proxy_scores = json.load(f)

    compare_scores_and_return_groups(ic_scores, proxy_scores, out_dir)





