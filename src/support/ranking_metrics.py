import torch
import numpy as np

from scipy.stats import spearmanr, kendalltau


def jaccard_index(list1, list2, k):
    set1, set2 = set(list1[:k]), set(list2[:k])
    return len(set1 & set2) / len(set1 | set2)


def precision_at_k(list1, list2, k):
    set1, set2 = set(list1[:k]), set(list2[:k])
    return len(set1 & set2) / k


def dcg_at_k(ranking, ideal_ranking, k):
    dcg = sum((1 / np.log2(i + 2)) for i in ranking[:k])
    idcg = sum((1 / np.log2(i + 2)) for i in ideal_ranking[:k])
    return dcg / idcg if idcg > 0 else 0


def ndcg(list1, list2, k):
    ranking = [list2.index(x) if x in list2 else len(list2) for x in list1]
    ideal_ranking = sorted(ranking)
    return dcg_at_k(ranking, ideal_ranking, k)


def coverage(data, influencer_nodes, max_steps=3):
    influenced_nodes = _count_influenced_nodes(data, influencer_nodes, max_steps)
    return (influenced_nodes + len(influencer_nodes)) / data.num_nodes


def reachability(data, influencer_nodes, max_steps=3):
    influenced_nodes = _count_influenced_nodes(data, influencer_nodes, max_steps)
    return influenced_nodes / (data.num_nodes - len(influencer_nodes))


def _count_influenced_nodes(data, influencer_nodes, max_steps):
    influencer_nodes = set(influencer_nodes)
    user_edges = []
    for edge_type in data.edge_types:
        if edge_type[0] == data.target_type and edge_type[2] == data.target_type:
            user_edges.append(data[edge_type].edge_index)

    if not user_edges:
        return 0

    edge_index = torch.cat(user_edges, dim=1)
    influenced_users = set()
    frontier = list(influencer_nodes)
    visited = set(frontier)

    for _ in range(max_steps):
        next_frontier = set()
        for user in frontier:
            mask = edge_index[0] == user
            neighbors = edge_index[1][mask].tolist()
            for neighbor in neighbors:
                if neighbor not in visited:
                    next_frontier.add(neighbor)
                    visited.add(neighbor)

        if not next_frontier:
            break

        frontier = list(next_frontier)
        influenced_users.update(next_frontier - influencer_nodes)

    return len(influenced_users)
