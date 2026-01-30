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


def rbo(list1, list2, p=0.9):
    k = min(len(list1), len(list2))
    overlap = 0.0
    agreement = 0.0
    for d in range(k):
        overlap += len(set(list1[:d+1]) & set(list2[:d+1])) / (d + 1)
        agreement += (p ** d) * overlap
    return (1 - p) * agreement


# Example Usage
list1 = ['A', 'B', 'C', 'D', 'E']
list2 = ['B', 'A', 'D', 'C', 'E']

print("Spearman's rho:", spearmanr(range(len(list1)), [list2.index(x) for x in list1])[0])
print("Kendall's tau:", kendalltau(range(len(list1)), [list2.index(x) for x in list1])[0])
print("Jaccard Index @3:", jaccard_index(list1, list2, 3))
print("Precision@3:", precision_at_k(list1, list2, 3))
print("NDCG@3:", ndcg(list1, list2, 3))
print("RBO:", rbo(list1, list2))
