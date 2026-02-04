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
