import networkx as nx
from collections import defaultdict
from network_diffusion.mln.functions import core_number #, degree


# Influence(v) = core_number(v) × H_index(v)
# Influence(v) = Σ_l p_l · core_l(v) · H_l(v)


def h_index_single_layer(G):
    deg = dict(G.degree())
    h = {}

    for v in G.nodes():
        neigh_degs = sorted((deg[u] for u in G.neighbors(v)), reverse=True)
        hv = 0
        for i, d in enumerate(neigh_degs, start=1):
            if d >= i:
                hv = i
            else:
                break
        h[v] = hv

    return h


""" Normalize scores to [0, 1] by max value. """
def normalize(scores):

    m = max(scores.values())
    if m == 0:
        return scores

    return {k: v / m for k, v in scores.items()}



"""
Compute IC-like influence: Influence(v) = Σ_l p_l · core_l(v) · H_l(v)
multilayer_graph : dict[layer, nx.Graph]
layer_prob       : dict[layer, float]
"""
def compute_ic_like_influence_scores(multilayer_graph, layer_prob, normalize=False):

    final_score = defaultdict(float)

    for layer, G in multilayer_graph.items():
        p = layer_prob.get(layer, 1.0)

        # core number on this layer
        core = nx.core_number(G)

        # H-index on this layer
        h = h_index_single_layer(G)

        if normalize:
            core = normalize(core)
            h = normalize(h)

        for v in G.nodes():
            final_score[v] += p * core[v] * h[v]

    return dict(final_score)

