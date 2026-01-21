import networkx as nx
import torch

from collections import defaultdict


"""
Constructs weighted homogeneous graphs for each metapath and edge that starts and ends in data.target_type.

Returns:
    dictionary of networkx.Graph
"""
def build_metapath_graphs(data):

    target = data.target_type
    graphs = {}

    for (src, etype, dst) in data.edge_types:

        # --- Case 1: Metapath relations ---
        if etype.startswith("metapath_"):

            if src != target or dst != target:
                continue

            metapath = data.metapath_dict[(src, etype, dst)]

            # adjacency step graphs for each hop
            adj = []
            for (t_src, rel, t_dst) in metapath:
                ei = data.edge_index_dict[(t_src, rel, t_dst)].t().tolist()
                neighbors = defaultdict(list)
                for u, v in ei:
                    neighbors[u].append(v)
                adj.append(neighbors)

            # Build weighted graph G
            G = nx.Graph()
            num_target_nodes = data.num_nodes_dict[target]
            G.add_nodes_from(range(num_target_nodes))

            for start in range(num_target_nodes):
                frontier = {start: 1}

                for step_neighbors in adj:
                    new_frontier = defaultdict(int)
                    for u, c in frontier.items():
                        for v in step_neighbors.get(u, []):
                            new_frontier[v] += c
                    frontier = new_frontier

                for end, w in frontier.items():
                    if w > 0 and end != start:
                        oldw = G[start][end]["weight"] if G.has_edge(start, end) else 0
                        G.add_edge(start, end, weight=oldw + w)

            key = metapath_key(metapath)
            graphs[key] = G

        # --- Case 2: Direct target–target relations ---
        elif src == target and dst == target:

            if etype.startswith("is_"):
                continue

            G = nx.Graph()
            num_target_nodes = data.num_nodes_dict[target]
            G.add_nodes_from(range(num_target_nodes))

            # fetch edge index for this relation
            ei = data.edge_index_dict[(src, etype, dst)].t().tolist()

            for u, v in ei:
                if u == v:
                    continue
                if G.has_edge(u, v):
                    G[u][v]["weight"] += 1
                else:
                    G.add_edge(u, v, weight=1)

            isolated = list(nx.isolates(G))
            G.remove_nodes_from(isolated)

            edge = (src, etype, dst)
            key = edge_key(edge)
            graphs[key] = G

    return graphs


def compute_layer_probabilities(graphs, beta):
    probs = {}
    for mp in graphs:
        len_mp = (len(mp)-1)-1
        probs[mp] = beta ** len_mp
    return probs


"""
Returns:
 - neighbor_probs[layer][u] = [(v, p_eff), ...] where p_eff = layer_prob * edge_weight
 - nodes_in_layer[layer] = set(nodes)
"""
def precompute_neighbor_probs(layer_dict, layer_probs):

    neighbor_probs = {}
    nodes_in_layer = {}

    for layer_name, G in layer_dict.items():
        neighbor_probs[layer_name] = {}
        nodes_in_layer[layer_name] = set(G.nodes())
        p_layer = layer_probs.get(layer_name, 0.0)

        # iterate all edges (u,v)
        # store for each source u: list of (v, p_eff)
        for u, v, ed in G.edges(data=True):
            weight = ed.get("weight", 1.0)
            #p_eff = p_layer * weight
            p_eff = 1 - (1 - p_layer)**weight
            if p_eff > 1:
                p_eff = 1.0

            neighbor_probs[layer_name].setdefault(u, []).append((v, p_eff))

            # If undirected graph, also add reverse (v->u)
            if not getattr(G, "is_directed", lambda: False)():
                neighbor_probs[layer_name].setdefault(v, []).append((u, p_eff))
        # ensure nodes present but with no outgoing edges have an empty list
        for n in G.nodes():
            neighbor_probs[layer_name].setdefault(n, [])

    return neighbor_probs, nodes_in_layer


def create_nodes_dict_empty(data):
    res = {}
    for node_type in data.node_types:
        res[node_type] = []
    return res


def k_hop_subgraph(data, seeds_mask, k):
    subgraph_mask = create_nodes_dict_empty(data)
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        if dst_type == data.target_type:
            for idx, node in enumerate(data[edge_type]["edge_index"][1]):
                if node.item() in seeds_mask:
                    subgraph_mask[src_type].append(data[edge_type]["edge_index"][0][idx].item())

        if src_type == data.target_type:
            for idx, node in enumerate(data[edge_type]["edge_index"][0]):
                if node.item() in seeds_mask:
                    subgraph_mask[dst_type].append(data[edge_type]["edge_index"][1][idx].item())

    #TODO add meta paths for heterogeneous graphs
    subgraph_mask[data.target_type] = seeds_mask.tolist()
    subgraph_mask = _merge_masks(subgraph_mask, _hop_traveling(data, subgraph_mask, k-1))
    for ntype in subgraph_mask.keys():
        subgraph_mask[ntype] = torch.tensor(subgraph_mask[ntype]).to(data.device)

    subgraph_data = data.subgraph(subgraph_mask)
    return subgraph_data, subgraph_mask


def _hop_traveling(data, subgraph_mask, k):
    if k == 0:
        return subgraph_mask

    next_hop_subgraph_mask = create_nodes_dict_empty(data)
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        for node_type in subgraph_mask.keys():
            if node_type != data.target_type:
                prevoius_hop_nodes = subgraph_mask[node_type]
                if dst_type == node_type and src_type != data.target_type:
                    for idx, node in enumerate(data[edge_type]["edge_index"][1]):
                        if node.item() in prevoius_hop_nodes:
                            next_hop_subgraph_mask[src_type].append(data[edge_type]["edge_index"][0][idx].item())

                if src_type == node_type and dst_type != data.target_type:
                    for idx, node in enumerate(data[edge_type]["edge_index"][0]):
                        if node.item() in prevoius_hop_nodes:
                            next_hop_subgraph_mask[dst_type].append(data[edge_type]["edge_index"][1][idx].item())

    return _hop_traveling(data, next_hop_subgraph_mask, k-1)


def _merge_masks(first_mask, second_mask):
    merged_mask = {}
    for ntype in first_mask.keys():
        merged_mask[ntype] = []

    for ntype in second_mask.keys():
        merged_mask[ntype] = []

    for ntype in merged_mask.keys():
        mask_1 = first_mask[ntype] if ntype in first_mask else []
        mask_2 = second_mask[ntype] if ntype in second_mask else []
        merged_mask[ntype] = list(set(mask_1 + mask_2))

    return merged_mask


def metapath_key(metapath): #metapath: list of (src, rel, dst)

    node_types = []
    for i, (src, _, dst) in enumerate(metapath):
        if i == 0:
            node_types.append(src)
        node_types.append(dst)
    initials = [nt[0].upper() for nt in node_types]
    # Compress consecutive duplicates
    key = []
    for ch in initials:
        if not key or key[-1] != ch:
            key.append(ch)
    return "".join(key)


def edge_key(edge): #edge: tuple (src, rel, dst)

    src, rel, dst = edge
    return f"{src[0].upper()}{dst[0].upper()}_{rel[0].lower()}"
