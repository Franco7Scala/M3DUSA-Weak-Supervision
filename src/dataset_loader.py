import os
import torch
import torch.nn as nn

from torch_geometric.datasets import IMDB, AMiner
from torch_geometric.datasets.dblp import DBLP
import torch_geometric.transforms as T
from torch_geometric.transforms import AddMetaPaths
from src.utils import get_base_dir, build_heterodata
from src.utils import get_metapaths

import warnings

from src.utils_graph import k_hop_subgraph

warnings.filterwarnings("ignore")


def load_dataset(dataset_name, reduction_factor, k, device):
    path = os.path.join(get_base_dir(), dataset_name)
    snapshot_masks = []

    if dataset_name.lower() == "imdb".lower():
        target_type = "actor"
        dataset = IMDB(path)
        data = dataset.data

    elif dataset_name.lower() == "aminer".lower():
        target_type = "author"
        dataset = AMiner(path)
        data = dataset.data

    elif dataset_name.lower() == "dblp".lower():
        target_type = "author"
        dataset = DBLP(path)
        data = dataset.data

    elif dataset_name.lower() == "politifact".lower() or dataset_name.lower() == "mumin".lower():
        target_type = "user"
        data = build_heterodata(dataset_name, target_type)

    else:
        raise Exception(f"Unknown dataset '{dataset_name}'!")

    in_dim = 128
    #embeddings = nn.ModuleDict()
    for ntype in data.metadata()[0]:  # metadata()[0] returns node types list
        if 'x' not in data[ntype]:

            num_nodes = data[ntype].num_nodes
            data[ntype].x = torch.zeros((num_nodes, in_dim), device=device)
            #data[ntype].x = nn.Embedding(num_nodes, in_dim)(torch.arange(num_nodes)).detach()

    metapaths = get_metapaths(dataset_name)
    data.mps = metapaths

    data.target_type = target_type
    #data.to(device)
    #data.device = data.x_dict[data.target_type].device

    if reduction_factor == 1:
        data = AddMetaPaths(metapaths=metapaths, weighted=True)(data)
        data.to(device)
        data.device = data.x_dict[data.target_type].device
        return data

    data.to(device)
    data.device = data.x_dict[data.target_type].device

    size = data[target_type].x.shape[0]
    for i in range(reduction_factor):
        mask = torch.arange(0, size)
        mask = mask[int(i * size / reduction_factor): int((i + 1) * size / reduction_factor)]
        snapshot_masks.append(k_hop_subgraph(data, mask, k)[0])

    return AddMetaPaths(metapaths=metapaths, weighted=True)(snapshot_masks[0])

    #return data, target_type, snapshot_masks
