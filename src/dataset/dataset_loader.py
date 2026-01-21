import os
import torch

from torch_geometric.datasets import IMDB, AMiner
from torch_geometric.datasets.dblp import DBLP
import torch_geometric.transforms as T
from torch_geometric.transforms import AddMetaPaths
from src.support.utils import get_base_dir
from src.support.utils_graph import k_hop_subgraph


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
        data = _build_heterodata(dataset_name, target_type)

    else:
        raise Exception(f"Unknown dataset '{dataset_name}'!")

    in_dim = 128
    #embeddings = nn.ModuleDict()
    for ntype in data.metadata()[0]:  # metadata()[0] returns node types list
        if 'x' not in data[ntype]:

            num_nodes = data[ntype].num_nodes
            data[ntype].x = torch.zeros((num_nodes, in_dim), device=device)
            #data[ntype].x = nn.Embedding(num_nodes, in_dim)(torch.arange(num_nodes)).detach()

    metapaths = _get_metapaths(dataset_name)
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


def _get_metapaths(dataset_name):
    if dataset_name.lower() == "imdb":
        metapaths = [[('actor', 'to', 'movie'),
                      ('movie', 'to', 'actor')], # AMA
                     [('actor', 'to', 'movie'),
                     ('movie', 'to', 'director'),
                      ('director', 'to', 'movie'),
                      ('movie', 'to', 'actor')]] #AMDMA

    elif dataset_name.lower() == "aminer":
        metapaths = [[('author', 'writes', 'paper'),
                      ('paper', 'written_by', 'author')],  # APA
                     [('author', 'writes', 'paper'),
                      ('paper', 'published_in', 'venue'),
                      ('venue', 'publishes', 'paper'),
                      ('paper', 'written_by', 'author')]]  # APVPA

    elif dataset_name.lower() == "dblp":
        metapaths = [[('author', 'to', 'paper'),
                      ('paper', 'to', 'author')], # APA
                     [('author', 'to', 'paper'),
                      ('paper', 'to', 'conference'),
                      ('conference', 'to', 'paper'),
                      ('paper', 'to', 'author')]] #APCPA
        """
                     [('author', 'to', 'paper'),
                      ('paper', 'to', 'term'),
                      ('term', 'to', 'paper'),
                      ('paper', 'to', 'author')] """ #APTPA

    elif dataset_name.lower() == "politifact":
        metapaths = [[('user', 'posted', 'tweet'),
                      ('tweet', 'is_retweeted_by', 'user')],  # UTU
                     [('user', 'posted', 'tweet'),
                      ('tweet', 'has_hashtag', 'hashtag'),
                      ('hashtag', 'is_hashtag_of', 'tweet'),
                      ('tweet', 'is_posted_by', 'user')]]  # UTHTU

    elif dataset_name.lower() == "mumin":
        metapaths = [[('user', 'posted', 'tweet'),
                      ('tweet', 'is_retweeted_by', 'user')], #UTU (uno scrive, l'altro retweet)
                     [('user', 'posted', 'tweet'),
                      ('tweet', 'is_quoted_by', 'reply'),
                      ('reply', 'is_posted_by', 'user')], ##UTRU (uno scrive, l'altro risponde)
                     [('user', 'posted', 'reply'),
                     ('reply', 'is_posted_by', 'user')],  # URU (entrambi rispondono)
                     [('user', 'posted', 'tweet'),
                      ('tweet', 'has_hashtag', 'hashtag'),
                      ('hashtag', 'is_hashtag_of', 'tweet'),
                      ('tweet', 'is_posted_by', 'user')]]  # UTHTU

    else:
        raise Exception("no metapaths defined for this dataset! Cretina!")

    return metapaths


def _build_heterodata(dataset_name, target_type):
    data = HeteroData()
    heterodata_dir = os.path.join(get_base_dir(), dataset_name, "heterodata")
    feats_dir = os.path.join(heterodata_dir, "features")
    edges_dir = os.path.join(heterodata_dir, "edgelists")

    # nodes
    files_feats = [f for f in os.listdir(feats_dir) if os.path.isfile(os.path.join(feats_dir, f))]
    for fname in files_feats:
        n_type = fname[:-3]  # remove the last 4 characters (".pt")
        data[n_type].x = torch.load(os.path.join(feats_dir, fname))

    # ground truth for target_type
    #data[target_type].y = torch.load(os.path.join(heterodata_dir, f'{target_type}_labels.pt'))

    # edges
    files_edges = [f for f in os.listdir(edges_dir) if os.path.isfile(os.path.join(edges_dir, f))]
    for fname in files_edges:
        n_type_src, n_type_tgt, e_type = _extract_edge_info(fname)
        edge_index = torch.load(os.path.join(heterodata_dir, "edgelists", fname))
        if edge_index.dtype == torch.float64:
            edge_index = edge_index.to(torch.int64)
        data[n_type_src, e_type, n_type_tgt].edge_index = edge_index

    transform = T.RandomNodeSplit(num_val=0, num_test=0.30)  # train-val-test split: 70-0-30
    data = transform(data)

    return data
