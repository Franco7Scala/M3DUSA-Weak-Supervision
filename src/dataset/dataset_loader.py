import os
import torch
import pandas as pd
import numpy as np

from torch_geometric.data.hetero_data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.transforms import AddMetaPaths

from src.support.utils import get_base_dir, load_embeddings


def get_target_type(dataset_name):
    if dataset_name == "mumin":
        return "claim"
    if dataset_name == "politifact":
        return "news"
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

# EARLY FUSION
def build_heterodata(dataset_name, target_type): # "politifact", "mumin"
    data = HeteroData()
    heterodata_dir = os.path.join(get_base_dir(), dataset_name, "heterodata")
    feats_dir = os.path.join(heterodata_dir, "features")
    edges_dir = os.path.join(heterodata_dir, "edgelists")

    # nodes
    files_feats = [f for f in os.listdir(feats_dir) if os.path.isfile(os.path.join(feats_dir, f))]
    for fname in files_feats:
        n_type = fname[:-3]  # remove the last 4 characters (".pt")
        data[n_type].x = torch.load(os.path.join(feats_dir, fname))

    # ground_truth for target_type: (ground truth, surrogate ground truth)
    gt = torch.load(os.path.join(heterodata_dir, f"{target_type}_labels.pt"))
    surrogate_gt = torch.load(os.path.join(heterodata_dir, f"{target_type}_labels_surrogate.pt"))
    confidence_surrogate = torch.load(os.path.join(heterodata_dir, "confidence_surrogate.pt"))
    data[target_type].y = {}
    data[target_type].y["ground_truth"] = gt
    data[target_type].y["ground_truth_surrogate"] = surrogate_gt
    data[target_type].y["confidence_surrogate"] = confidence_surrogate

    # edges
    files_edges = [f for f in os.listdir(edges_dir) if os.path.isfile(os.path.join(edges_dir, f))]
    for fname in files_edges:
        n_type_src, n_type_tgt, e_type = _extract_edge_info(fname)
        edge_index = torch.load(os.path.join(heterodata_dir, "edgelists", fname))
        if edge_index.dtype == torch.float64:
            edge_index = edge_index.to(torch.int64)
        data[n_type_src, e_type, n_type_tgt].edge_index = edge_index

    # metapaths
    metapaths = _get_metapaths(dataset_name)
    data = AddMetaPaths(metapaths, weighted=True)(data)

    transform = T.RandomNodeSplit(num_val=0.15, num_test=0.25)  # train-val-test split: 60-15-25
    data = transform(data)

    """
    indices_masks = load_from_pickle(os.path.join(base_dir, f'masks_indices_{split}.pkl'))
    mask_dim = data[target_type].x.shape[0]
    data[target_type].train_mask = index_to_mask(indices_masks[seed]["train"], mask_dim)
    data[target_type].val_mask = index_to_mask(indices_masks[seed]["val"], mask_dim)
    data[target_type].test_mask = index_to_mask(indices_masks[seed]["test"], mask_dim)
    """

    data.embeddings = load_embeddings(dataset_name, target_type)

    return data


def _extract_edge_info(fname):
    last_dot = fname.rfind(".")
    fname_base = fname[:last_dot]
    first_underscore = fname_base.find("_")  # first occurrence
    last_underscore = fname_base.rfind("_")  # last occurrence
    n_type_src = fname_base[:first_underscore]
    e_type = fname_base[first_underscore + 1:last_underscore]
    n_type_tgt = fname_base[last_underscore + 1:]
    return n_type_src, n_type_tgt, e_type


def _get_metapaths(dataset_name):
    if dataset_name.lower() == "politifact":
        metapaths = [[('user', 'posted', 'tweet'),
                      ('tweet', 'is_retweeted_by', 'user')],  # UTU
                     [('user', 'posted', 'tweet'),
                      ('tweet', 'has_hashtag', 'hashtag'),
                      ('hashtag', 'is_hashtag_of', 'tweet'),
                      ('tweet', 'is_posted_by', 'user')]]  # UTHTU

    if dataset_name.lower() == "mumin":
        metapaths = [
            [('claim', 'is_discussed_by', 'tweet'),
                      ('tweet', 'is_posted_by', 'user'),
                      ('user', 'posted', 'tweet'),
                      ('tweet', 'discusses', 'claim')],  # CTUTC
                     [('claim', 'is_discussed_by', 'tweet'),
                      ('tweet', 'has_hashtag', 'hashtag'),
                      ('hashtag', 'is_hashtag_of', 'tweet'),
                      ('tweet', 'discusses', 'claim')],  # CTHTC
                     [('claim', 'is_discussed_by', 'tweet'),
                      ('tweet', 'is_replied_by', 'reply'),
                      ('reply', 'reply_to', 'tweet'),
                      ('tweet', 'discusses', 'claim')],  # CTRTC_r
                     [('claim', 'is_discussed_by', 'tweet'),
                      ('tweet', 'is_quoted_by', 'reply'),
                      ('reply', 'quote_of', 'tweet'),
                      ('tweet', 'discusses', 'claim')]  # CTRTC_q
        ]


    else: #politifact
        metapaths = [
            [('news', 'is_discussed_by', 'tweet'),
             ('tweet', 'is_posted_by', 'user'),
             ('user', 'posted', 'tweet'),
             ('tweet', 'discusses', 'news')],  # NTUTN
            [('news', 'is_discussed_by', 'tweet'),
             ('tweet', 'has_hashtag', 'hashtag'),
             ('hashtag', 'is_hashtag_of', 'tweet'),
             ('tweet', 'discusses', 'news')],  # NTHTN
            [('news', 'is_discussed_by', 'tweet'),
             ('tweet', 'is_posted_by', 'user'),
             ('user', 'mentions', 'user'),
             ('user', 'posted', 'tweet'),
             ('tweet', 'discusses', 'news')]  # NTUUTN
        ]

    return metapaths


if __name__ == '__main__':
    data = build_heterodata("politifact", "news")
    print(data)

