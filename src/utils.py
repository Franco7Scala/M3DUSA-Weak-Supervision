import os
import torch
import csv
import json
import time
import numpy
import copy
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")     #"cpu"#


def get_base_dir():
    return '/home/martirano/data'

def get_time_in_millis():
    return int(round(time.time() * 1000))


def get_metapaths(dataset_name):

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
                      ('paper', 'to', 'author')], #APCPA
                     [('author', 'to', 'paper'),
                      ('paper', 'to', 'term'),
                      ('term', 'to', 'paper'),
                      ('paper', 'to', 'author')]] #APTPA

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


def build_heterodata(dataset_name, target_type):
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

def _extract_edge_info(fname):
    #fname_base = fname[:-3]  # remove the last 3 characters (".pt") #if .csv?
    last_dot = fname.rfind(".")
    fname_base = fname[:last_dot]
    first_underscore = fname_base.find("_")  # first occurrence
    last_underscore = fname_base.rfind("_")  # last occurrence
    n_type_src = fname_base[:first_underscore]
    e_type = fname_base[first_underscore + 1:last_underscore]
    n_type_tgt = fname_base[last_underscore + 1:]
    return n_type_src, n_type_tgt, e_type



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
