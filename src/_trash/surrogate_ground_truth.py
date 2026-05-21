import os
import pandas as pd
import torch

from src.utils import get_base_dir


def load_texts():
    df = pd.read_csv(os.path.join(get_base_dir(), "politifact", "news_texts_with_label.csv"))
    df_new = df[["text"]]
    return df_new

def test_mumin_original():
    df = pd.read_parquet(os.path.join(get_base_dir(), "mumin", "claim_LP_split_60-15-25_with_embeddings.parquet"))
    df_new = df[["embedding", "label", "reviewers", "date", "language", "keywords", "cluster_keywords", "cluster"]]
    return df_new

def create_surrogate_labels(dataset_name):
    df = pd.read_csv(os.path.join(get_base_dir(), dataset_name, "claims_all.csv"))
    tens = torch.tensor(df["surrogate_labels"].values)
    return tens


if __name__ == '__main__':
    #df = load_texts()
    #df = test_mumin_original()
    #df.to_csv(os.path.join(get_base_dir(), "mumin", "claims.csv"), index=False)


    dataset_name = "mumin" #"politifact"
    heterodata_dir = os.path.join(get_base_dir(), dataset_name, "heterodata")
    gt = torch.load(os.path.join(heterodata_dir, "claim_labels.pt"))
    #gt_surrogate = create_surrogate_labels(dataset_name)
    #torch.save(gt_surrogate, os.path.join(heterodata_dir, "claim_labels_surrogate.pt"))
    #gt = torch.load(os.path.join(heterodata_dir, "news_labels.pt"))
    gt_surr = torch.load(os.path.join(heterodata_dir, "claim_labels_surrogate.pt"))
    print()