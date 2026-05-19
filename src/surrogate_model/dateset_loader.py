import pandas

from datasets import load_dataset


def load_datasets():
    # loading gossipcop dataset
    df_mf = load_dataset("Jinyan1/GossipCop", split="MF")
    df_hf = load_dataset("Jinyan1/GossipCop", split="HF")
    df_mr = load_dataset("Jinyan1/GossipCop", split="MR")
    df_hr = load_dataset("Jinyan1/GossipCop", split="HR")
    df_mf_pd = df_mf.map(lambda x: {"label": 1}).to_pandas()
    df_hf_pd = df_hf.map(lambda x: {"label": 1}).to_pandas()
    df_mr_pd = df_mr.map(lambda x: {"label": 0}).to_pandas()
    df_hr_pd = df_hr.map(lambda x: {"label": 0}).to_pandas()
    gossipcop_df = pandas.concat([df_mr_pd, df_hr_pd, df_mf_pd, df_hf_pd], ignore_index=True)
    gossipcop_df = gossipcop_df[["text", "label"]]
    # loading pheme dataset
    base_url = "https://huggingface.co/datasets/difraud/difraud/resolve/main/twitter_rumours/{}.jsonl"
    pheme_train = pandas.read_json(base_url.format("train"), lines=True)
    pheme_test = pandas.read_json(base_url.format("test"), lines=True)
    pheme_valid = pandas.read_json(base_url.format("validation"), lines=True)
    pheme_df = pandas.concat([pheme_train, pheme_test, pheme_valid], ignore_index=True)
    pheme_df = pheme_df[["text", "label"]]
    # merging datasets
    combined_df = pandas.concat([gossipcop_df, pheme_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    #returning new dataset
    return combined_df["text"].tolist(), combined_df["label"].astype(int).tolist()
