import pandas


def load_datasets():
    #TODO
    df_gossip = pandas.DataFrame({
        "text": ["a fake celebrity story", "a real verified news about a movie"],
        "label": ["fake", "real"]
    })

    df_pheme = pandas.DataFrame({
        "text": ["breaking: completely fabricated rumor!", "confirmed reports from the ground."],
        "label": ["false", "true"]
    })

    df_combined = pandas.concat([df_gossip, df_pheme], ignore_index=True)
    label_map = {"fake": 1, "false": 1, "real": 0, "true": 0}
    df_combined["y"] = df_combined["label"].str.lower().map(label_map)
    df_combined = df_combined.dropna(subset=["y"])
    return df_combined["text"].tolist(), df_combined["y"].astype(int).tolist()
