import os
import pandas as pd

from src.support.utils import get_base_dir


def load_texts():
    df = pd.read_csv(os.path.join(get_base_dir(), "politifact", "news.csv"))
    df_new = df[["text"]]
    return df_new


if __name__ == '__main__':
    df = load_texts()
    df.to_csv(os.path.join(get_base_dir(), "politifact", "news_texts.csv"), index=False)