"""Data loading and paths."""

import pandas as pd

DATA_PATH = "data/512890.SH_hfq.parquet"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df
