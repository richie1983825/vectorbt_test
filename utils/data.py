"""行情数据加载模块。

从 parquet 文件读取 K 线数据，默认数据源为通达信转换的后复权数据。
"""

import pandas as pd

# 默认数据路径：512890.SH（中证红利 ETF）后复权
DATA_PATH = "data/512890.SH_hfq.parquet"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """加载 parquet 格式的行情数据。

    读取后将 trade_date 索引转换为 DatetimeIndex，方便按日期切片和分组。

    Returns:
        DataFrame，列为 Open/High/Low/Close/Volume/Amount/adj_factor
    """
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df
