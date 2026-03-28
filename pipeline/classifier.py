import pandas as pd


def classify_slow_movers(kpis: pd.DataFrame, percentile: int = 30) -> pd.DataFrame:
    kpis = kpis.copy()
    cutoff = kpis["turnover_rate"].quantile(percentile / 100)
    kpis["slow_mover"] = (kpis["turnover_rate"] < cutoff).astype(int)
    return kpis
