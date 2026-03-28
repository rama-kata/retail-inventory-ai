import pandas as pd


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    sales_sum = df.groupby("sku")["total_sales"].sum()
    sales_mean = df.groupby("sku")["total_sales"].mean()

    turnover = (sales_sum / sales_mean).rename("turnover_rate")

    proxy_stock = df.groupby("sku").last()["total_sales"]
    days_on_hand = (proxy_stock / (sales_mean / 30)).rename("days_on_hand")

    annual_sales = sales_sum.sort_values(ascending=False)
    cum_pct = annual_sales.cumsum() / annual_sales.sum()
    prev_cum = cum_pct.shift(1, fill_value=0.0)
    abc_class = prev_cum.map(
        lambda x: "A" if x < 0.70 else ("B" if x < 0.90 else "C")
    ).rename("abc_class")

    kpis = pd.concat([turnover, days_on_hand, abc_class], axis=1).reset_index()
    return kpis
