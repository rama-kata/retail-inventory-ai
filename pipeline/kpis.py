import pandas as pd


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    sales_sum = df.groupby("sku")["total_sales"].sum()

    # Turnover rate: total sales / average inventory proxy (mean warehouse sales).
    # Using wh_sales as the best available proxy for on-hand inventory.
    avg_inventory = df.groupby("sku")["wh_sales"].mean().replace(0, float("nan"))
    turnover = (sales_sum / avg_inventory).rename("turnover_rate")

    # Days-on-hand: how many days of sales the current inventory covers,
    # using the last month's warehouse sales as proxy for current stock.
    proxy_stock = df.groupby("sku").last()["wh_sales"].fillna(0)
    daily_sales = sales_sum / df.groupby("sku").size() / 30
    days_on_hand = (proxy_stock / daily_sales.replace(0, float("nan"))).rename("days_on_hand")

    # ABC classification using cumulative revenue share.
    # We classify based on each item's *prior* cumulative share so the first
    # item (highest revenue) always lands in class A, even if it alone exceeds
    # the 70% threshold. Strict < (not <=) keeps boundary items in the higher class.
    annual_sales = sales_sum.sort_values(ascending=False)
    cum_pct = annual_sales.cumsum() / annual_sales.sum()
    prev_cum = cum_pct.shift(1, fill_value=0.0)
    abc_class = prev_cum.map(
        lambda x: "A" if x < 0.70 else ("B" if x < 0.90 else "C")
    ).rename("abc_class")

    kpis = pd.concat([turnover, days_on_hand, abc_class], axis=1).reset_index()
    return kpis
