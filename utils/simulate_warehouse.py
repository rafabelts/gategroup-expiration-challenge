import polars as pl
import random
from datetime import date, timedelta

def simulate_warehouse(df: pl.DataFrame) -> pl.DataFrame:
    """Simulate random warehouse updates; Expiry_Date handled as string for schema consistency."""
    if df.is_empty():
        return df

    df = df.clone()
    n_rows = df.height
    sample_size = max(1, int(n_rows * 0.3))
    sample_idx = random.sample(range(n_rows), sample_size)

    df_dict = df.to_dict(as_series=False)

    # --- Update existing rows ---
    for i in sample_idx:
        df_dict["Quantity"][i] = int(max(0, int(df_dict["Quantity"][i]) - random.randint(5, 20)))
        df_dict["Days_to_Expire"][i] = int(df_dict["Days_to_Expire"][i] - random.choice([0, 1]))
        df_dict["Risk_Score"][i] = float(max(0, min(100, 100 - df_dict["Days_to_Expire"][i] * 10)))

        days = df_dict["Days_to_Expire"][i]
        if days < 0:
            df_dict["Status"][i] = "Expired"
        elif days <= 2:
           df_dict["Status"][i] = "Critical"
        elif days <= 7:
            df_dict["Status"][i] = "Medium"
        else:
            df_dict["Status"][i] = "OK"

    # build the updated df
    df_updated = pl.DataFrame(df_dict, strict=False)

    # --- Add new mock row ---
    new_row = {
        "Product_ID": f"NEW{random.randint(100,999)}",
        "Product_Name": random.choice(["Snack Box", "Juice Pack", "Salad Bowl", "Cheese Portion"]),
        "Weight_or_Volume": random.choice(["100g", "250ml", "180g"]),
        "LOT_Number": f"LOT-{random.randint(100,999)}",
        # 👇 store as string to match CSV schema
        "Expiry_Date": (date.today() + timedelta(days=random.randint(3, 45))).isoformat(),
        "Quantity": int(random.randint(50, 300)),
        "Days_to_Expire": int(random.randint(3, 45)),
        "Status": "Vigente",
        "Avg_Usage_per_Day": float(round(random.uniform(1.0, 8.0), 2)),
        "Risk_Score": 0.0,
    }
    new_row["Risk_Score"] = float(max(0, min(100, 100 - new_row["Days_to_Expire"] * 10)))
    new_df = pl.DataFrame([new_row], strict=False)

    # 👇 Force column to Utf8 to match base schema
    new_df = new_df.with_columns(pl.col("Expiry_Date").cast(pl.Utf8, strict=False))
    df_updated = df_updated.with_columns(pl.col("Expiry_Date").cast(pl.Utf8, strict=False))

    # concatenate safely
    df_final = pl.concat([df_updated, new_df], how="vertical")

    # enforce numeric types
    df_final = df_final.with_columns([
        pl.col("Quantity").cast(pl.Int64, strict=False),
        pl.col("Days_to_Expire").cast(pl.Int64, strict=False),
        pl.col("Risk_Score").cast(pl.Float64, strict=False),
        pl.col("Avg_Usage_per_Day").cast(pl.Float64, strict=False),
    ])

    return df_final

