import polars as pl
from datetime import date

def recalc_risk(df: pl.DataFrame) -> pl.DataFrame:
    today = date.today()

    # --- 1. Detect dtype actual ---
    dtype = df.schema.get("Expiry_Date")

    # --- 2. Convertir a Date si es string (Utf8) ---
    if dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("Expiry_Date").str.strptime(pl.Date, strict=False).alias("Expiry_Date")
        )
    elif dtype != pl.Date:
        df = df.with_columns(
            pl.col("Expiry_Date").cast(pl.Date, strict=False).alias("Expiry_Date")
        )

    # --- 3. Calcular Days_to_Expire de forma segura ---
    df = df.with_columns(
        (pl.col("Expiry_Date") - pl.lit(today).cast(pl.Date))
        .dt.total_days()
        .alias("Days_to_Expire")
    )

    # --- 4. Calcular Risk_Score ---
    df = df.with_columns(
        (100 - (pl.col("Days_to_Expire") * 10))
        .clip(0, 100)
        .alias("Risk_Score")
    )

    return df

