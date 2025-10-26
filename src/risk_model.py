import polars as pl
from datetime import date

df = pl.read_csv("data/expirations_processed.csv")

today = date.today()

df = df.with_columns([
  
  pl.col("Expiry_Date").str.strptime(pl.Date, strict=False).alias("Expiry_Date"),

  # actualiza respecto al día actual
  (pl.col("Expiry_Date") - pl.lit(today)).dt.days().alias("Days_to_Expire"),


  # Recalcula Status en base a Days_to_Expire
  pl.when(pl.col("Days_to_Expire") < 0).then(pl.lit("Expired"))
  .when(pl.col("Days_to_Expire") <= 2).then(pl.lit("Critical"))
  .when(pl.col("Days_to_Expire") <= 7).then(pl.lit("Medium"))
  .otherwise(pl.lit("OK"))
  .alias("Status"),


  # riesgo dinamico
  (100 - (pl.col("Days_to_Expire") * 10)).clip(0, 100).alias("Risk_Score"),
])

df.write_csv("data/data_with_risk.csv")

