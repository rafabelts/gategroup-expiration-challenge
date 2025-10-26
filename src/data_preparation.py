import os
from datetime import date
import pandas as pd # importado para leer xlsx
import polars as pl

from config import expirations_preparation as ep
from utils.normalize_text_col import normalize_text_col
from utils.parse_expiry import ParseExpiry

parse_expiry = ParseExpiry()

# --- Carga ----
df_pd = pd.read_excel(ep.INPUT_XLSX)
df = pl.from_pandas(df_pd)

# ---------- Tipos y normalización básica ----------
df = df.with_columns([
    # Normaliza textos
    normalize_text_col(pl.col("Product_ID")).alias("Product_ID"),
    normalize_text_col(pl.col("Product_Name")).alias("Product_Name"),
    normalize_text_col(pl.col("Weight_or_Volume")).alias("Weight_or_Volume"),
    normalize_text_col(pl.col("LOT_Number")).str.to_uppercase().alias("LOT_Number"),

    # Cantidad segura (int >= 0, nulos -> 0)
    pl.col("Quantity")
    .cast(pl.Int64, strict=False)
    .fill_null(0)
    .clip(lower_bound=0)
    .alias("Quantity"),

    # Fecha
    parse_expiry.parse_expiry_expr("Expiry_Date").alias("Expiry_Date")
])

# ---------- Calidad de datos: detectar registros malos ----------
bad_missing = df.filter(
    pl.col("Product_ID").is_null() |
        pl.col("Product_Name").is_null() |
        pl.col("Expiry_Date").is_null()
).with_columns(pl.lit("MISSING_REQUIRED").alias("quality_issue"))

bad_future_weird = pl.DataFrame([])


# Combina logs “malos”
dfs = [df for df in [bad_missing, bad_future_weird] if df.width > 0]
if dfs:
    quality_log = pl.concat(dfs, how="vertical", rechunk=True)
else:
    quality_log = pl.DataFrame()


# Filtra los buenos para seguir
df = df.filter(~pl.any_horizontal(
    pl.col("Product_ID").is_null(),
    pl.col("Product_Name").is_null(),
    pl.col("Expiry_Date").is_null()
))

# ---------- Deduplicación por clave (suma Quantity) ----------
# Clave mínima razonable para lotes
key_cols = ["Product_ID", "LOT_Number", "Expiry_Date"]
agg_cols = [
    pl.col("Quantity").sum().alias("Quantity"),
    # Conserva el primer valor para columnas no clave
    pl.col("Product_Name").first().alias("Product_Name"),
    pl.col("Weight_or_Volume").first().alias("Weight_or_Volume"),
]
df = df.group_by(key_cols).agg(agg_cols).select(
    "Product_ID","Product_Name","Weight_or_Volume","LOT_Number","Expiry_Date","Quantity"
)

# ---------- Derivados: Days_to_Expire, Status ----------
today = date.today()
df = df.with_columns([
    (pl.col("Expiry_Date") - pl.lit(today)).dt.total_days().cast(pl.Int64).alias("Days_to_Expire"),
])

df = df.with_columns([
    pl.when(pl.col("Days_to_Expire") < 0).then(pl.lit("Expirado"))
      .when(pl.col("Days_to_Expire") <= 2).then(pl.lit("Crítico"))
      .when(pl.col("Days_to_Expire") <= 7).then(pl.lit("Medio"))
      .otherwise(pl.lit("Vigente"))
      .alias("Status")
])


# ---------- Enriquecimiento: Avg_Usage_per_Day y Zone ----------
# Valor por defecto editable en el dashboard
df = df.with_columns([
    (pl.col("Quantity") / (pl.col("Days_to_Expire").clip(lower_bound=1))).round(2)
    .alias("Avg_Usage_per_Day")
])

# ---------- Orden y exportación ----------
df = df.select([
    "Product_ID","Product_Name","Weight_or_Volume","LOT_Number","Expiry_Date","Quantity",
    "Days_to_Expire","Status","Avg_Usage_per_Day"
]).sort(["Days_to_Expire","Quantity"], descending=[False, True])

# Guarda outputs
os.makedirs("data", exist_ok=True)
df.write_csv(ep.OUTPUT_CLEAN)

if quality_log.height > 0:
    quality_log.write_csv(ep.OUTPUT_QUALITY_LOG)
else:
    # crea un log vacío con mismas columnas + reason
    pl.DataFrame({"quality_issue": [], "note": []}).write_csv(ep.OUTPUT_QUALITY_LOG)

print(f"Datos limpios → {ep.OUTPUT_CLEAN}")
print(f"Log de calidad → {ep.OUTPUT_QUALITY_LOG}")

