import polars as pl
import numpy as np

# --- 1️⃣ Cargar dataset original ---
df = pl.read_csv("data/waste_training_history.csv")

# --- 2️⃣ Añadir ruido controlado en Risk ---
np.random.seed(42)
# Ruido normal con desviación estándar proporcional al valor original
noise = np.random.normal(0, 5, len(df))  # ±5 puntos en promedio
df = df.with_columns(
    (pl.col("Risk") + noise)
    .clip(0, 100)
    .round(2)
    .alias("Risk_noisy")
)

# --- 3️⃣ Recalcular etiquetas Waste_Label con algo de imperfección ---
# Si Risk alto -> probabilidad mayor de 1, pero no perfecta
risk = df["Risk_noisy"].to_numpy()
prob = (risk / 100) * 0.8 + np.random.uniform(0, 0.2, len(df))
labels = (prob > 0.6).astype(int)

df = df.with_columns(pl.Series("Waste_Label_noisy", labels))

# --- 4️⃣ Guardar nuevo dataset ---
output_path = "data/waste_training_history_noisy.csv"
df.select([
    "Product_ID",
    "Quantity",
    "Days_to_Expire",
    "Avg_Usage_per_Day",
    "Risk_noisy",
    "Waste_Label_noisy"
]).rename({
    "Risk_noisy": "Risk",
    "Waste_Label_noisy": "Waste_Label"
}).write_csv(output_path)

print(f"✅ Dataset con ruido guardado en {output_path}")
print(df.head(10))

