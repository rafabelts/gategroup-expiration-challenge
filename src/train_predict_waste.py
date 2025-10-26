import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- Reproducibilidad ---
np.random.seed(42)
n = 500

# --- Crear DataFrame con Polars ---
df = pl.DataFrame({
    "Product_ID": np.random.choice(["MLK003","BIS007","FRU009","SAL008","SNK001","CHS010"], size=n),
    "Quantity": np.random.randint(50, 800, size=n),
    "Days_to_Expire": np.random.randint(-10, 30, size=n),
    "Avg_Usage_per_Day": np.random.uniform(1, 15, size=n),
})

# --- Calcular riesgo base ---
df = df.with_columns(
    (
        100 - (pl.col("Days_to_Expire") * pl.col("Avg_Usage_per_Day")) / ((pl.col("Quantity") / 10) + 1)
    ).clip(0, 100).alias("Risk")
)

# --- Etiqueta real: desperdicio (1) o no (0) ---
df = df.with_columns(
    (
        ((pl.col("Days_to_Expire") < 0) |
         ((pl.col("Days_to_Expire") < 5) & (pl.col("Quantity") > pl.col("Avg_Usage_per_Day") * 5)))
        .cast(pl.Int8)
    ).alias("Waste_Label")
)

# --- Separar X e y ---
X = df.select(["Quantity", "Days_to_Expire", "Avg_Usage_per_Day", "Risk"]).to_numpy()
y = df["Waste_Label"].to_numpy()

# --- Entrenar modelo ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --- Evaluación ---
y_pred = model.predict(X_test)
print("=== Reporte de Clasificación ===")
print(classification_report(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# --- Nuevos datos para predecir ---
new_data = pl.DataFrame({
    "Quantity": [200, 500, 600],
    "Days_to_Expire": [2, 10, -3],
    "Avg_Usage_per_Day": [5, 8, 3],
    "Risk": [80, 45, 95]
})

pred = model.predict_proba(new_data.to_numpy())[:, 1]
new_data = new_data.with_columns(pl.Series("Prob_Waste", np.round(pred * 100, 2)))

print("\nPredicciones de desperdicio:")
print(new_data)

