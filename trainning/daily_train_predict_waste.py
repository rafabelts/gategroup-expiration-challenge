import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime

# --- Rutas ---
DATA_PATH = "data/waste_training_history.csv"
MODEL_PATH = "data/waste_model.pkl"
LOG_PATH = "data/model_log.txt"

# --- Cargar dataset ---
df = pl.read_csv(DATA_PATH)

required = ["Quantity", "Days_to_Expire", "Avg_Usage_per_Day", "Risk", "Waste_Label"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Faltan columnas en {DATA_PATH}: {missing}")

# --- Features y etiqueta ---
X = df.select(["Quantity", "Days_to_Expire", "Avg_Usage_per_Day", "Risk"]).to_numpy()
y = df["Waste_Label"].to_numpy()

# --- División train/test (75% / 25%) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# --- Entrenamiento ---
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --- Evaluación ---
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

# --- Reporte más completo ---
report = classification_report(y_test, model.predict(X_test), digits=3)

# --- Guardar modelo ---
joblib.dump(model, MODEL_PATH)

# --- Log actualizado ---
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(LOG_PATH, "a") as f:
    f.write(
        f"\n[{timestamp}] Modelo reentrenado\n"
        f"Precisión (train): {train_acc:.3f}\n"
        f"Precisión (test): {test_acc:.3f}\n"
        f"{report}\n"
    )

# --- Consola ---
print(f"Modelo reentrenado y guardado en {MODEL_PATH}")
print(f"Última actualización: {timestamp}")
print(f"Precisión (train): {train_acc:.3f}")
print(f"Precisión (test):  {test_acc:.3f}")
