import joblib
import polars as pl
import numpy as np

def simulate_scenario(df: pl.DataFrame, delay_hours: float = 0, consumption_factor: float = 1.0,
                      model_path: str = "data/waste_model.pkl") -> pl.DataFrame:
    """
    Simulate how RandomForest predictions change if flight is delayed or consumption rate changes.
    Uses Polars for processing.
    """

    # --- Load model ---
    try:
        model = joblib.load(model_path)
    except:
        raise RuntimeError("No trained model found. Please train it first in the Predictive AI page.")

    # --- Detect correct risk column name ---
    risk_col = "Risk_Score" if "Risk_Score" in df.columns else "Risk"

    # --- Adjusted columns (simulated scenario) ---
    df_sim = df.with_columns([
        (pl.col("Days_to_Expire") - (delay_hours / 24)).alias("Days_to_Expire_adj"),
        (pl.col("Avg_Usage_per_Day") * consumption_factor).alias("Avg_Usage_per_Day_adj")
    ])

    # --- Required features ---
    features = ["Quantity", "Days_to_Expire", "Avg_Usage_per_Day", risk_col]

    # Ensure features exist
    for col in features:
        if col not in df_sim.columns:
            raise ValueError(f"Missing required feature: {col}")

    # --- Prepare numpy arrays for model ---
    X_current = df_sim.select(features).to_numpy()
    X_scenario = df_sim.select([
        "Quantity",
        "Days_to_Expire_adj",
        "Avg_Usage_per_Day_adj",
        risk_col
    ]).to_numpy()

    # --- Predict probabilities ---
    prob_current = model.predict_proba(X_current)[:, 1] * 100
    prob_sim = model.predict_proba(X_scenario)[:, 1] * 100

    # --- Add results to dataframe ---
    df_sim = df_sim.with_columns([
        pl.Series("Prob_Waste_Current", prob_current),
        pl.Series("Prob_Waste_Simulated", prob_sim),
        pl.Series("Delta_Signed", prob_sim - prob_current)
    ])

    return df_sim

def predict_probability(df, model_path="data/waste_model.pkl"):
    """
    Predict probability of expiration for each lot using the trained RandomForest model.
    Works with both Polars and Pandas DataFrames.
    """

    # --- Load model ---
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        raise RuntimeError("⚠️ Model not found. Train it first in the Predictive AI page.")

    # --- Convert Polars → Pandas if needed ---
    df_pd = df.to_pandas() if isinstance(df, pl.DataFrame) else df.copy()

    # --- Detect risk column name ---
    risk_col = "Risk_Score" if "Risk_Score" in df_pd.columns else "Risk"

    # --- Ensure required features exist ---
    required = ["Quantity", "Days_to_Expire", "Avg_Usage_per_Day", risk_col]
    for col in required:
        if col not in df_pd.columns:
            raise ValueError(f"Missing required feature: {col}")

    # --- Prepare features for the model ---
    X = df_pd[required]

    # --- Predict probability of waste (class 1) ---
    probs = model.predict_proba(X)[:, 1] * 100

    # --- Add prediction column ---
    df_pd["Probability_of_Expiration"] = probs.round(2)

    # --- Return as Polars again (optional) ---
    return pl.from_pandas(df_pd)
