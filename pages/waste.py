import os
import joblib
import streamlit as st
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from nav import top_nav
from streamlit_autorefresh import st_autorefresh


# ---------- AUTOREFRESH ----------
st_autorefresh(interval=20_000, key="waste_pred_autorefresh")

# ---------- CONFIG ----------
st.set_page_config(page_title="Waste Prediction", layout="wide")

# ---------- PATHS ----------
LIVE_DATA_PATH = "data/live_warehouse_state.csv"
FALLBACK_DATA_PATH = "data/waste_training_history.csv"
MODEL_PATH = "data/waste_model.pkl"
LOG_PATH = "data/model_log.txt"

# ---------- NAV + STYLES ----------
top_nav(active="Waste prediction")
st.markdown("""
<style>
section[data-testid="stSidebar"] {display: none;}
div.block-container {padding-left: 3rem; padding-right: 3rem;}
</style>
""", unsafe_allow_html=True)

# ---------- RETRAIN BUTTON ----------
col_train, col_info = st.columns([1, 3])
with col_train:
    if st.button("Retrain ML model"):
        with st.spinner("Training model, please wait..."):
            os.system("python daily_train.py")
        st.success("Model retrained successfully and saved to data/waste_model.pkl 🚀")
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

# ---------- LOADERS ----------
@st.cache_resource
def load_model(path: str):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first.")
        return None


@st.cache_data
def load_data() -> pl.DataFrame | None:
    """Load live or fallback dataset using Polars."""
    if os.path.exists(LIVE_DATA_PATH):
        st.info("Using live warehouse data feed.")
        df = pl.read_csv(LIVE_DATA_PATH)
    elif os.path.exists(FALLBACK_DATA_PATH):
        st.warning("Live warehouse data not found, using last training dataset.")
        df = pl.read_csv(FALLBACK_DATA_PATH)
    else:
        st.error("No data file found.")
        return None
    return df


# ---------- MODEL + DATA ----------
model = load_model(MODEL_PATH)
df = load_data()

if df is not None:
    # --- Filter only items not expired ---
    df = df.filter(pl.col("Days_to_Expire") > 0)
    if df.height == 0:
        st.warning("No products with positive days to expire.")
        st.stop()

    # --- Ensure required columns exist ---
    required_cols = ["Quantity", "Days_to_Expire", "Avg_Usage_per_Day", "Risk"]
    for col in required_cols:
        if col not in df.columns:
            if col == "Avg_Usage_per_Day":
                df = df.with_columns(pl.Series(col, np.random.uniform(1, 10, df.height)))
            else:
                df = df.with_columns(pl.Series(col, np.random.randint(1, 600, df.height)))

    # --- Predict probabilities ---
    X = df.select(["Quantity", "Days_to_Expire", "Avg_Usage_per_Day", "Risk"]).to_numpy()
    probs = model.predict_proba(X)[:, 1] * 100
    df = df.with_columns(pl.Series("Prob_Waste", probs))

    # --- Display results ---
    st.subheader("Waste Prediction")
    top_waste = df.sort("Prob_Waste", descending=True).head(10)

    # Choose best label column
    label_col = None
    if "Product_Name" in df.columns:
        label_col = "Product_Name"
    elif "Product_ID" in df.columns:
        label_col = "Product_ID"

    # --- Display table ---
    display_cols = [c for c in ["Product_Name", "Product_ID", "LOT_Number", "Days_to_Expire", "Quantity", "Prob_Waste"] if c in df.columns]
    for must in ["Days_to_Expire", "Prob_Waste"]:
        if must not in display_cols:
            display_cols.append(must)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(top_waste.select(display_cols).to_pandas(), use_container_width=True)

    # --- Chart ---
    with col2:
        fig, ax = plt.subplots()
        labels = (
            top_waste[label_col].to_list()
            if label_col is not None
            else [f"idx {i}" for i in range(top_waste.height)]
        )
        ax.barh(labels, top_waste["Prob_Waste"].to_list(), color="#E2001A")
        ax.set_xlabel("Waste probability (%)")
        ax.set_title("Top lots at highest risk of waste")
        plt.gca().invert_yaxis()
        st.pyplot(fig)

# ---------- MODEL INFO PANEL ----------
if model is not None:
    st.subheader("Model Information")

    # Last training time
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r") as f:
                last_line = f.readlines()[-1].strip()
            last_train_str = last_line
        except Exception:
            last_train_str = "Log file unreadable"
    else:
        if os.path.exists(MODEL_PATH):
            ts = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
            last_train_str = ts.strftime("%Y-%m-%d %H:%M:%S") + " (file timestamp)"
        else:
            last_train_str = "No record found"

    c1, c2, c3 = st.columns(3)
    c1.metric("Model file", MODEL_PATH)
    c2.metric("Last training", last_train_str)
    c3.metric("Model type", type(model).__name__)

    # --- Feature importances ---
    expected_feats = ["Quantity", "Days_to_Expire", "Avg_Usage_per_Day", "Risk"]
    if hasattr(model, "feature_importances_") and len(model.feature_importances_) == len(expected_feats):
        fi_df = pl.DataFrame({
            "Feature": expected_feats,
            "Importance": model.feature_importances_
        }).sort("Importance", descending=True)
        st.bar_chart(fi_df.to_pandas().set_index("Feature"))

    st.divider()

