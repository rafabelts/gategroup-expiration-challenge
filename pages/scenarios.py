from pathlib import Path
import streamlit as st
import polars as pl
import matplotlib.pyplot as plt
from datetime import datetime

from nav import top_nav
from utils import predictive_ai
from streamlit_autorefresh import st_autorefresh

# ---------- NAV ----------
top_nav(active="Scenarios")

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="SmartTwin Simulation Scenarios", layout="wide")

# ---------- STYLES ----------
st.markdown("""
<style>
section[data-testid="stSidebar"] {display: none;}
div.block-container {padding-left: 3rem; padding-right: 3rem;}
</style>
""", unsafe_allow_html=True)

theme_path = Path(__file__).parent.parent / "assets" / "theme.css"
st.markdown(f"<style>{theme_path.read_text()}</style>", unsafe_allow_html=True)

# ---------- TITLE ----------
st.title("Simulation Scenarios")
st.caption("AI-driven risk simulation for flight delays and consumption changes.")
st.divider()


# ---------- LOAD DATA ----------
@st.cache_data(ttl=24*60*60)
def load_data():
    try:
        df = pl.read_csv("data/data_with_risk.csv")
    except:
        df = pl.read_csv("data/expirations_processed.csv")
    return df

df = load_data()

# ---------- FILTER OUT EXPIRED LOTS ----------
df = df.filter(pl.col("Days_to_Expire") >= 0)

if df.is_empty():
    st.warning("No valid lots to simulate. All are expired.")
    st.stop()


# ==================================================
# SCENARIO CONFIGURATION
# ==================================================
st.subheader("Define Simulation Parameters")

col1, col2, col3 = st.columns(3)
delay = col1.slider("Flight Delay (hours)", 0, 48, 0, step=2)
consumption = col2.slider("Consumption Factor (×)", 0.5, 2.0, 1.0, 0.1)
simulate_btn = col3.button("Run Simulation")

st.markdown("""
<small>
🔹 Delay = flight or handling delay before usage.<br>
🔹 Consumption factor = speed of product usage (1.2 = 20% faster use).<br>
🔹 Only non-expired lots are included.
</small>
""", unsafe_allow_html=True)


# ==================================================
# RUN AI SIMULATION
# ==================================================
if simulate_btn:
    try:
        df_sim = predictive_ai.simulate_scenario(
            df, delay_hours=delay, consumption_factor=consumption
        )
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    st.success(f"Simulation complete — model re-evaluated with delay={delay}h and consumption×{consumption}")

    # ---- Compute Delta safely in Polars ----
    df_sim = df_sim.with_columns(
        (pl.col("Prob_Waste_Simulated") - pl.col("Prob_Waste_Current")).alias("Delta_Signed")
    )

    # ---- Convert to pandas for plotting ----
    df_pd = df_sim.to_pandas()

    # ==================================================
    # VISUAL COMPARISON
    # ==================================================
    st.divider()
    st.subheader("AI Prediction Comparison (Current vs Scenario)")

    sample = df_pd.sample(min(10, len(df_pd)))

    # ---- Configurar posición desplazada de las barras ----
    import numpy as np
    y = np.arange(len(sample))
    height = 0.35  # grosor de las barras

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        y - height/2,
        sample["Prob_Waste_Current"],
        height=height,
        label="Current (AI)",
        alpha=0.85,
        color="#1f77b4",  # azul
    )
    ax.barh(
        y + height/2,
        sample["Prob_Waste_Simulated"],
        height=height,
        label="Simulated (AI)",
        alpha=0.85,
        color="#d62728",  # rojo
    )

    ax.set_yticks(y)
    ax.set_yticklabels(sample["Product_Name"])
    ax.set_xlabel("Predicted Waste Probability (%)")
    ax.set_title("AI-predicted impact of flight delay or consumption change", fontsize=11, weight="bold")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


    # ==================================================
    # SIMULATION SUMMARY
    # ==================================================
    st.divider()
    st.subheader("Simulation Summary")

    colA, colB, colC = st.columns(3)
    colA.metric("Average Waste Change (%)", f"{df_pd['Delta_Signed'].mean():.2f}")
    colB.metric("Max Waste Increase (%)", f"{df_pd['Delta_Signed'].max():.2f}")
    colC.metric("Products Impacted", (df_pd['Delta_Signed'].abs() > 5).sum())

    # ==================================================
    # TOP IMPACTED PRODUCTS
    # ==================================================
    st.markdown("#### Top 15 Most Affected Products")

    st.dataframe(
        df_pd[["Product_Name", "Days_to_Expire", "Prob_Waste_Current", "Prob_Waste_Simulated", "Delta_Signed"]]
        .sort_values("Delta_Signed", ascending=False)
        .head(15),
        use_container_width=True
    )

    st.caption(f"Simulation run at {datetime.now():%Y-%m-%d %H:%M:%S}")

else:
    st.info("Adjust the sliders and click **Run Simulation** to see AI-based risk projections.")

