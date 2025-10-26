from pathlib import Path
import streamlit as st
import polars as pl
from datetime import datetime

from nav import top_nav
from utils import predictive_ai


# ---------- NAV ----------
top_nav(active="Operational Intelligence")

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="SmartTwin Operational Intelligence", layout="wide")

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
st.title("Operational Intelligence")
st.caption("Action-oriented AI layer — detect risks and suggest next operational moves.")
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

# Exclude expired lots
df = df.filter(pl.col("Days_to_Expire") >= 0)

if df.is_empty():
    st.warning("No active lots in warehouse.")
    st.stop()

# Predict with model (optional if already done)
try:
    df_pred = predictive_ai.predict_probability(df)
except:
    df_pred = df.with_columns(pl.lit(0).alias("Probability_of_Expiration"))


# ==================================================
# ALERT LOGIC
# ==================================================
st.subheader("Automatic Alerts")

high_risk = df_pred.filter(pl.col("Risk_Score") > 85)
high_prob = df_pred.filter(pl.col("Probability_of_Expiration") > 75)

if high_risk.is_empty() and high_prob.is_empty():
    st.success("All lots are currently within safe thresholds.")
else:
    if not high_risk.is_empty():
        st.warning(f"{len(high_risk)} lots exceed 85% Risk_Score — immediate attention required.")
        st.dataframe(high_risk.select(["Product_Name", "LOT_Number", "Days_to_Expire", "Risk_Score"]), use_container_width=True)

    if not high_prob.is_empty():
        st.error(f"{len(high_prob)} lots have >75% probability of expiration (AI prediction).")
        st.dataframe(high_prob.select(["Product_Name", "LOT_Number", "Probability_of_Expiration", "Days_to_Expire"]), use_container_width=True)


# ==================================================
# RECOMMENDED ACTIONS
# ==================================================
st.divider()
st.subheader("Recommended Actions (Next Moves)")

# Define mock zones (simulating warehouse areas)
ZONES = ["A1", "A2", "A3", "B1", "B2", "C1", "C2", "C3"]

def recommend_action(row):
    """Simple rule engine for operational decisions."""
    if row["Risk_Score"] > 90 or row["Probability_of_Expiration"] > 80:
        return f"🔁 Move lot {row['LOT_Number']} to zone A1 (fast rotation area)"
    elif row["Risk_Score"] > 70:
        return f"📦 Prioritize lot {row['LOT_Number']} for next dispatch"
    elif row["Days_to_Expire"] <= 3:
        return f"🧊 Store lot {row['LOT_Number']} in cold zone (B1)"
    else:
        return f"✅ Keep lot {row['LOT_Number']} in current zone"

# Convert to pandas for apply
df_pd = df_pred.to_pandas()
df_pd["Suggested_Action"] = df_pd.apply(recommend_action, axis=1)

# Show top recommendations
actions = df_pd[["Product_Name", "LOT_Number", "Days_to_Expire", "Risk_Score", "Probability_of_Expiration", "Suggested_Action"]]
st.dataframe(actions.sort_values(["Risk_Score", "Days_to_Expire"], ascending=[False, True]), use_container_width=True)


# ==================================================
# 🧩 DECISION SUMMARY
# ==================================================
st.divider()
st.subheader("Operational Summary")

col1, col2, col3 = st.columns(3)
col1.metric("Lots above 85% Risk", len(high_risk))
col2.metric("Predicted Expiring Lots", len(high_prob))
col3.metric("Actions Suggested", len(actions))

st.caption(f"Last evaluated: {datetime.now():%Y-%m-%d %H:%M:%S}")

