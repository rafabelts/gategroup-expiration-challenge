from pathlib import Path
import streamlit as st
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta

from nav import top_nav
from streamlit_autorefresh import st_autorefresh

# === UTILITIES ===
from utils import simulate_warehouse, risk_utils


# ---------- NAV ----------
top_nav(active="SmartTwin Warehouse")

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="SmartTwin Warehouse", layout="wide")

# ---------- STYLES ----------
st.markdown("""
<style>
/* Hide sidebar completely */
section[data-testid="stSidebar"] {display: none;}
/* Adjust container width to occupy full screen */
div.block-container {padding-left: 3rem; padding-right: 3rem;}
</style>
""", unsafe_allow_html=True)

theme_path = Path(__file__).parent / "assets" / "theme.css"
st.markdown(f"<style>{theme_path.read_text()}</style>", unsafe_allow_html=True)

# ==================================================
#           AUTOREFRESH CONTROLLER
# ==================================================
# This triggers a full rerun every 30 seconds (30,000 ms)
count = st_autorefresh(interval=20_000, limit=None, key="warehouse_autoupdate")

# ==================================================
#           DATA LOADING
# ==================================================
@st.cache_data(ttl=24*60*60)
def load_and_compute():
    try:
        df = pl.read_csv("data/expirations_processed.csv")
    except:
        df = pl.read_csv("data/data_with_risk.csv")

    today = date.today()

    # --- Detect and normalize Expiry_Date dtype ---
    dtype = df.schema.get("Expiry_Date")

    if dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("Expiry_Date")
            .str.strptime(pl.Date, strict=False)
            .alias("Expiry_Date")
        )

    df = df.with_columns(
        pl.when(pl.col("Expiry_Date").is_null())
        .then(pl.lit(today))
        .otherwise(pl.col("Expiry_Date"))
        .alias("Expiry_Date")
    )

    # --- Compute days to expire ---
    df = df.with_columns([
        (pl.col("Expiry_Date").cast(pl.Date) - pl.lit(today).cast(pl.Date))
        .dt.total_days()
        .cast(pl.Int64, strict=False)
        .alias("Days_to_Expire")
    ])

    # --- Status and risk ---
    df = df.with_columns([
        pl.when(pl.col("Days_to_Expire") < 0).then(pl.lit("Expired"))
        .when(pl.col("Days_to_Expire") <= 2).then(pl.lit("Critical"))
        .when(pl.col("Days_to_Expire") <= 7).then(pl.lit("Medium"))
        .otherwise(pl.lit("Active"))
        .alias("Status"),

        (100 - (pl.col("Days_to_Expire") * 10))
        .clip(0, 100)
        .alias("Risk_Score"),
    ])

    return df, datetime.now()


df, computed_at = load_and_compute()

# ---------- REFRESH INFO ----------
next_refresh = computed_at + timedelta(days=1)
st.info(
    f"**Daily calculation** | Last update: {computed_at:%Y-%m-%d %H:%M} | "
    f"Next full cache refresh: {next_refresh:%Y-%m-%d %H:%M}"
)

# ---------- BUTTON STYLE ----------
st.markdown("""
    <style>
    div[data-testid="stButton"] button {
        background-color: #000164;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #000164;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- RELOAD BUTTON ----------
col_refresh, _ = st.columns([1, 3])
with col_refresh:
    if st.button("Recalculate now"):
        st.cache_data.clear()
        st.rerun()

# ==================================================
#        LIVE WAREHOUSE SIMULATION
# ==================================================
# Automatically simulate every refresh or on button click
if count > 0:
    df = simulate_warehouse.simulate_warehouse(df)
    df = risk_utils.recalc_risk(df)
    st.toast("Warehouse updated")

    df.write_csv("data/live_warehouse_state.csv")

# ==================================================
#            VISUALIZATIONS
# ==================================================
st.title("Warehouse Twin")

# Convert to pandas for matplotlib
df_pd = df.to_pandas()
df_visible = df_pd[df_pd["Risk_Score"] < 100]

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    risky = df_visible[df_visible["Risk_Score"] > 0]
    top_risk = risky.nlargest(10, "Risk_Score")
    ax.bar(top_risk["Product_Name"], top_risk["Risk_Score"], color="#E2001A")
    ax.set_title("Top 10 products with highest expiration risk", fontsize=11, weight="bold")
    ax.set_xlabel("Product")
    ax.set_ylabel("Risk (%)")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig, use_container_width=True)

with col2:
    df_no_expired = df_visible[df_visible["Status"].str.lower() != "expired"]
    state_counts = df_no_expired["Status"].value_counts()

    fig2, ax2 = plt.subplots()
    if state_counts.empty:
        ax2.text(0.5, 0.5, "No non-expired lots", ha="center", va="center")
        ax2.axis("off")
    else:
        ax2.pie(
            state_counts,
            labels=state_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            textprops={'fontsize': 8}
        )
        ax2.set_title("Lot distribution by status", fontsize=11, weight="bold")

    st.pyplot(fig2, use_container_width=True)

st.divider()

# ==================================================
#           KPIs & TABLE
# ==================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total lots", len(df_pd))
col2.metric("Expired", (df_pd["Days_to_Expire"] < 0).sum())
col3.metric("Critical (≤2 days)", (df_pd["Days_to_Expire"] <= 2).sum())
col4.metric("Medium risk (≤7 days)", ((df_pd["Days_to_Expire"] > 2) & (df_pd["Days_to_Expire"] <= 7)).sum())

st.divider()

# ---------- FILTERS ----------
status_opts = df_visible["Status"].dropna().unique().tolist()
status_selected = st.multiselect("Filter by status", status_opts, default=status_opts)

search = st.text_input("Search by Product or Lot")

filtered = df_visible[df_visible["Status"].isin(status_selected)]
if search:
    filtered = filtered[
        filtered["Product_Name"].str.contains(search, case=False, na=False)
        | filtered["LOT_Number"].astype(str).str.contains(search, case=False, na=False)
    ]

st.subheader("Current lot status")
st.dataframe(filtered, use_container_width=True)

