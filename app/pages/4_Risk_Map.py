import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import plotly.express as px
import pandas as pd

from model.predict import get_bubble_data
from app.utils import require_customer

st.set_page_config(
    page_title="Risk Map — ApplauseML",
    page_icon="🗺️",
    layout="wide",
)

st.title("QA Predictive Radar")
st.caption(
    "Device-level risk map. "
    "X = historical test case failure rate. "
    "Y = weighted bug severity score (Critical=4, High=3, Med=2, Low=1). "
    "Bubble size = Predictive Risk Score = (failure rate × 50) + (severity × 10)."
)

customer    = require_customer()
all_bubbles = get_bubble_data()

if all_bubbles.empty:
    st.warning(
        "Bubble data not yet generated. Retrain from the **Data Upload** page.",
        icon="⚠️",
    )
    st.stop()

if "Customer" in all_bubbles.columns:
    bubble_df = all_bubbles[all_bubbles["Customer"] == customer].copy()
else:
    bubble_df = all_bubbles.copy()

if bubble_df.empty:
    st.warning(f"No device-level data found for **{customer}**.")
    st.stop()

st.subheader(f"Customer: {customer}")

CLUSTER_COLORS = {
    "Critical Hotspot":                   "red",
    "Nuisance Zone (High Fail, Low Sev)": "orange",
    "Stable Yielder":                     "green",
    "Low ROI":                            "lightgray",
}

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    clusters = sorted(bubble_df["Optimization_Cluster"].dropna().unique().tolist())
    selected_clusters = st.multiselect("Risk Cluster", clusters, default=clusters)

    max_score = int(bubble_df["Predictive_Risk_Score"].max())
    min_score = st.slider(
        "Minimum Risk Score",
        min_value=0,
        max_value=max(max_score, 1),
        value=0,
        step=1,
        help="Hide devices below this Predictive Risk Score.",
    )

    st.divider()

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = bubble_df[
    bubble_df["Optimization_Cluster"].isin(selected_clusters)
    & bubble_df["Predictive_Risk_Score"].ge(min_score)
].copy()

if filtered.empty:
    st.info("No devices match the current filters.")
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Devices shown", f"{len(filtered):,}")
m2.metric("Critical Hotspots",  str((filtered["Optimization_Cluster"] == "Critical Hotspot").sum()))
m3.metric("Nuisance Zone",      str((filtered["Optimization_Cluster"] == "Nuisance Zone (High Fail, Low Sev)").sum()))
m4.metric("Stable Yielders",    str((filtered["Optimization_Cluster"] == "Stable Yielder").sum()))

st.divider()

# ── Bubble chart ──────────────────────────────────────────────────────────────
hover_cols = {"Primary_Failing_Component": True, "Total_Bugs": True, "Total_Runs": True, "Predictive_Risk_Score": True}
# Only include columns that exist
hover_cols = {k: v for k, v in hover_cols.items() if k in filtered.columns}

fig = px.scatter(
    filtered,
    x="Historical_Failure_Rate",
    y="Severity_Index",
    size="Predictive_Risk_Score",
    color="Optimization_Cluster",
    hover_name="Device",
    hover_data=hover_cols,
    title=f"QA Predictive Radar — {customer}",
    labels={
        "Historical_Failure_Rate": "Historical Failure Probability (%)",
        "Severity_Index":          "Weighted Bug Severity Score",
        "Optimization_Cluster":    "Risk Cluster",
    },
    size_max=50,
    color_discrete_map=CLUSTER_COLORS,
    template="plotly_dark",
)

fig.update_layout(
    xaxis_tickformat=".0%",
    height=600,
    legend=dict(
        title="Risk Cluster",
        orientation="h",
        yanchor="bottom",
        y=-0.22,
        xanchor="center",
        x=0.5,
    ),
    margin=dict(t=60, b=80, l=60, r=40),
)

st.plotly_chart(fig, use_container_width=True)

# ── Cluster legend ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.markdown("🔴 **Critical Hotspot** — failure >10% AND severity >10. Prioritise immediately.")
c2.markdown("🟡 **Nuisance Zone** — failure >10% but low severity. Test volume issue.")
c3.markdown("🟢 **Stable Yielder** — healthy device / component combination.")
c4.markdown("⚪ **Low ROI** — zero failures and zero bugs. Consider removing from matrix.")

st.divider()

# ── Detail table ──────────────────────────────────────────────────────────────
with st.expander("View underlying data table"):
    display_cols = [
        "Device", "Optimization_Cluster", "Predictive_Risk_Score",
        "Historical_Failure_Rate", "Severity_Index",
        "Total_Bugs", "Total_Runs", "Failed_Runs",
    ]
    if "Primary_Failing_Component" in filtered.columns:
        display_cols.insert(2, "Primary_Failing_Component")

    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols]
        .rename(columns={
            "Optimization_Cluster":    "Cluster",
            "Predictive_Risk_Score":   "Risk Score",
            "Historical_Failure_Rate": "Failure Rate",
            "Severity_Index":          "Severity Score",
            "Primary_Failing_Component": "Top Failing Component",
        })
        .sort_values("Risk Score", ascending=False)
        .style.format({
            "Failure Rate":   "{:.1%}",
            "Severity Score": "{:.1f}",
            "Risk Score":     "{:.1f}",
        }),
        use_container_width=True,
    )
