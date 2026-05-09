import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from model.predict import artifacts_exist, get_risk_tables

st.set_page_config(page_title="Risk Dashboard — ApplauseML", page_icon="📊", layout="wide")
st.title("Risk Dashboard")
st.caption("Historical High/Critical bug rates by component, platform, and environment.")

if not artifacts_exist():
    st.error("Model artifacts not found. Run `python model/train.py` first.")
    st.stop()

tables = get_risk_tables()
baseline = tables["baseline"]
total = tables["total_bugs"]

m1, m2 = st.columns(2)
m1.metric("Overall H/C Baseline Rate", f"{baseline:.1%}")
m2.metric("Total Bugs in Training Data", f"{total:,}")

st.divider()


def risk_bar(df: pd.DataFrame, dim_col: str, title: str, top_n: int = 20):
    if df.empty:
        st.info(f"No data available for {title}.")
        return
    plot_df = df.head(top_n).copy()
    plot_df["color"] = plot_df["hc_rate"].apply(
        lambda r: "#d62728" if r > 0.50 else "#ff7f0e" if r > 0.35 else "#1f77b4"
    )
    plot_df["label"] = (
        plot_df["hc_rate"].map("{:.0%}".format)
        + " (n="
        + plot_df["n_bugs"].astype(str)
        + ")"
    )
    fig = go.Figure(
        go.Bar(
            x=plot_df["hc_rate"],
            y=plot_df[dim_col].astype(str),
            orientation="h",
            marker_color=plot_df["color"],
            text=plot_df["label"],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "H/C Rate: %{x:.1%}<br>"
                f"vs Baseline: baseline<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.add_vline(
        x=baseline,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Baseline {baseline:.1%}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=title,
        xaxis_title="High/Critical Rate",
        yaxis=dict(autorange="reversed"),
        height=max(350, len(plot_df) * 28 + 80),
        margin=dict(l=10, r=120, t=50, b=40),
        plot_bgcolor="white",
        xaxis=dict(tickformat=".0%", range=[0, min(1.0, plot_df["hc_rate"].max() * 1.35)]),
    )
    st.plotly_chart(fig, use_container_width=True)


tab1, tab2, tab3, tab4 = st.tabs(
    ["By Component", "By Platform", "By Environment", "By Testing Approach"]
)

with tab1:
    st.subheader("App Components — Ranked by H/C Rate")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Specific Components**")
        risk_bar(
            tables.get("App Component", pd.DataFrame()),
            "App Component",
            "High/Critical Rate by App Component",
        )
    with col_b:
        st.markdown("**Parent Components (High-Level)**")
        risk_bar(
            tables.get("Parent App Component", pd.DataFrame()),
            "Parent App Component",
            "High/Critical Rate by Parent Component",
        )

with tab2:
    risk_bar(
        tables.get("Platform Product Name", pd.DataFrame()),
        "Platform Product Name",
        "High/Critical Rate by Platform",
    )

with tab3:
    risk_bar(
        tables.get("Development Stage", pd.DataFrame()),
        "Development Stage",
        "High/Critical Rate by Development Stage",
    )

with tab4:
    risk_bar(
        tables.get("Testing Approach", pd.DataFrame()),
        "Testing Approach",
        "High/Critical Rate by Testing Approach",
    )

st.divider()
st.caption(
    "Red = >50% H/C rate  |  Orange = 35–50%  |  Blue = <35%  |  "
    "Dashed line = overall baseline. Minimum 10 bugs per row."
)
