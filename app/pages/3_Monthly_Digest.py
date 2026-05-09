import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from model.predict import artifacts_exist, get_risk_tables

st.set_page_config(page_title="Monthly Digest — ApplauseML", page_icon="📅", layout="wide")
st.title("Monthly Digest")
st.caption(
    "Historical pattern analysis — trends, top risk factors, and period-over-period changes. "
    "Refresh by rerunning `python model/train.py`."
)

if not artifacts_exist():
    st.error("Model artifacts not found. Run `python model/train.py` first.")
    st.stop()

tables = get_risk_tables()
baseline = tables["baseline"]

st.metric("Overall H/C Baseline Rate (all-time)", f"{baseline:.1%}")
st.divider()

monthly = tables.get("monthly_trend")
if monthly is not None and not monthly.empty:
    st.subheader("H/C Rate Over Time")
    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=monthly["month"],
            y=monthly["hc_rate"],
            mode="lines+markers",
            name="H/C Rate",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
            hovertemplate="Month: %{x}<br>H/C Rate: %{y:.1%}<extra></extra>",
        )
    )
    fig_trend.add_hline(
        y=baseline,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"All-time baseline {baseline:.1%}",
        annotation_position="top left",
    )
    fig_trend.update_layout(
        xaxis_title="Month",
        yaxis_title="High/Critical Rate",
        yaxis=dict(tickformat=".0%"),
        height=350,
        plot_bgcolor="white",
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    if len(monthly) >= 2:
        last = monthly.iloc[-1]
        prev = monthly.iloc[-2]
        delta = last["hc_rate"] - prev["hc_rate"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Month", last["month"])
        col2.metric(
            "Latest H/C Rate",
            f"{last['hc_rate']:.1%}",
            delta=f"{delta:+.1%} vs prior month",
            delta_color="inverse",
        )
        col3.metric("Bugs Recorded", f"{int(last['n_bugs']):,}")
else:
    st.info(
        "Monthly trend data not available. The training data may not include a bug creation date column.",
        icon="ℹ️",
    )

st.divider()
st.subheader("Top Risk Areas — All Time")

tab1, tab2, tab3 = st.tabs(["Components", "Platforms", "Environments"])

def top_risk_chart(df: pd.DataFrame, dim_col: str, title: str, n: int = 15):
    if df.empty:
        st.info("No data available.")
        return
    top = df.head(n).copy()
    fig = px.bar(
        top,
        x="hc_rate",
        y=dim_col,
        orientation="h",
        color="hc_rate",
        color_continuous_scale=["#2ca02c", "#ff7f0e", "#d62728"],
        range_color=[0, 1],
        text=top["hc_rate"].map("{:.0%}".format),
        hover_data={"n_bugs": True},
        labels={"hc_rate": "H/C Rate", dim_col: ""},
        title=title,
    )
    fig.add_vline(
        x=baseline,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Baseline {baseline:.1%}",
    )
    fig.update_layout(
        height=max(300, n * 30 + 80),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(tickformat=".0%"),
        coloraxis_showscale=False,
        plot_bgcolor="white",
        margin=dict(l=10, r=80, t=60, b=40),
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        top_risk_chart(
            tables.get("App Component", pd.DataFrame()),
            "App Component",
            "Top Risky Components",
        )
    with col_b:
        top_risk_chart(
            tables.get("Parent App Component", pd.DataFrame()),
            "Parent App Component",
            "Top Risky Parent Components",
        )

with tab2:
    top_risk_chart(
        tables.get("Platform Product Name", pd.DataFrame()),
        "Platform Product Name",
        "Top Risky Platforms",
    )

with tab3:
    top_risk_chart(
        tables.get("Development Stage", pd.DataFrame()),
        "Development Stage",
        "H/C Rate by Development Stage",
        n=10,
    )

st.divider()
st.subheader("Customer Risk Profile")
customer_tbl = tables.get("Customer", pd.DataFrame())
if not customer_tbl.empty:
    top_risk_chart(customer_tbl, "Customer", "H/C Rate by Customer", n=20)
else:
    st.info("No customer data available.")
