import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from model.predict import artifacts_exist, get_risk_tables, get_customer_list, get_customer_risk_tables

st.set_page_config(page_title="Risk Dashboard — ApplauseML", page_icon="📊", layout="wide")
st.title("Risk Dashboard")
st.caption("Historical High/Critical bug rates by component, platform, and environment.")

if not artifacts_exist():
    st.error("Model artifacts not found. Run `python model/train.py` first.")
    st.stop()

# Customer filter
customers = get_customer_list()
with st.sidebar:
    st.header("Filter")
    customer_options = ["(All Customers)"] + customers
    selected_customer = st.selectbox("Customer", customer_options)

customer = None if selected_customer == "(All Customers)" else selected_customer
tables = get_customer_risk_tables(customer)
baseline = tables["baseline"]
total = tables["total_bugs"]

if customer:
    st.info(f"Showing data for **{customer}** only.")

m1, m2 = st.columns(2)
m1.metric("H/C Baseline Rate", f"{baseline:.1%}", help="For the selected customer scope")
m2.metric("Bugs in Scope", f"{total:,}")

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
    plot_df["_delta"] = plot_df["hc_rate"] - baseline
    fig = go.Figure(
        go.Bar(
            x=plot_df["hc_rate"],
            y=plot_df[dim_col].astype(str),
            orientation="h",
            marker_color=plot_df["color"],
            text=plot_df["label"],
            textposition="outside",
            customdata=plot_df["_delta"].values.reshape(-1, 1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "H/C Rate: %{x:.1%}<br>"
                f"vs Baseline ({baseline:.1%}): %{{customdata[0]:+.1%}}<br>"
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
    ["By Component", "By Platform", "By Environment", "By Testing Type"]
)

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        risk_bar(
            tables.get("App Component", pd.DataFrame()),
            "App Component",
            "High/Critical Rate by App Component",
        )
    with col_b:
        risk_bar(
            tables.get("Bug Type", pd.DataFrame()),
            "Bug Type",
            "High/Critical Rate by Bug Type",
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
        tables.get("Test Cycle Testing Type", pd.DataFrame()),
        "Test Cycle Testing Type",
        "High/Critical Rate by Testing Type",
    )

st.divider()
st.caption(
    "Red = >50% H/C rate  |  Orange = 35–50%  |  Blue = <35%  |  "
    "Dashed line = baseline for selected scope. Minimum 10 bugs per row."
)
