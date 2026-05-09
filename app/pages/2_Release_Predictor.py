import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from model.predict import (
    artifacts_exist, get_feature_info, get_customer_list, predict_release_risk,
)

st.set_page_config(page_title="Release Predictor — ApplauseML", page_icon="🎯", layout="wide")
st.title("Release Predictor")
st.caption(
    "Describe your upcoming release and get a predicted High/Critical bug risk score, "
    "plus a breakdown of where bugs are most likely to surface."
)

if not artifacts_exist():
    st.error("Model artifacts not found. Run `python model/train.py` first.")
    st.stop()

feature_info = get_feature_info()
categories = feature_info["categories"]
customers = get_customer_list()


def cat_options(col: str) -> list:
    opts = categories.get(col, [])
    return ["(not specified)"] + opts


def to_input(val):
    return None if val == "(not specified)" else val


st.subheader("Release Details")
st.markdown("Fill in what you know about the upcoming release. Unknown fields can be left unspecified.")

# Customer is the first field and scopes all downstream breakdown data
customer_options = ["(not specified)"] + customers
customer_val = st.selectbox(
    "Customer",
    customer_options,
    help="Selecting a customer scopes the risk breakdown to that customer's historical data.",
)
customer = to_input(customer_val)
if customer:
    st.caption(f"Risk breakdown will use **{customer}** data only.")

col1, col2 = st.columns(2)

with col1:
    testing_type = st.selectbox(
        "Testing Type",
        cat_options("Test Cycle Testing Type"),
        help="Functional, Accessibility, Security, or Usability.",
    )
    platform = st.selectbox(
        "Platform",
        cat_options("Platform Product Name"),
        help="e.g. iOS, Android, Web",
    )

with col2:
    dev_stage = st.selectbox(
        "Development Stage",
        cat_options("Development Stage"),
        help="Pre-production, production, etc.",
    )
    bug_source_type = st.selectbox(
        "Bug Source Type",
        cat_options("Bug Source Type"),
        help="Structured (scripted) vs exploratory.",
    )

col3, col4 = st.columns(2)

with col3:
    app_component = st.selectbox(
        "App Component",
        cat_options("App Component"),
        help="The specific component being released or tested.",
    )
    bug_type = st.selectbox(
        "Bug Type",
        cat_options("Bug Type"),
        help="Category of bug expected (Functional, Visual, etc.).",
    )

with col4:
    cycle_duration = st.number_input(
        "Estimated Cycle Duration (days)",
        min_value=0,
        max_value=90,
        value=0,
        help="Leave at 0 if unknown.",
    )
    bug_rate = st.number_input(
        "Tester Pay Rate (if known)",
        min_value=0.0,
        value=0.0,
        step=0.5,
        help="Leave at 0 if unknown.",
    )

run = st.button("Predict Release Risk", type="primary", use_container_width=False)

if run:
    inputs = {
        "Customer": customer,
        "Test Cycle Testing Type": to_input(testing_type),
        "Platform Product Name": to_input(platform),
        "Development Stage": to_input(dev_stage),
        "Bug Source Type": to_input(bug_source_type),
        "App Component": to_input(app_component),
        "Bug Type": to_input(bug_type),
        "Test Cycle Duration Activation to Lock/Close/Today": cycle_duration or None,
        "Bug Rate Amount": bug_rate or None,
    }

    result = predict_release_risk(inputs, customer=customer)
    risk = result["risk_score"]
    baseline = result["baseline"]
    view_baseline = result["view_baseline"]
    delta = result["risk_delta"]
    label = result["risk_label"]
    color = result["risk_color"]

    st.divider()
    st.subheader("Prediction Report")

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Predicted H/C Risk",
        f"{risk:.1%}",
        delta=f"{delta:+.1%} vs baseline",
        delta_color="inverse",
    )
    m2.metric("Overall Baseline", f"{baseline:.1%}")
    m3.metric("Risk Level", label)

    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk * 100,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 35], "color": "#e8f5e9"},
                    {"range": [35, 50], "color": "#fff3e0"},
                    {"range": [50, 100], "color": "#ffebee"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": baseline * 100,
                },
            },
            title={"text": "High/Critical Bug Probability"},
        )
    )
    fig_gauge.update_layout(height=280, margin=dict(t=30, b=0, l=30, r=30))
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.caption(f"Black marker on gauge = overall baseline ({baseline:.1%})")

    st.divider()
    scope_label = f"**{customer}**" if customer else "all customers"
    st.subheader("Where Bugs Are Most Likely to Arise")
    st.caption(f"Component risk breakdown from {scope_label} historical data.")

    comp_tbl = result["component_breakdown"]
    if not comp_tbl.empty:
        display_tbl = comp_tbl.copy()
        display_tbl["vs_baseline"] = display_tbl["hc_rate"] - view_baseline

        fig_comp = px.bar(
            display_tbl.head(20),
            x="hc_rate",
            y="App Component",
            orientation="h",
            color="hc_rate",
            color_continuous_scale=["#2ca02c", "#ff7f0e", "#d62728"],
            range_color=[0, 1],
            text=display_tbl.head(20)["hc_rate"].map("{:.0%}".format),
            hover_data={"n_bugs": True, "hc_rate": ":.1%", "vs_baseline": ":.1%"},
            labels={"hc_rate": "H/C Rate", "App Component": ""},
        )
        fig_comp.add_vline(
            x=view_baseline,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Baseline {view_baseline:.1%}",
        )
        fig_comp.update_layout(
            height=max(350, len(display_tbl.head(20)) * 28 + 80),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(tickformat=".0%"),
            coloraxis_showscale=False,
            margin=dict(l=10, r=80, t=20, b=40),
            plot_bgcolor="white",
        )
        fig_comp.update_traces(textposition="outside")
        st.plotly_chart(fig_comp, use_container_width=True)

        with st.expander("View as table"):
            st.dataframe(
                display_tbl[["App Component", "hc_rate", "n_bugs", "n_hc", "vs_baseline"]]
                .rename(columns={
                    "hc_rate": "H/C Rate",
                    "n_bugs": "Total Bugs",
                    "n_hc": "H/C Bugs",
                    "vs_baseline": "vs Baseline",
                })
                .style.format({
                    "H/C Rate": "{:.1%}",
                    "vs Baseline": "{:+.1%}",
                }),
                use_container_width=True,
            )
    else:
        st.info("Component breakdown not available for the selected scope.")
