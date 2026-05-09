import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from model.predict import artifacts_exist, get_feature_info, predict_release_risk

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


def cat_options(col: str) -> list:
    opts = categories.get(col, [])
    return ["(not specified)"] + opts


st.subheader("Release Details")
st.markdown("Fill in what you know about the upcoming release. Unknown fields can be left unspecified.")

col1, col2 = st.columns(2)

with col1:
    app_component = st.selectbox(
        "App Component",
        cat_options("App Component"),
        help="The specific component being released or tested.",
    )
    parent_component = st.selectbox(
        "Parent App Component",
        cat_options("Parent App Component"),
        help="High-level product area.",
    )
    platform = st.selectbox(
        "Platform",
        cat_options("Platform Product Name"),
        help="e.g. iOS, Android, Web",
    )
    dev_stage = st.selectbox(
        "Development Stage",
        cat_options("Development Stage"),
        help="Pre-production, production, etc.",
    )

with col2:
    testing_approach = st.selectbox(
        "Testing Approach",
        cat_options("Testing Approach"),
    )
    bug_source_type = st.selectbox(
        "Bug Source Type",
        cat_options("Bug Source Type"),
        help="Structured (scripted) vs exploratory.",
    )
    bug_request_source = st.selectbox(
        "Bug Request Source",
        cat_options("Bug Request Source"),
    )
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


def to_input(val):
    return None if val == "(not specified)" else val


run = st.button("Predict Release Risk", type="primary", use_container_width=False)

if run:
    inputs = {
        "App Component": to_input(app_component),
        "Parent App Component": to_input(parent_component),
        "Platform Product Name": to_input(platform),
        "Development Stage": to_input(dev_stage),
        "Testing Approach": to_input(testing_approach),
        "Bug Source Type": to_input(bug_source_type),
        "Bug Request Source": to_input(bug_request_source),
        "Test Cycle Duration Activation to Lock/Close/Today": cycle_duration or None,
        "Bug Rate Amount": bug_rate or None,
    }

    result = predict_release_risk(inputs)
    risk = result["risk_score"]
    baseline = result["baseline"]
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
    st.subheader("Where Bugs Are Most Likely to Arise")

    comp_tbl = result["component_breakdown"]
    if not comp_tbl.empty:
        selected_parent = to_input(parent_component)
        display_tbl = comp_tbl.copy()

        st.markdown("**Component Risk Breakdown** (historical H/C rate, all data)")
        top20 = display_tbl.head(20).copy()
        top20["hc_rate_pct"] = top20["hc_rate"].map("{:.1%}".format)
        top20["vs_baseline"] = top20["vs_baseline"].map(
            lambda x: f"+{x:.1%}" if x >= 0 else f"{x:.1%}"
        )

        if to_input(app_component):
            top20["Selected"] = top20["App Component"] == to_input(app_component)
        else:
            top20["Selected"] = False

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
            x=baseline,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Baseline {baseline:.1%}",
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
        st.info("Component breakdown not available.")
