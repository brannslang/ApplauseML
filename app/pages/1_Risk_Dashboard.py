import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from model.predict import (
    artifacts_exist,
    get_customer_risk_tables,
    get_feature_importances,
    get_graph_network_data,
)
from app.utils import require_customer

st.set_page_config(page_title="Risk Dashboard — ApplauseML", page_icon="📊", layout="wide")
st.title("Risk Dashboard")
st.caption("Historical High/Critical bug rates by component, platform, and environment.")

customer = require_customer()
tables   = get_customer_risk_tables(customer)

if not tables:
    st.warning(f"No training data found for **{customer}**.")
    st.stop()

st.subheader(f"Customer: {customer}")
baseline = tables["baseline"]
total    = tables["total_bugs"]

m1, m2 = st.columns(2)
m1.metric("H/C Baseline Rate", f"{baseline:.1%}")
m2.metric("Total Bugs", f"{total:,}")

st.divider()

FLAG_LABELS = {
    "text_flag_crash":          "Crash / Freeze / Hang",
    "text_flag_data_integrity": "Data Integrity Issues",
    "text_flag_error":          "Error / Exception",
    "text_flag_security":       "Security / Bypass",
    "text_flag_visibility":     "Blank / Broken UI",
    "text_flag_performance":    "Performance / Timeout",
    "text_flag_access":         "Auth / Login / Permissions",
}

CORE_LABELS = {
    "App Component":             "App Component",
    "Parent App Component":      "Parent Component",
    "Platform Product Name":     "Platform",
    "Development Stage":         "Development Stage",
    "Bug Request Source":        "Bug Request Source",
    "Bug Source Type":           "Bug Source Type",
    "Testing Approach":          "Testing Approach",
    "Bug Rate Amount":           "Tester Pay Rate",
    "Test Cycle Duration Activation to Lock/Close/Today": "Cycle Duration (days)",
}

GRAPH_LABELS = {
    "graph_comp_pagerank":     "Component PageRank",
    "graph_comp_degree":       "Component Degree Centrality",
    "graph_comp_clustering":   "Component Clustering",
    "graph_platform_pagerank": "Platform PageRank",
    "graph_customer_pagerank": "Customer PageRank",
}


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


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "By Component",
    "By Platform",
    "By Environment",
    "By Testing Approach",
    "Feature Importance",
    "Risk Network",
])

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

with tab5:
    st.subheader("What Drives the Prediction")
    st.caption("Feature importances are model-wide across all customers, not scoped to the selection above.")
    st.caption(
        "Relative importance of each feature group and top individual features "
        "in the trained Random Forest classifier."
    )

    imp_df = get_feature_importances()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**By Feature Group**")
        group_df = (
            imp_df.groupby("group")["importance"]
            .sum()
            .reset_index()
            .sort_values("importance", ascending=True)
        )
        group_df["pct"] = group_df["importance"] / group_df["importance"].sum()
        fig_group = px.bar(
            group_df,
            x="importance",
            y="group",
            orientation="h",
            text=group_df["pct"].map("{:.1%}".format),
            color="importance",
            color_continuous_scale=["#aec7e8", "#1f77b4"],
            labels={"importance": "Total Importance", "group": ""},
        )
        fig_group.update_layout(
            height=320,
            coloraxis_showscale=False,
            plot_bgcolor="white",
            margin=dict(l=10, r=80, t=20, b=40),
            xaxis=dict(showticklabels=False),
        )
        fig_group.update_traces(textposition="outside")
        st.plotly_chart(fig_group, use_container_width=True)

    with col_b:
        st.markdown("**Top 20 Individual Features**")
        # Exclude SVD/embedding/NMF components — aggregate groups only for individual view
        indiv_df = imp_df[
            ~imp_df["feature"].str.startswith(("text_svd_", "text_emb_", "nmf_factor_"))
        ].head(20).copy()

        def display_name(f):
            if f in FLAG_LABELS:    return FLAG_LABELS[f]
            if f in CORE_LABELS:    return CORE_LABELS[f]
            if f in GRAPH_LABELS:   return GRAPH_LABELS[f]
            return f

        indiv_df["label"] = indiv_df["feature"].apply(display_name)
        indiv_df = indiv_df.sort_values("importance", ascending=True)

        fig_indiv = px.bar(
            indiv_df,
            x="importance",
            y="label",
            orientation="h",
            color="group",
            color_discrete_map={
                "Core Features":              "#1f77b4",
                "Text: Keyword Flags":        "#ff7f0e",
                "Graph: Network Metrics":     "#2ca02c",
                "Text: TF-IDF Topics":        "#9467bd",
                "Text: Semantic Embeddings":  "#8c564b",
                "NMF: Latent Risk Archetypes":"#e377c2",
            },
            labels={"importance": "Importance", "label": "", "group": "Feature Type"},
        )
        fig_indiv.update_layout(
            height=max(320, len(indiv_df) * 22 + 80),
            plot_bgcolor="white",
            margin=dict(l=10, r=40, t=20, b=40),
            xaxis=dict(showticklabels=False),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig_indiv, use_container_width=True)

with tab6:
    st.subheader("Risk Network — Entity Connectivity")
    st.caption("Network graph is model-wide across all customers, not scoped to the selection above.")
    st.caption(
        "Each bubble is an entity (component, platform, or customer). "
        "PageRank reflects structural risk connectivity — entities that "
        "co-occur with many other high-severity entities score higher, "
        "independent of their raw H/C rate."
    )

    net_df = get_graph_network_data()

    if net_df.empty:
        st.info("Graph network data not available. Retrain to generate.")
    else:
        top_n = st.slider("Show top N nodes by PageRank", 10, min(100, len(net_df)), 50)
        plot_df = net_df.head(top_n).copy()
        plot_df["size"] = (
            (plot_df["pagerank"] - plot_df["pagerank"].min())
            / (plot_df["pagerank"].max() - plot_df["pagerank"].min() + 1e-9)
            * 30 + 8
        )

        fig_net = px.scatter(
            plot_df,
            x="degree_centrality",
            y="pagerank",
            color="entity_type",
            size="size",
            size_max=40,
            hover_name="entity_name",
            hover_data={
                "pagerank":          ":.4f",
                "degree_centrality": ":.3f",
                "clustering":        ":.3f",
                "size":              False,
            },
            text=plot_df.apply(
                lambda r: r["entity_name"] if r["pagerank"] > plot_df["pagerank"].quantile(0.75) else "",
                axis=1,
            ),
            color_discrete_map={
                "App Component":         "#1f77b4",
                "Platform Product Name": "#ff7f0e",
                "Customer":              "#2ca02c",
                "Development Stage":     "#9467bd",
                "Testing Approach":      "#8c564b",
            },
            labels={
                "degree_centrality": "Degree Centrality (# of connections)",
                "pagerank":          "PageRank (structural risk importance)",
                "entity_type":       "Entity Type",
            },
        )
        fig_net.update_traces(textposition="top center", textfont_size=10)
        fig_net.update_layout(
            height=550,
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=20, b=60),
        )
        st.plotly_chart(fig_net, use_container_width=True)
        st.caption(
            "Top-right = high PageRank + high degree = most structurally risky entities. "
            "Labels shown for top 25% by PageRank. Bubble size scaled by PageRank."
        )

st.divider()
st.caption(
    "Red = >50% H/C rate  |  Orange = 35–50%  |  Blue = <35%  |  "
    "Dashed line = overall baseline. Minimum 10 bugs per row."
)
