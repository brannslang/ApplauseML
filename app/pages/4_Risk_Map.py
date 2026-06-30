import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from model.predict import get_bubble_data
from app.utils import require_customer

st.set_page_config(
    page_title="Risk Map — ApplauseML",
    page_icon="🗺️",
    layout="wide",
)

st.title("Interactive Risk Map")
st.caption(
    "Each bubble is a Device OS × App Component combination. "
    "X-axis = historical test case failure rate. "
    "Y-axis = ML-predicted High/Critical bug probability. "
    "Bubble size = risk score (larger = higher risk)."
)

customer  = require_customer()
all_bubbles = get_bubble_data()

if all_bubbles.empty:
    st.warning(
        "Bubble data not yet generated. "
        "Retrain the model with `python model/train.py` to produce it.",
        icon="⚠️",
    )
    st.stop()

if "Customer" in all_bubbles.columns:
    bubble_df = all_bubbles[all_bubbles["Customer"] == customer].copy()
else:
    bubble_df = all_bubbles.copy()

if bubble_df.empty:
    st.warning(f"No bubble data found for **{customer}**. The customer may not appear in the device-level datasets.")
    st.stop()

st.subheader(f"Customer: {customer}")

CLUSTER_COLORS = {
    "Critical Hazard": "#d62728",
    "Nuisance Zone":   "#ff7f0e",
    "Stable":          "#2ca02c",
}

st.divider()

# ── Filters ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    platforms = sorted(bubble_df["Platform Product Name"].dropna().unique().tolist())
    selected_platforms = st.multiselect(
        "Platform",
        options=platforms,
        default=platforms,
    )

    os_versions = sorted(bubble_df["Mobile OS Major Version"].dropna().unique().tolist())
    selected_os = st.multiselect(
        "OS Version",
        options=os_versions,
        default=os_versions,
    )

    min_bugs = st.slider(
        "Minimum bug count (hide sparse bubbles)",
        min_value=1,
        max_value=int(bubble_df["n_bugs"].quantile(0.90)),
        value=5,
        step=1,
    )

    clusters = sorted(bubble_df["cluster"].dropna().unique().tolist())
    selected_clusters = st.multiselect(
        "Risk Cluster",
        options=clusters,
        default=clusters,
    )

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = bubble_df[
    bubble_df["Platform Product Name"].isin(selected_platforms)
    & bubble_df["Mobile OS Major Version"].isin(selected_os)
    & bubble_df["n_bugs"].ge(min_bugs)
    & bubble_df["cluster"].isin(selected_clusters)
].copy()

if filtered.empty:
    st.info("No data matches the current filters. Adjust the sidebar filters.")
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Bubbles shown", f"{len(filtered):,}")
col2.metric(
    "Critical Hazard",
    f"{(filtered['cluster'] == 'Critical Hazard').sum()}",
)
col3.metric(
    "Nuisance Zone",
    f"{(filtered['cluster'] == 'Nuisance Zone').sum()}",
)
col4.metric(
    "Stable",
    f"{(filtered['cluster'] == 'Stable').sum()}",
)

st.divider()

# ── Build bubble chart ────────────────────────────────────────────────────────
SIZE_SCALE = 60  # max pixel diameter


def build_hover(row) -> str:
    lines = [
        f"<b>{row['App Component Name']}</b>",
        f"Platform: {row['Platform Product Name']}",
        f"OS: {row['Mobile OS Major Version']}",
        f"<br>ML Risk Score: {row['ml_prob']:.1%}",
        f"Test Failure Rate: {row['failure_rate']:.1%}",
        f"Total Bugs: {int(row['n_bugs'])}",
        f"Cluster: {row['cluster']}",
    ]
    sev_parts = []
    for col, label in [("n_critical", "Critical"), ("n_high", "High"),
                       ("n_medium", "Medium"), ("n_low", "Low")]:
        if col in row.index and row[col] > 0:
            sev_parts.append(f"{label}: {int(row[col])}")
    if sev_parts:
        lines.append("<br>Severity: " + " | ".join(sev_parts))
    return "<br>".join(lines)


fig = go.Figure()

for cluster_name, color in CLUSTER_COLORS.items():
    subset = filtered[filtered["cluster"] == cluster_name]
    if subset.empty:
        continue

    max_prob = filtered["ml_prob"].max() or 1.0
    sizes = (subset["ml_prob"] / max_prob * SIZE_SCALE).clip(lower=6)

    hover_texts = [build_hover(row) for _, row in subset.iterrows()]

    fig.add_trace(
        go.Scatter(
            x=subset["failure_rate"],
            y=subset["ml_prob"],
            mode="markers",
            name=cluster_name,
            marker=dict(
                size=sizes,
                color=color,
                opacity=0.75,
                line=dict(width=1, color="white"),
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            customdata=subset[["App Component Name", "Platform Product Name",
                               "Mobile OS Major Version"]].values,
        )
    )

# Quadrant reference lines
x_mid = filtered["failure_rate"].median()
y_mid = filtered["ml_prob"].median()

fig.add_hline(
    y=y_mid,
    line_dash="dot",
    line_color="gray",
    opacity=0.5,
    annotation_text=f"Median risk ({y_mid:.0%})",
    annotation_position="top left",
    annotation_font_size=11,
)
fig.add_vline(
    x=x_mid,
    line_dash="dot",
    line_color="gray",
    opacity=0.5,
    annotation_text=f"Median failure rate ({x_mid:.0%})",
    annotation_position="top right",
    annotation_font_size=11,
)

fig.update_layout(
    xaxis=dict(
        title="Test Case Failure Rate",
        tickformat=".0%",
        range=[-0.02, min(1.05, filtered["failure_rate"].max() * 1.15)],
        gridcolor="#eeeeee",
    ),
    yaxis=dict(
        title="ML Risk Score — P(High/Critical Bug)",
        tickformat=".0%",
        range=[-0.02, min(1.05, filtered["ml_prob"].max() * 1.15)],
        gridcolor="#eeeeee",
    ),
    legend=dict(
        title="Risk Cluster",
        orientation="h",
        yanchor="bottom",
        y=-0.18,
        xanchor="center",
        x=0.5,
    ),
    height=620,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(t=30, b=80, l=60, r=40),
    hovermode="closest",
)

st.plotly_chart(fig, use_container_width=True)

# ── Quadrant labels ───────────────────────────────────────────────────────────
q1, q2, q3, q4 = st.columns(4)
q1.markdown("↖ **Low failure / Low risk** — generally healthy")
q2.markdown("↗ **High failure / Low risk** — test coverage issues, low severity bugs")
q3.markdown("↙ **Low failure / High risk** — rare but severe when bugs appear")
q4.markdown("↘ **High failure / High risk** — Critical Hazard zone, prioritise immediately")

st.divider()

# ── Detail table ──────────────────────────────────────────────────────────────
with st.expander("View underlying data table"):
    display_cols = [
        "App Component Name",
        "Platform Product Name",
        "Mobile OS Major Version",
        "cluster",
        "ml_prob",
        "failure_rate",
        "n_bugs",
    ]
    for col in ["n_critical", "n_high", "n_medium", "n_low"]:
        if col in filtered.columns:
            display_cols.append(col)

    st.dataframe(
        filtered[display_cols]
        .rename(columns={
            "App Component Name":        "Component",
            "Platform Product Name":     "Platform",
            "Mobile OS Major Version":   "OS Version",
            "cluster":                   "Cluster",
            "ml_prob":                   "ML Risk Score",
            "failure_rate":              "Failure Rate",
            "n_bugs":                    "Bug Count",
            "n_critical":                "Critical",
            "n_high":                    "High",
            "n_medium":                  "Medium",
            "n_low":                     "Low",
        })
        .sort_values("ML Risk Score", ascending=False)
        .style.format({
            "ML Risk Score": "{:.1%}",
            "Failure Rate":  "{:.1%}",
        }),
        use_container_width=True,
    )
