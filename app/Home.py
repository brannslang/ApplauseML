import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

st.set_page_config(
    page_title="ApplauseML — Release Risk Intelligence",
    page_icon="🔍",
    layout="wide",
)

st.title("ApplauseML — Release Risk Intelligence")
st.markdown(
    "Predictive bug severity intelligence to help engineering teams ship with confidence."
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Risk Dashboard")
    st.markdown(
        "Ranked view of historically risky components, platforms, and environments. "
        "No inputs required — always-on signal from historical data."
    )
    if st.button("Open Risk Dashboard →", use_container_width=True):
        st.switch_page("pages/1_Risk_Dashboard.py")

with col2:
    st.subheader("Release Predictor")
    st.markdown(
        "Tell us about your upcoming release and get a predicted High/Critical bug "
        "probability, plus a breakdown of where bugs are most likely to appear."
    )
    if st.button("Open Release Predictor →", use_container_width=True):
        st.switch_page("pages/2_Release_Predictor.py")

with col3:
    st.subheader("Monthly Digest")
    st.markdown(
        "Historical pattern analysis — trend lines, top recurring risk factors, "
        "and period-over-period changes. Refreshed monthly."
    )
    if st.button("Open Monthly Digest →", use_container_width=True):
        st.switch_page("pages/3_Monthly_Digest.py")

st.divider()

st.subheader("Data Upload")
st.markdown(
    "Upload refreshed or new client Excel files and retrain the model — no command line needed."
)
if st.button("Open Data Upload →", use_container_width=True):
    st.switch_page("pages/5_Data_Upload.py")

st.divider()

from model.predict import artifacts_exist
if not artifacts_exist():
    st.warning(
        "Model artifacts not found. Upload data from the **Data Upload** page "
        "to train the model before using the predictor.",
        icon="⚠️",
    )
else:
    st.success("Model is trained and ready.", icon="✅")
