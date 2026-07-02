import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st


def require_customer() -> str:
    """
    Render a required Customer selector in the sidebar.
    Persists selection in session_state across page navigations.
    Calls st.stop() if no customer is selected, so pages can call this
    at the top and rely on the returned value being valid.
    """
    from model.predict import artifacts_exist, get_customers

    if not artifacts_exist():
        st.error("Model artifacts not found. Upload data from the **Data Upload** page to train the model.")
        st.stop()

    customers = get_customers()
    if not customers:
        st.error("No customers found in training data.")
        st.stop()

    with st.sidebar:
        st.markdown("### Customer")
        current = st.session_state.get("selected_customer")
        idx = (customers.index(current) + 1) if current in customers else 0

        selection = st.selectbox(
            "Customer",
            options=["— Select a customer —"] + customers,
            index=idx,
            label_visibility="collapsed",
            key="customer_selector",
        )

        if selection == "— Select a customer —":
            st.session_state["selected_customer"] = None
        else:
            st.session_state["selected_customer"] = selection

        st.divider()

    customer = st.session_state.get("selected_customer")
    if not customer:
        st.info("Select a customer from the sidebar to view data.", icon="👈")
        st.stop()

    return customer
