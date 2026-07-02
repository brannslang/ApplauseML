import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import shutil
import subprocess
import tempfile
from datetime import datetime

import pandas as pd
import streamlit as st

from config import ROOT, DATA_DIR, EXPECTED_DATA_FILES
from model.predict import reset_cache, get_customers

st.set_page_config(page_title="Data Upload — ApplauseML", page_icon="📤", layout="wide")
st.title("Data Upload")
st.caption(
    "Upload refreshed or new client data and retrain in one step — no command line needed. "
    "Files are matched by name; upload all 8 or just the ones that changed."
)

st.subheader("Expected files")
status_rows = []
for fname in EXPECTED_DATA_FILES:
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
        status_rows.append({"File": fname, "Status": "Present", "Last updated": mtime})
    else:
        status_rows.append({"File": fname, "Status": "Missing", "Last updated": "—"})
st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

st.divider()

uploaded_files = st.file_uploader(
    "Upload data file(s) (.xlsx) — filenames must match one of the names above",
    type=["xlsx"],
    accept_multiple_files=True,
)

matched = []
if uploaded_files:
    canonical_by_lower = {f.lower(): f for f in EXPECTED_DATA_FILES}
    unmatched = []
    for uf in uploaded_files:
        canonical = canonical_by_lower.get(uf.name.lower())
        if canonical:
            matched.append((canonical, uf))
        else:
            unmatched.append(uf.name)

    if unmatched:
        st.warning("Not a recognized filename, ignored: " + ", ".join(unmatched))
    if matched:
        st.success("Recognized: " + ", ".join(c for c, _ in matched))

process = st.button("Save & Retrain", type="primary", disabled=not matched)

if process:
    bad = []
    for canonical, uf in matched:
        uf.seek(0)
        try:
            pd.read_excel(uf, engine="openpyxl", nrows=5)
        except Exception as e:
            bad.append(f"{canonical}: {e}")

    if bad:
        st.error("Upload rejected — these files failed to parse as Excel:\n\n" + "\n".join(bad))
        st.stop()

    backup_dir = tempfile.mkdtemp(prefix="applauseml_upload_backup_")
    backed_up = []
    try:
        for canonical, _ in matched:
            existing = os.path.join(DATA_DIR, canonical)
            if os.path.exists(existing):
                shutil.copy2(existing, os.path.join(backup_dir, canonical))
                backed_up.append(canonical)

        for canonical, uf in matched:
            uf.seek(0)
            with open(os.path.join(DATA_DIR, canonical), "wb") as out:
                out.write(uf.getbuffer())

        with st.spinner("Files saved. Retraining model — this can take a minute..."):
            result = subprocess.run(
                [sys.executable, os.path.join(ROOT, "model", "train.py")],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

        if result.returncode != 0:
            for canonical in backed_up:
                shutil.copy2(os.path.join(backup_dir, canonical), os.path.join(DATA_DIR, canonical))
            st.error(
                "Training failed — uploaded data was rolled back and the previous "
                "model is still active.\n\n```\n" + result.stderr[-3000:] + "\n```"
            )
        else:
            reset_cache()
            new_customers = get_customers()
            st.success(
                f"Model retrained. {len(new_customers)} customers now available: "
                + ", ".join(new_customers)
            )
            with st.expander("Training log"):
                st.code(result.stdout)
    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)
