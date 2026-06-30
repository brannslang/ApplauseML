#!/bin/bash
# ApplauseML Launcher — double-click this file to start the app.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# If already running, just open the browser.
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "ApplauseML is already running — opening browser..."
    open "http://localhost:8501"
    exit 0
fi

echo "Starting ApplauseML at http://localhost:8501 ..."
echo "Close this window to stop the server."
echo ""
python3 -m streamlit run app/Home.py
