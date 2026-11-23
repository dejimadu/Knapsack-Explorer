#!/usr/bin/env bash
# One-command setup + launch (macOS / Linux). Usage: ./run.sh
set -e
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
streamlit run app.py
