@echo off
REM One-command setup + launch (Windows). Usage: run.bat
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install -q -r requirements.txt
streamlit run app.py
