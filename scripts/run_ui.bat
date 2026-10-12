@echo off
echo =====================================================================
echo  Starting DHPRS Healthcare Web Dashboard (Streamlit)
echo  Web UI will open automatically at: http://localhost:8501
echo =====================================================================
cd /d D:\DHPRS
set PYTHONPATH=D:\DHPRS;%PYTHONPATH%
call .venv\Scripts\activate
streamlit run ui/app.py
pause
