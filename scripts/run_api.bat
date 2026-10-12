@echo off
echo =====================================================================
echo  Starting DHPRS Backend API (FastAPI)
echo  Swagger UI Documentation will be available at: http://127.0.0.1:8000/docs
echo =====================================================================
cd /d D:\DHPRS
set PYTHONPATH=D:\DHPRS;%PYTHONPATH%
call .venv\Scripts\activate
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
pause
