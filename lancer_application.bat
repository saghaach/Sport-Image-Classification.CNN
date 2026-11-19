@echo off
title SportVision CNN - Application Web Streamlit
echo ============================================================
echo   Classification Automatique des Sports - Application Web
echo   Demarrage du serveur Streamlit...
echo ============================================================
echo.

cd /d "C:\Users\DELL\Desktop\Sport_Classification_CNN"

echo Lancement de l'application Streamlit...
echo Le serveur demarre sur http://localhost:8501
echo.

start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --new-window "http://localhost:8501"

python -m streamlit run app.py --server.headless true

pause
