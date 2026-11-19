@echo off
title Lancement Jupyter Notebook - Classification Sports CNN
echo ============================================================
echo   Classification Automatique des Sports - CNN
echo   Lancement de Jupyter Notebook...
echo ============================================================
echo.

cd /d "C:\Users\DELL\Desktop\Sport_Classification_CNN"

echo Demarrage de Jupyter Notebook...
echo Le navigateur va s'ouvrir automatiquement.
echo Pour arreter le serveur : appuyez sur CTRL+C dans cette fenetre.
echo.

C:\Users\DELL\AppData\Roaming\Python\Python311\Scripts\jupyter-notebook.exe notebooks/classification_sports.ipynb

pause
