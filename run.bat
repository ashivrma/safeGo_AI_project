@echo off
echo Starting SafeGo AI...
echo.

:: Terminal 1 - Backend
start cmd /k "cd /d C:\Users\ashi8\OneDrive\Desktop\safego_ai && venv\Scripts\activate && echo Starting Backend... && python app.py"

:: Wait 3 seconds for backend to start before launching camera
timeout /t 3 /nobreak > nul

:: Terminal 2 - Camera Detection
start cmd /k "cd /d C:\Users\ashi8\OneDrive\Desktop\safego_ai && venv\Scripts\activate && echo Starting Camera Detection... && python detect.py --camera 0 --location "My Street, City" --police "police@email.com""

echo Both windows launched!
echo - Backend running at http://localhost:5000
echo - Camera detection window will open shortly
pause
