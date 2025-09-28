@echo off
echo Starting Real-ESRGAN Service for Suwayomi...
echo.
call realesrgan_env\Scripts\activate.bat
python memory_optimized_real_esrgan_app.py
pause
