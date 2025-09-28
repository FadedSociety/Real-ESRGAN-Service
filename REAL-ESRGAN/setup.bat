@echo off
echo Real-ESRGAN Setup Script
echo ========================
echo.

echo [1/3] Running Python setup...
python simple_setup.py

echo.
echo [2/3] Setup completed!
echo.

echo [3/3] Checking if service can start...
echo Press any key to test the service startup...
pause > nul

echo.
echo Testing service startup...
python memory_optimized_real_esrgan_app.py

echo.
echo Setup and test completed!
echo Press any key to exit...
pause > nul