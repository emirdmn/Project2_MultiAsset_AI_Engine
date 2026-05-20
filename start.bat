@echo off
title Quantum Fin-AI Terminal Bootloader
color 0A
cls

echo ===================================================
echo   LAUNCHING QUANTUM MULTI-ASSET AI ENGINE
echo   Designed by Lead Engineer Emir Duman
echo ===================================================
echo.
echo [INFO] Checking and verifying system dependencies...
echo.

:: Windows'un gizli Python yerel adresini bulur
set LOCAL_PATH="%USERPROFILE%\AppData\Local\Programs\Python"
set PYTHON_EXE=python

if exist %LOCAL_PATH% (
    cd /d %LOCAL_PATH%
    for /d %%i in (Python*) do (
        if exist "%%i\python.exe" (
            set PYTHON_EXE="%USERPROFILE%\AppData\Local\Programs\Python\%%i\python.exe"
            goto core_launch
        )
    )
)

:core_launch
:: Klasöre geri dönüyoruz
cd /d "%~dp0"

echo [INFO] Target Environment Vector: %PYTHON_EXE%
echo [INFO] Upgrading and checking mandatory libraries (Streamlit, Pandas, Numpy, Scikit-Learn)...
echo [INFO] This might take a few seconds if running for the first time...
echo.

:: 🚀 İLK DEFA İNDİRECEK PC'LER İÇİN OTOMATİK KÜTÜPHANE YÜKLEME ZIRHI
%PYTHON_EXE% -m pip install --upgrade pip --quiet
%PYTHON_EXE% -m pip install streamlit pandas numpy scikit-learn --quiet

echo.
echo [SUCCESS] All libraries verified successfully!
echo [INFO] Launching Streamlit web interface...
echo.

:: Paneli sorunsuz başlatır
%PYTHON_EXE% -m streamlit run "dashboard.py"

pause