@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Loi cai dat thu vien. Kiem tra Python va pip.
    pause
    exit /b 1
)
python run_svd_experiment.py --input "USC_SIPI_Dataset" --output "SVD_Results"
echo.
echo Da chay xong. Mo thu muc SVD_Results de xem ket qua.
pause
