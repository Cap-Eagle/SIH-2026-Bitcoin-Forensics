@echo off
setlocal

title SIH V6.1 Bitcoin Forensics

echo ======================================================================
echo SIH V6.1 - COMPLETE BITCOIN FORENSICS SYSTEM
echo ======================================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found.
    echo Install Python and make sure it is available in PATH.
    pause
    exit /b 1
)

echo Python:
python --version
echo.

if not exist requirements.txt (
    echo ERROR: requirements.txt not found.
    pause
    exit /b 1
)

echo Checking dataset...
echo.

set NEED_DATA=0

if not exist data\raw\blockchain_transactions.csv set NEED_DATA=1
if not exist data\raw\network_events.csv set NEED_DATA=1
if not exist data\raw\ip_enrichment.csv set NEED_DATA=1
if not exist data\ground_truth\ground_truth.csv set NEED_DATA=1
if not exist data\ground_truth\entity_wallet_map.csv set NEED_DATA=1

if "%NEED_DATA%"=="1" (
    echo Dataset not found.
    echo Generating synthetic SIH dataset...
    echo.

    python generate_dataset.py

    if errorlevel 1 (
        echo.
        echo ERROR: Dataset generation failed.
        pause
        exit /b 1
    )
) else (
    echo Existing dataset found.
)

echo.
echo ======================================================================
echo VALIDATING INPUT DATA
echo ======================================================================

python src\ingest_data.py --validate-existing

if errorlevel 1 (
    echo.
    echo ERROR: Input validation failed.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo RUNNING FORENSIC PIPELINE
echo ======================================================================

call run_pipeline.bat

if errorlevel 1 (
    echo.
    echo ERROR: Pipeline failed.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo VALIDATING PROJECT
echo ======================================================================

python src\validate_project.py

if errorlevel 1 (
    echo.
    echo ERROR: Project validation failed.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo SIH V6.1 COMPLETE
echo ======================================================================
echo.
echo Final alerts:
echo     outputs\final_entity_alerts_v61.csv
echo.
echo Model:
echo     models\final_detector_v61.joblib
echo.
echo Launch dashboard with:
echo.
echo     streamlit run app\dashboard.py
echo.

pause
