@echo off
setlocal

echo ======================================================================
echo SIH V6.1 - BITCOIN FORENSICS PIPELINE
echo ======================================================================

echo.
echo [1/10] Blockchain - P2P Correlation
python src\correlate_csv.py
if errorlevel 1 goto :error

echo.
echo [2/10] Group B Correlation Handoff
python src\build_group_b_handoff.py
if errorlevel 1 goto :error

echo.
echo [3/10] Blockchain Graph Construction
python src\build_graph.py
if errorlevel 1 goto :error

echo.
echo [4/10] Entity Resolution
python src\resolve_entities.py
if errorlevel 1 goto :error

echo.
echo [5/10] V3 Base Entity Feature Engineering
python src\build_features_v3.py
if errorlevel 1 goto :error

echo.
echo [6/10] Ground-Truth Entity Resolution
python src\resolve_ground_truth_v3.py
if errorlevel 1 goto :error

echo.
echo [7/10] V6 Behavioral Feature Engineering
python src\build_behavior_features_v6.py
if errorlevel 1 goto :error

echo.
echo [8/10] V6.1 Temporal / Cross-Entity Feature Engineering
python src\build_behavior_features_v61.py
if errorlevel 1 goto :error

echo.
echo [9/10] Isolation Forest Novelty Model
python src\train_anomaly_model_v5.py
if errorlevel 1 goto :error

echo.
echo [10/10] Final V6.1 Forensic Detector
python src\final_detector_v61.py
if errorlevel 1 goto :error

echo.
echo ======================================================================
echo PIPELINE COMPLETE
echo ======================================================================
echo.
echo Final alerts:
echo     outputs\final_entity_alerts_v61.csv
echo.
exit /b 0


:error
echo.
echo ======================================================================
echo PIPELINE FAILED
echo ======================================================================
echo.
echo A pipeline stage returned an error.
exit /b 1

