@echo off
set "CACHE_DIR=%USERPROFILE%\.scispacy"
if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%"
cd /d "%~dp0"

docker run --rm ^
  --shm-size=8g ^
  -v "%CD%:/data" ^
  -v "%CACHE_DIR%:/root/.scispacy" ^
  -v "%CACHE_DIR%:/root/.cache" ^
  -v "%CACHE_DIR%:/tmp" ^
  -e PYTHONUNBUFFERED=1 ^
  -e OMP_NUM_THREADS=1 ^
  -e MKL_NUM_THREADS=1 ^
  note-umls:latest ^
  python mapConditions.py ^
  --in /data/Note_terms.csv ^
  --out /data/output_note_level_terms.csv ^
  --verbose

echo.
echo Done. Output: %CD%\output_note_level_terms.csv
pause
