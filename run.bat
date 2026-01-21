@echo off
echo ========================================
echo Advanced Factorization Function Search System
echo ========================================

REM Check dependencies
echo Checking dependencies...
python -c "import sympy, math" || (echo Error: Missing Python libraries && exit /b 1)

REM Create backup directory
if not exist "backups" mkdir backups

REM Run search
echo Starting search on 75 functions...
python main.py --search --functions 75 --time-limit 60

REM Create backup
if exist "factorization_research_v2.db" (
    copy factorization_research_v2.db "backups\backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.db"
    echo Backup saved
)

REM Show best results
echo.
echo ========================================
echo Top 10 Functions:
echo ========================================
python main.py --best --limit 10 --min-score 0.3

echo.
echo ✅ Execution completed successfully!
pause