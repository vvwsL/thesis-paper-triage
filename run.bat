@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [setup] создаю виртуальное окружение и ставлю зависимости...
    py -3 -m venv .venv || python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)

if not exist ".env" (
    echo [setup] .env не найден — копирую из .env.example. Впиши ключи в .env!
    copy ".env.example" ".env" >nul
)

".venv\Scripts\python.exe" -m src.menu
pause
