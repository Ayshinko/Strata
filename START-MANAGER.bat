@echo off
rem Strata Manager - an optional local GUI for managing this Strata install (models, context, Vision,
rem Low RAM, custom GGUF, Start/Stop/Restart) as ONE unified web page: Manager / Chat / Monitor / About.
rem
rem This launcher opens NO console window: it runs the Manager with pythonw.exe (Windows GUI mode) and
rem returns immediately, so the browser page is the only thing the user sees.  The Manager's own output
rem (or a startup error) goes to logs\manager.log instead of a terminal.  Nothing is changed in how
rem Strata itself starts; START-HERE.bat stays the way to install/update the model.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo  The environment .venv is missing - run START-HERE.bat once first (it installs the model too).
  pause
  exit /b 1
)
if not exist "logs" mkdir "logs"
rem pythonw: no console, ever.  start: the cmd window itself goes away immediately (no persistent black
rem terminal the user could close by accident and kill the Manager).  Output -> logs\manager.log.
start "" /min ".venv\Scripts\pythonw.exe" gui\manager.py %* >> "logs\manager.log" 2>&1
exit /b 0