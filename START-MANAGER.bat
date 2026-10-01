@echo off
rem Strata Manager - an optional local GUI for managing this Strata install (models, context, Vision,
rem Low RAM, custom GGUF, Start/Stop/Restart).  It uses the private environment START-HERE.bat makes,
rem so run START-HERE.bat once first if .venv is missing.  It does not change how Strata starts.
setlocal
title Strata Manager
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo  The environment .venv is missing - run START-HERE.bat once first (it installs the model too).
  pause
  exit /b 1
)
".venv\Scripts\python.exe" gui\manager.py %*
if errorlevel 1 pause
exit /b