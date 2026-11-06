@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "PY=C:\Users\hua\AppData\Local\Programs\Python\Python310\python.exe"
if not exist "%PY%" set "PY=python"

echo 正在启动膳食识别上位机...
"%PY%" "%~dp0上位机.py"
if errorlevel 1 (
  echo.
  echo 启动失败。若提示缺少 PySide6，请执行：
  echo   "%PY%" -m pip install PySide6-Essentials -i https://pypi.tuna.tsinghua.edu.cn/simple
  pause
)
