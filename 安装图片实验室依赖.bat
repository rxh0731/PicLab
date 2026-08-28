@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo 创建 Python 虚拟环境失败，请安装 Python 3.12。
        pause
        exit /b 1
    )
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo 图片实验室依赖安装失败。
    pause
    exit /b 1
)
echo 图片实验室依赖安装完成。
pause

