@echo off
chcp 65001 > nul
echo ===================================
echo 深度学习训练可视化平台启动脚本
echo ===================================
echo.

echo [1/3] 激活虚拟环境...
if not exist .venv (
  echo 未检测到虚拟环境，请先运行 setup_env.ps1 安装环境
  echo 可使用: powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
  pause
  exit /b
)

call .venv\Scripts\activate.bat

echo [2/3] 检查依赖...
python -c "import streamlit" > nul 2>&1
if %errorlevel% neq 0 (
  echo 安装缺失依赖...
  pip install -r requirements.txt
)

echo [3/3] 启动Web应用...
echo 应用启动后，请访问浏览器: http://localhost:8501
echo 启动中，请稍候...
echo.
echo 按 Ctrl+C 可以停止应用
echo ===================================

streamlit run app.py

pause 