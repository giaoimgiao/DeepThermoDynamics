#!/usr/bin/env bash
# 一键安装脚本（Linux/macOS）
# 运行：bash setup_env.sh
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "[INFO] 创建虚拟环境 .venv ..."
  python3 -m venv .venv
else
  echo "[INFO] 检测到已存在虚拟环境 .venv，跳过创建。"
fi

# shellcheck source=/dev/null
source .venv/bin/activate

# 避免中文输出乱码
export PYTHONIOENCODING=utf-8

echo "[INFO] 升级 pip ..."
pip install --upgrade pip

echo "[INFO] 安装依赖 ..."
pip install -r requirements.txt

echo "[SUCCESS] 环境安装完成！使用以下命令开始训练："
echo "    source .venv/bin/activate"
echo "    python mit_cn.py" 