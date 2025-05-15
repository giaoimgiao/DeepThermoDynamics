<#
PowerShell 环境一键安装脚本
使用方法（在项目根目录执行）：
  1. 右键本文件，选择"使用 PowerShell 运行"，或在终端执行：
     powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
  2. 执行完毕后，在当前终端运行：
     . .\.venv\Scripts\Activate.ps1   # 激活虚拟环境
  3. 运行训练脚本：
     python mit_cn.py
#>

Set-StrictMode -Version 3.0

[Console]::OutputEncoding = [Text.UTF8Encoding]::UTF8  # 确保 PowerShell 输出 UTF-8

Write-Host '[INFO] 切换到脚本所在目录...'
Set-Location $PSScriptRoot

$ErrorActionPreference = 'Stop'

# 创建虚拟环境
if (-Not (Test-Path .venv)) {
    Write-Host '[INFO] 创建虚拟环境 .venv ...'
    python -m venv .venv
} else {
    Write-Host '[INFO] 检测到已存在虚拟环境 .venv，跳过创建。'
}

# 激活虚拟环境
Write-Host '[INFO] 激活虚拟环境 ...'
. .\.venv\Scripts\Activate.ps1

# 使用 UTF-8 代码页，避免中文乱码
Write-Host '[INFO] 切换终端到 UTF-8 编码 (65001) ...'
chcp 65001 | Out-Null

# 升级 pip
Write-Host '[INFO] 升级 pip ...'
python -m pip install --upgrade pip

# 安装依赖
Write-Host '[INFO] 安装 requirements.txt 中列出的依赖 ...'
python -m pip install -r requirements.txt

Write-Host '[SUCCESS] 环境安装完成！请执行以下命令开始训练：'
Write-Host '    . .\.venv\Scripts\Activate.ps1'
Write-Host '    python mit_cn.py' 