# 启动 pptbg 本地服务。用法：
#   .\tools\run_local.ps1              直接启动（依赖已装好）
#   .\tools\run_local.ps1 -Install     先装依赖再启动
#   .\tools\run_local.ps1 -Port 8764   换端口
param(
    [switch]$Install,
    [int]$Port = 8763
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $py) { Write-Host '找不到 python，请先安装 Python 3.10 以上版本' -ForegroundColor Red; exit 1 }

if ($Install) {
    Write-Host '[1/2] 安装依赖（首次约 3-8 分钟，取决于网速）' -ForegroundColor Cyan
    & $py.Source -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Write-Host '依赖安装失败，见上面的报错' -ForegroundColor Red; exit 1 }
}

# 权重放在项目内，避免污染系统缓存
$env:HF_HOME = Join-Path $root 'tools\hf'
if (-not (Test-Path (Join-Path $env:HF_HOME 'stable-diffusion-v1-5\model_index.json'))) {
    Write-Host '[提示] 还没下载基座权重，本次会走程序化兜底（无 LoRA 效果）。' -ForegroundColor Yellow
    Write-Host '       想看到完整效果，先跑：python tools\fetch_model.py sd15（约 3.1 GB）'
}

Write-Host "[2/2] 启动服务 http://127.0.0.1:$Port" -ForegroundColor Cyan
Start-Process "http://127.0.0.1:$Port/"
& $py.Source app\server.py --port $Port
