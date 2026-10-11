$ErrorActionPreference = 'Stop'
$experimentRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $experimentRoot
$hardwareRoot = Join-Path $workspaceRoot '硬件开发/esp32_firmware'
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
$pythonPath = if (Test-Path 'D:/python-3.13.9/python.exe') { 'D:/python-3.13.9/python.exe' } elseif ($pythonCommand) { $pythonCommand.Source } else { throw '请先安装 Python，或设置 python 的 PATH。' }
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:WEB_ALLOWED_ORIGINS = 'https://doc-b-yang.github.io,http://127.0.0.1:8000,http://localhost:8000'
if (-not (Get-NetTCPConnection -State Listen -LocalPort 8000 -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath $pythonPath -ArgumentList '-u','run_api.py' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $PSScriptRoot 'data/api-current.stdout.log') -RedirectStandardError (Join-Path $PSScriptRoot 'data/api-current.stderr.log')
}
if (-not (Get-NetTCPConnection -State Listen -LocalPort 8011 -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath $pythonPath -ArgumentList '-u','experiment_wave_viewer.py' -WorkingDirectory $hardwareRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $hardwareRoot 'diagnostics/viewer-current.stdout.log') -RedirectStandardError (Join-Path $hardwareRoot 'diagnostics/viewer-current.stderr.log')
}
Write-Host '已启动电脑端服务。Chrome 使用项目网页；若询问本地网络访问，请允许。BLE 可直接显示双波形，USB 串口由波形服务读取。'
