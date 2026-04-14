# 电信客户流失预测Agent启动脚本 (PowerShell版本)

# 设置项目根目录
$ProjectRoot = "$PSScriptRoot\ml_course_design"

# 检查项目根目录是否存在
if(-not (Test-Path $ProjectRoot)){
    Write-Host "错误: 项目根目录不存在于 $ProjectRoot" -ForegroundColor Red
    Write-Host "请确保该脚本与 ml_course_design 文件夹位于同一目录下" -ForegroundColor Yellow
    Pause
    exit 1
}

# 检查uv是否已安装
if(-not (Get-Command "uv" -ErrorAction SilentlyContinue)){
    Write-Host "错误: 未找到uv命令" -ForegroundColor Red
    Write-Host "请先安装uv: pip install uv" -ForegroundColor Yellow
    Pause
    exit 1
}

# 切换到项目根目录并启动Agent应用
Write-Host "正在启动客户流失预测Agent..." -ForegroundColor Green
Write-Host "项目根目录: $ProjectRoot" -ForegroundColor Cyan
Set-Location -Path $ProjectRoot

# 启动Agent应用
uv run python -m src.agent_app

# 等待用户按下任意键退出
Write-Host "\n按任意键退出..." -ForegroundColor Gray
$x = $host.ui.RawUI.ReadKey("NoEcho,IncludeKeyDown")
