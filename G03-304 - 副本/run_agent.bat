@echo off
REM 电信客户流失预测Agent启动脚本

set "PROJECT_ROOT=%~dp0ml_course_design"

REM 检查项目根目录是否存在
if not exist "%PROJECT_ROOT%" (
    echo 错误: 项目根目录不存在于 "%PROJECT_ROOT%"
    echo 请确保该脚本与 ml_course_design 文件夹位于同一目录下
    pause
    exit /b 1
)

REM 检查uv是否已安装
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo 错误: 未找到uv命令
    echo 请先安装uv: pip install uv
    pause
    exit /b 1
)

REM 切换到项目根目录并启动Agent应用
echo 正在启动客户流失预测Agent...
echo 项目根目录: %PROJECT_ROOT%
cd /d "%PROJECT_ROOT%"
uv run python -m src.agent_app

REM 等待用户按下任意键退出
pause
