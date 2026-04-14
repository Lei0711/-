#!/usr/bin/env python3
"""
电信客户流失预测Agent启动脚本
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent
    
    # 项目根目录
    project_root = script_dir / "ml_course_design"
    
    print(f"当前脚本目录: {script_dir}")
    print(f"项目根目录: {project_root}")
    
    # 检查项目根目录是否存在
    if not project_root.exists():
        print(f"错误: 项目根目录不存在于 {project_root}")
        print("请确保该脚本与 ml_course_design 文件夹位于同一目录下")
        input("按回车键退出...")
        return 1
    
    # 检查uv是否已安装
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        print("错误: 未找到uv命令")
        print("请先安装uv: pip install uv")
        input("按回车键退出...")
        return 1
    
    # 切换到项目根目录并启动Agent应用
    print("正在启动客户流失预测Agent...")
    print(f"\n使用以下命令启动Agent:")
    print(f"cd {project_root} && uv run python -m src.agent_app")
    
    # 执行命令
    try:
        subprocess.run(
            ["uv", "run", "python", "-m", "src.agent_app"],
            cwd=str(project_root),
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"启动失败: {e}")
        input("按回车键退出...")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
