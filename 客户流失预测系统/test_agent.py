import os
import sys
from pathlib import Path

# 获取项目根目录
project_root = Path(r"c:\Users\HUANGYING\Desktop\jqxx\新建文件夹 (4)\ml_course_design")

# 将项目根目录添加到Python路径
sys.path.insert(0, str(project_root))

print(f"项目根目录: {project_root}")
print(f"Python路径中包含项目根目录: {str(project_root) in sys.path}")

# 测试导入
print("\n测试导入...")
try:
    from src.agent_app import ChurnPredictionAgent
    print("✅ 成功导入ChurnPredictionAgent!")
    
    # 测试创建Agent实例
    agent = ChurnPredictionAgent()
    print("✅ 成功创建Agent实例!")
    
    print(f"\n🎉 测试成功! 现在可以使用以下命令运行Agent应用:")
    print(r"   cd 'c:\Users\HUANGYING\Desktop\jqxx\新建文件夹 (4)\ml_course_design' ; uv run python -m src.agent_app")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 其他错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)