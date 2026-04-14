import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from typing import List, Optional

# 添加项目根目录到Python路径，解决直接运行时的导入问题
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features import CustomerFeatures
from src.infer import ModelInferencer

# 加载环境变量
load_dotenv()


class DecisionResult(BaseModel):
    """Agent决策结果模型"""
    risk_score: float = Field(ge=0, le=1, description="流失风险分数")
    decision: str = Field(description="决策建议")
    actions: List[str] = Field(description="建议采取的行动")
    rationale: str = Field(description="决策理由")


class CustomerInfo(BaseModel):
    """客户信息模型"""
    age: Optional[int] = Field(description="客户年龄")
    gender: Optional[str] = Field(description="客户性别")
    tenure: Optional[int] = Field(description="在网时长(月)")
    monthly_charges: Optional[float] = Field(description="月费用")
    total_charges: Optional[float] = Field(description="总费用")
    contract_type: Optional[str] = Field(description="合同类型")
    internet_service: Optional[str] = Field(description="互联网服务类型")
    payment_method: Optional[str] = Field(description="支付方式")
    has_partner: Optional[bool] = Field(description="是否有伴侣")
    has_dependents: Optional[bool] = Field(description="是否有家属")
    is_senior: Optional[bool] = Field(description="是否为老年人")


class ChurnPredictionAgent:
    """客户流失预测Agent"""
    
    def __init__(self):
        """初始化Agent"""
        # 获取API Key
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY环境变量未设置，请在.env文件中配置")
        
        # 初始化推理器
        self.inferencer = ModelInferencer()
        
        # 创建Agent
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """创建Agent实例
        
        Returns:
            Agent实例
        """
        agent = Agent(
            model="deepseek:deepseek-chat",
            output_type=DecisionResult,
            system_prompt="你是一名专业的电信客户流失预测分析师，你的任务是根据客户信息预测流失风险并提供决策建议。\n\n" \
            "你可以使用以下工具：\n" \
            "1. predict_churn: 使用机器学习模型预测客户流失风险\n" \
            "2. explain_churn: 解释影响客户流失的关键因素\n\n" \
            "请确保你的回答专业、准确，并提供具体的行动建议。"
        )
        
        # 注册工具
        agent.tool(self.predict_churn)
        agent.tool(self.explain_churn)
        
        return agent
    
    def predict_churn(self, ctx: RunContext, customer_info: CustomerFeatures) -> float:
        """预测客户流失风险
        
        Args:
            customer_info: 客户特征信息
            
        Returns:
            流失风险分数 (0-1)
        """
        result = self.inferencer.predict_single(customer_info)
        return result["probability"]
    
    def explain_churn(self, ctx: RunContext, customer_info: CustomerFeatures) -> List[str]:
        """解释影响客户流失的关键因素
        
        Args:
            customer_info: 客户特征信息
            
        Returns:
            影响因素列表
        """
        result = self.inferencer.explain_prediction(customer_info)
        return result["explanation"]
    
    def process_query(self, query: str) -> DecisionResult:
        """处理用户查询
        
        Args:
            query: 用户的自然语言查询
            
        Returns:
            结构化的决策结果
        """
        print(f"正在处理查询: {query}")
        
        # 运行Agent
        result = self.agent.run_sync(query)
        
        print("查询处理完成")
        return result
    
    def run_interactive(self):
        """启动交互式对话"""
        print("欢迎使用客户流失预测Agent！")
        print("请输入客户信息，我将为您预测流失风险并提供建议。")
        print("输入'退出'或'quit'结束对话。")
        
        while True:
            try:
                query = input("\n请输入查询: ")
                
                if query.lower() in ["退出", "quit", "q"]:
                    print("感谢使用，再见！")
                    break
                
                result = self.process_query(query)
                
                print("\n=== 预测结果 ===")
                print(f"流失风险分数: {result.risk_score:.4f}")
                print(f"决策建议: {result.decision}")
                print("建议采取的行动:")
                for action in result.actions:
                    print(f"  - {action}")
                print(f"决策理由: {result.rationale}")
                print("=================")
                
            except Exception as e:
                print(f"处理查询时发生错误: {e}")
                print("请检查输入或稍后重试。")


if __name__ == "__main__":
    try:
        # 初始化并启动Agent
        agent = ChurnPredictionAgent()
        agent.run_interactive()
    except Exception as e:
        print(f"启动Agent时发生错误: {e}")
        print("请确保已正确配置DEEPSEEK_API_KEY环境变量。")
