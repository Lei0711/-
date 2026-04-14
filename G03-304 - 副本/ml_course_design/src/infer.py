import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

from .features import CustomerFeatures


class ModelInferencer:
    """模型推理类"""
    
    def __init__(self, model_path: str | Path = None):
        """初始化模型推理器
        
        Args:
            model_path: 模型路径，如果为None则使用默认路径
        """
        if model_path is None:
            self.model_path = Path(__file__).parent.parent / "models" / "best_model_lr.joblib"
        else:
            self.model_path = Path(model_path)
        
        # 加载模型
        self.model = self.load_model()
    
    def load_model(self) -> Any:
        """加载训练好的模型
        
        Returns:
            加载的模型对象
        """
        print(f"正在加载模型: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        model = joblib.load(self.model_path)
        print(f"模型加载成功: {type(model).__name__}")
        
        return model
    
    def predict_single(self, features: CustomerFeatures) -> Dict[str, Any]:
        """对单个客户进行流失预测
        
        Args:
            features: 客户特征对象
            
        Returns:
            预测结果，包含流失概率和预测类别
        """
        # 将特征转换为DataFrame
        features_dict = features.model_dump()
        df = pd.DataFrame([features_dict])
        
        # 进行预测
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]
        
        # 构造结果
        result = {
            "prediction": int(prediction),  # 0=不流失, 1=流失
            "probability": float(probability),  # 流失概率
            "churn": bool(prediction),  # 是否流失
            "features": features_dict
        }
        
        return result
    
    def predict_batch(self, features_list: List[CustomerFeatures]) -> List[Dict[str, Any]]:
        """对多个客户进行批量流失预测
        
        Args:
            features_list: 客户特征对象列表
            
        Returns:
            批量预测结果列表
        """
        # 将特征列表转换为DataFrame
        features_dicts = [features.model_dump() for features in features_list]
        df = pd.DataFrame(features_dicts)
        
        # 进行批量预测
        predictions = self.model.predict(df)
        probabilities = self.model.predict_proba(df)[:, 1]
        
        # 构造结果列表
        results = []
        for i in range(len(predictions)):
            result = {
                "prediction": int(predictions[i]),
                "probability": float(probabilities[i]),
                "churn": bool(predictions[i]),
                "features": features_dicts[i]
            }
            results.append(result)
        
        return results
    
    def explain_prediction(self, features: CustomerFeatures) -> Dict[str, Any]:
        """解释预测结果
        
        Args:
            features: 客户特征对象
            
        Returns:
            包含预测结果和解释的字典
        """
        # 获取基本预测结果
        prediction_result = self.predict_single(features)
        
        # 分析影响流失的关键因素
        key_factors = []
        
        # 根据业务知识分析影响因素
        if features.Contract == "Month-to-month":
            key_factors.append("月付合同增加了流失风险")
        
        if features.tenure < 12:
            key_factors.append("在网时长较短增加了流失风险")
        
        if features.MonthlyCharges > 70:
            key_factors.append("月费用较高增加了流失风险")
        
        if features.InternetService == "Fiber optic":
            key_factors.append("光纤互联网服务用户流失风险较高")
        
        if features.PaymentMethod == "Electronic check":
            key_factors.append("电子支票支付方式增加了流失风险")
        
        if features.PaperlessBilling == "Yes":
            key_factors.append("无纸化账单用户流失风险较高")
        
        # 如果没有找到明显因素
        if not key_factors:
            key_factors.append("客户特征组合导致流失风险处于平均水平")
        
        # 添加解释到结果中
        prediction_result["explanation"] = key_factors
        
        return prediction_result


if __name__ == "__main__":
    # 测试推理功能
    print("测试模型推理功能...")
    
    # 创建测试特征
    test_features = CustomerFeatures(
        gender="Female",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="No",
        tenure=1,
        PhoneService="No",
        MultipleLines="No phone service",
        InternetService="DSL",
        OnlineSecurity="No",
        OnlineBackup="Yes",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=29.85,
        TotalCharges=29.85
    )
    
    # 初始化推理器
    inferencer = ModelInferencer()
    
    # 进行单例预测
    result = inferencer.predict_single(test_features)
    print("\n单例预测结果:")
    print(result)
    
    # 进行预测解释
    explained_result = inferencer.explain_prediction(test_features)
    print("\n预测解释:")
    print(f"流失概率: {explained_result['probability']:.4f}")
    print(f"预测结果: {'流失' if explained_result['churn'] else '不流失'}")
    print("影响因素:")
    for factor in explained_result['explanation']:
        print(f"  - {factor}")
