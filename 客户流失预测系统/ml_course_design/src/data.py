import polars as pl
import pandas as pd
from pathlib import Path
from typing import Tuple

class DataProcessor:
    """数据处理类，用于加载和预处理Telco Customer Churn数据集"""
    
    def __init__(self, data_path: str | Path = None):
        """初始化数据处理器
        
        Args:
            data_path: 数据集路径，如果为None则使用默认路径
        """
        if data_path is None:
            self.data_path = Path(__file__).parent.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        else:
            self.data_path = Path(data_path)
    
    def load_data(self) -> pl.DataFrame:
        """加载原始数据集
        
        Returns:
            加载后的Polars DataFrame
        """
        print(f"正在加载数据: {self.data_path}")
        
        # 使用Lazy API加载数据，提高效率
        lf = pl.scan_csv(self.data_path)
        df = lf.collect()
        
        print(f"数据加载完成，共 {df.shape[0]} 行，{df.shape[1]} 列")
        return df
    
    def preprocess_data(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.Series]:
        """预处理数据集
        
        Args:
            df: 原始数据集
            
        Returns:
            预处理后的特征数据和目标变量
        """
        print("开始数据预处理...")
        
        # 1. 处理缺失值和异常值
        # 检查TotalCharges列的类型，如果是字符串类型则处理空字符串
        if df["TotalCharges"].dtype == pl.String:
            df = df.with_columns(
                pl.col("TotalCharges").str.strip_chars().replace("", None)
            )
            
            # 将TotalCharges转换为浮点型
            df = df.with_columns(
                pl.col("TotalCharges").cast(pl.Float64, strict=False)
            )
        
        # 处理缺失值 - 删除TotalCharges为None的行
        df = df.filter(pl.col("TotalCharges").is_not_null())
        
        # 2. 处理目标变量
        # 将Churn转换为数值型 (0=No, 1=Yes)
        df = df.with_columns(
            pl.col("Churn").replace({"No": 0, "Yes": 1}).cast(pl.Int32).alias("Churn")
        )
        
        # 3. 选择特征列
        # 排除customerID（唯一标识，对模型训练无用）
        feature_cols = [col for col in df.columns if col not in ["customerID", "Churn"]]
        
        # 分离特征和目标变量
        X = df.select(feature_cols)
        y = df.select("Churn").to_series()
        
        print(f"数据预处理完成，特征数据形状: {X.shape}, 目标变量形状: {y.shape}")
        return X, y
    
    def get_processed_data(self) -> Tuple[pl.DataFrame, pl.Series]:
        """获取完整处理后的数据
        
        Returns:
            预处理后的特征数据和目标变量
        """
        df = self.load_data()
        X, y = self.preprocess_data(df)
        return X, y

# 用于测试数据处理模块
if __name__ == "__main__":
    processor = DataProcessor()
    X, y = processor.get_processed_data()
    
    print("\n特征数据示例:")
    print(X.head())
    
    print("\n目标变量示例:")
    print(y.head())
    
    print(f"\n目标变量分布: {y.value_counts().sort("Churn")}")
