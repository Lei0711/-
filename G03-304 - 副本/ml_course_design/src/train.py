import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# 添加项目根目录到Python路径，解决直接运行时的导入问题
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import DataProcessor
from src.features import data_schema


class ModelTrainer:
    """模型训练类"""
    
    def __init__(self, models_dir: str | Path = None):
        """初始化模型训练器
        
        Args:
            models_dir: 模型保存目录，如果为None则使用默认路径
        """
        if models_dir is None:
            self.models_dir = Path(__file__).parent.parent / "models"
        else:
            self.models_dir = Path(models_dir)
        
        # 确保模型目录存在
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_data(self) -> tuple:
        """准备训练数据
        
        Returns:
            训练集、验证集和测试集（X_train, X_val, X_test, y_train, y_val, y_test）
        """
        print("准备训练数据...")
        
        # 加载和预处理数据
        processor = DataProcessor()
        X, y = processor.get_processed_data()
        
        # 转换为pandas DataFrame以便与scikit-learn兼容
        X_pandas = X.to_pandas()
        y_pandas = y.to_pandas()
        
        # 划分训练集和测试集 (80% train, 20% test)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_pandas, y_pandas, test_size=0.2, random_state=42, stratify=y_pandas
        )
        
        # 从训练集中划分验证集 (75% train, 25% val)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
        )
        
        print(f"训练集: {X_train.shape}")
        print(f"验证集: {X_val.shape}")
        print(f"测试集: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_preprocessor(self, X_train: pd.DataFrame) -> ColumnTransformer:
        """创建数据预处理管道
        
        Args:
            X_train: 训练集数据，用于获取特征信息
            
        Returns:
            数据预处理管道
        """
        print("创建数据预处理管道...")
        
        # 分离数值特征和分类特征
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
        
        print(f"数值特征: {numeric_features}")
        print(f"分类特征: {categorical_features}")
        
        # 创建数值特征处理管道
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # 创建分类特征处理管道
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # 创建完整的预处理管道
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor
    
    def train_logistic_regression(self, preprocessor: ColumnTransformer, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """训练Logistic Regression模型
        
        Args:
            preprocessor: 数据预处理管道
            X_train: 训练集特征
            y_train: 训练集目标变量
            
        Returns:
            训练好的Logistic Regression模型管道
        """
        print("训练Logistic Regression模型...")
        
        # 创建Logistic Regression模型管道
        lr_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        # 训练模型
        lr_pipeline.fit(X_train, y_train)
        
        return lr_pipeline
    
    def train_lightgbm(self, preprocessor: ColumnTransformer, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
        """训练LightGBM模型
        
        Args:
            preprocessor: 数据预处理管道
            X_train: 训练集特征
            y_train: 训练集目标变量
            
        Returns:
            预处理后的特征、训练好的LightGBM模型
        """
        print("训练LightGBM模型...")
        
        # 预处理训练数据
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        
        # 获取特征名称
        num_features = preprocessor.transformers_[0][2]
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
        feature_names = num_features + list(cat_features)
        
        # 创建LightGBM数据集
        lgb_train = lgb.Dataset(X_train_preprocessed, y_train, feature_name=feature_names)
        
        # 设置LightGBM参数
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'n_estimators': 500,
            'random_state': 42,
            'verbose': -1
        }
        
        # 训练LightGBM模型
        lgb_model = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_train],
            callbacks=[lgb.log_evaluation(period=100)]
        )
        
        return feature_names, lgb_model
    
    def evaluate_model(self, model: Pipeline | lgb.Booster, preprocessor: ColumnTransformer, 
                      X: pd.DataFrame, y: pd.Series, model_name: str) -> dict:
        """评估模型性能
        
        Args:
            model: 要评估的模型
            preprocessor: 数据预处理管道
            X: 测试数据特征
            y: 测试数据目标变量
            model_name: 模型名称
            
        Returns:
            模型性能指标
        """
        print(f"评估{model_name}模型...")
        
        # 预测概率
        if isinstance(model, Pipeline):
            y_pred_proba = model.predict_proba(X)[:, 1]
            y_pred = model.predict(X)
        else:
            # LightGBM模型
            X_preprocessed = preprocessor.transform(X)
            y_pred_proba = model.predict(X_preprocessed)
            y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # 计算性能指标
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        print(f"{model_name} 模型性能:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        return metrics
    
    def save_model(self, model: Pipeline | lgb.Booster, preprocessor: ColumnTransformer, 
                   feature_names: list = None, model_name: str = "best_model"):
        """保存模型和预处理工具
        
        Args:
            model: 要保存的模型
            preprocessor: 数据预处理管道
            feature_names: 特征名称列表（仅LightGBM需要）
            model_name: 模型名称
        """
        print(f"保存{model_name}模型...")
        
        if isinstance(model, Pipeline):
            # 保存完整的管道模型
            model_path = self.models_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path)
        else:
            # 保存LightGBM模型
            model_path = self.models_dir / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            
            # 保存预处理管道
            preprocessor_path = self.models_dir / "preprocessor.joblib"
            joblib.dump(preprocessor, preprocessor_path)
            
            # 保存特征名称
            features_path = self.models_dir / "features.joblib"
            joblib.dump(feature_names, features_path)
        
        print(f"模型保存成功: {model_path}")
    
    def train_and_evaluate(self):
        """完整的训练和评估流程"""
        print("开始模型训练和评估流程...")
        
        # 1. 准备数据
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
        
        # 2. 创建预处理管道
        preprocessor = self.create_preprocessor(X_train)
        
        # 3. 训练Logistic Regression模型
        lr_model = self.train_logistic_regression(preprocessor, X_train, y_train)
        
        # 4. 评估Logistic Regression模型
        lr_train_metrics = self.evaluate_model(lr_model, preprocessor, X_train, y_train, "Logistic Regression (训练集)")
        lr_val_metrics = self.evaluate_model(lr_model, preprocessor, X_val, y_val, "Logistic Regression (验证集)")
        
        # 5. 训练LightGBM模型
        feature_names, lgb_model = self.train_lightgbm(preprocessor, X_train, y_train)
        
        # 6. 评估LightGBM模型
        lgb_train_metrics = self.evaluate_model(lgb_model, preprocessor, X_train, y_train, "LightGBM (训练集)")
        lgb_val_metrics = self.evaluate_model(lgb_model, preprocessor, X_val, y_val, "LightGBM (验证集)")
        
        # 7. 选择最佳模型
        print("\n选择最佳模型...")
        best_model = None
        best_model_name = ""
        
        if lr_val_metrics['roc_auc'] > lgb_val_metrics['roc_auc']:
            best_model = lr_model
            best_model_name = "Logistic Regression"
        else:
            best_model = lgb_model
            best_model_name = "LightGBM"
        
        print(f"最佳模型: {best_model_name}")
        
        # 8. 在测试集上评估最佳模型
        print(f"\n在测试集上评估{best_model_name}模型...")
        if isinstance(best_model, Pipeline):
            best_test_metrics = self.evaluate_model(best_model, preprocessor, X_test, y_test, "Best Model (测试集)")
        else:
            best_test_metrics = self.evaluate_model(best_model, preprocessor, X_test, y_test, "Best Model (测试集)")
        
        # 9. 保存最佳模型
        if isinstance(best_model, Pipeline):
            self.save_model(best_model, preprocessor, model_name="best_model_lr")
        else:
            self.save_model(best_model, preprocessor, feature_names, model_name="best_model")
        
        print("\n模型训练和评估流程完成!")
        
        return best_model, best_test_metrics


if __name__ == "__main__":
    # 运行模型训练和评估
    trainer = ModelTrainer()
    best_model, test_metrics = trainer.train_and_evaluate()
