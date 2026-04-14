# 客户流失预测系统


## 📋 项目概述

本项目是一个基于机器学习的电信客户流失预测系统，结合了智能Agent技术，能够通过自然语言交互和可视化界面提供客户流失风险预测服务。

### 项目目标

- 构建一个结构化数据集的分类/回归模型
- 实现一个能够理解自然语言的智能Agent
- 提供用户友好的可视化交互界面

### 技术栈

- **数据处理**: Polars + pandas
- **可视化**: Seaborn + Streamlit + Plotly
- **数据验证**: Pydantic + pandera
- **机器学习**: scikit-learn + LightGBM
- **智能Agent**: pydantic-ai
- **LLM服务**: DeepSeek

## 🚀 快速开始

### 1. 环境配置

#### 安装依赖

```bash
# 进入项目目录，因为 pyproject.toml 位于 ml_course_design 文件夹中
cd ml_course_design

# 如果 uv 命令不存在，先安装 uv CLI
python -m pip install uv

# 使用 uv 安装项目依赖
uv sync
```

#### 配置API Key

```bash
# 复制环境变量示例文件
cp .env.example .env

# 编辑.env文件，配置DeepSeek API Key
# DEEPSEEK_API_KEY="your-key-here"
```

### 2. 运行应用

#### 方式A: 运行Streamlit演示应用

```bash
uv run streamlit run src/streamlit_app.py
```

#### 方式B: 运行智能Agent演示

```bash
uv run python src/agent_app.py
```

#### 方式C: 运行模型训练脚本

```bash
uv run python src/train.py
```

### 3. 从任意目录运行（可选）

如果你想从项目根目录外运行应用，可以使用完整路径：

```bash
# 运行智能Agent演示
uv run python "path/to/ml_course_design/src/agent_app.py"

# 运行模型训练脚本
uv run python "path/to/ml_course_design/src/train.py"

# 运行Streamlit演示应用
uv run -C "path/to/ml_course_design" streamlit run src/streamlit_app.py
```

## 📊 数据说明

### 数据集

本项目使用了Kaggle上的**Telco Customer Churn**数据集，包含了7043名电信客户的信息和流失状态。

### 数据字段

- **客户信息**: 性别、年龄、是否有伴侣/家属、在网时长
- **服务信息**: 电话服务、互联网服务、在线安全、云备份等
- **合同信息**: 合同类型、支付方式、月费用、总费用
- **目标变量**: 是否流失(Churn)

### 数据预处理

- 使用Polars Lazy API进行高效数据处理
- 处理缺失值和异常值
- 特征编码和标准化

## 🧠 机器学习实现

### 模型架构

- **基准模型**: Logistic Regression
- **高级模型**: LightGBM

### 评估指标

| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | ROC-AUC |
|------|--------|--------|--------|--------|---------|
| Logistic Regression | 0.8068 | 0.6600 | 0.5629 | 0.6076 | 0.8547 |
| LightGBM | 0.9723 | 0.9358 | 0.9616 | 0.9485 | 0.9951 |

### 特征重要性

影响客户流失的关键特征包括：
- 合同类型（月付客户流失风险更高）
- 在网时长（新客户流失风险更高）
- 月费用（高费用客户流失风险更高）
- 支付方式（电子支票支付客户流失风险更高）

## 🤖 Agent 实现

### 功能概述

智能Agent能够理解自然语言输入，提取客户信息，并提供流失风险预测和决策建议。

### 工具列表

| 工具名称 | 功能 | 输入 | 输出 |
|---------|------|------|------|
| `predict_churn` | 使用ML模型预测流失风险 | CustomerFeatures | float |
| `explain_churn` | 解释影响流失的关键因素 | CustomerFeatures | list[str] |

### 交互示例

**输入**: 
```
我有一个女性客户，35岁，在网2个月，月费用89.99，使用电子支票支付，采用月付合同
```

**输出**: 
```json
{
  "risk_score": 0.72,
  "decision": "高风险客户，建议重点关注",
  "actions": ["主动联系客户", "提供个性化优惠", "分析使用习惯"],
  "rationale": "月付合同、在网时长短和电子支票支付是导致高流失风险的主要因素"
}
```

## 🎨 Streamlit 应用

### 功能特点

- **直观的输入界面**: 分步填写客户信息
- **实时预测结果**: 立即显示流失风险评分
- **风险等级可视化**: 使用颜色和进度条直观展示风险
- **影响因素分析**: 提供详细的风险因素解释
- **数据统计展示**: 可视化展示不同特征与流失率的关系

### 使用方法

1. 在左侧边栏填写客户信息
2. 点击"预测流失风险"按钮
3. 在主界面查看预测结果和建议

## 📁 项目结构

```
ml_course_design/
├── pyproject.toml            # 项目依赖配置
├── .env.example              # 环境变量示例
├── .gitignore                # Git忽略规则
├── README.md                 # 项目说明文档
├── data/                     # 数据集目录
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/                   # 模型保存目录
│   └── best_model_lr.joblib
├── src/                      # 源代码目录
│   ├── __init__.py
│   ├── data.py               # 数据处理模块
│   ├── features.py           # 特征定义模块
│   ├── train.py              # 模型训练模块
│   ├── infer.py              # 推理接口模块
│   ├── agent_app.py          # Agent应用
│   └── streamlit_app.py      # Streamlit应用
└── tests/                    # 测试目录
```

## 🔧 核心模块说明

### 1. 数据处理模块 (data.py)

```python
# 使用Polars Lazy API高效处理数据
lf = pl.scan_csv("data/train.csv")
result = (
    lf.filter(pl.col("age") > 30)
    .group_by("category")
    .agg(pl.col("value").mean())
    .collect()
)
```

### 2. 特征定义模块 (features.py)

```python
# 使用Pydantic定义特征模型
class CustomerFeatures(BaseModel):
    gender: gender_types
    SeniorCitizen: int = Field(ge=0, le=1)
    tenure: int = Field(ge=0, le=100)
    MonthlyCharges: float = Field(ge=0, le=200)
    # ... 其他特征
```

### 3. 模型训练模块 (train.py)

```python
# 创建预处理管道
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

# 训练LightGBM模型
lgb_model = lgb.train(
    params,
    lgb_train,
    num_boost_round=500
)
```

### 4. 推理接口模块 (infer.py)

```python
# 单例预测
result = inferencer.predict_single(customer_features)

# 预测解释
result = inferencer.explain_prediction(customer_features)
```

## 📈 模型性能

### 训练集性能

| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | ROC-AUC |
|------|--------|--------|--------|--------|---------|
| Logistic Regression | 0.8068 | 0.6600 | 0.5629 | 0.6076 | 0.8547 |
| LightGBM | 0.9723 | 0.9358 | 0.9616 | 0.9485 | 0.9951 |

### 测试集性能

| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | ROC-AUC |
|------|--------|--------|--------|--------|---------|
| Logistic Regression | 0.7982 | 0.6364 | 0.5615 | 0.5966 | 0.8357 |

## 🎯 项目亮点

1. **高效数据处理**: 使用Polars Lazy API实现大规模数据的快速处理
2. **严格数据验证**: 结合Pydantic和pandera确保数据质量
3. **双模型架构**: 同时实现基准模型和高级模型，便于对比分析
4. **智能Agent交互**: 支持自然语言查询，提供人性化服务
5. **可视化界面**: 直观的Streamlit应用，降低使用门槛
6. **可解释性**: 提供详细的预测解释和影响因素分析

## 📝 开发日志

### Day 1: 项目初始化
- 完成项目结构搭建
- 配置开发环境
- 数据探索和分析

### Day 2: 数据处理
- 实现数据加载和预处理
- 特征工程
- 数据验证规则定义

### Day 3: 模型训练
- 实现Logistic Regression模型
- 实现LightGBM模型
- 模型评估和对比

### Day 4: Agent和应用开发
- 实现智能Agent
- 开发Streamlit应用
- 功能测试和优化

### Day 5: 项目完善
- 文档编写
- 代码优化
- 最终测试

## 4️⃣ 开发心得

### 4.1 主要困难与解决方案

在项目开发过程中，遇到的主要困难及其解决方案如下：

1. **模块导入问题**
   - **困难**：当从项目根目录外运行脚本时，Python无法找到`src`模块，出现`ModuleNotFoundError`
   - **解决方案**：在脚本中添加路径处理逻辑，自动将项目根目录添加到Python路径中，确保模块能够正确导入

2. **环境兼容性问题**
   - **困难**：用户使用的PowerShell 5不支持现代Shell语法（如`&&`命令分隔符）
   - **解决方案**：创建了基于Python的跨平台启动脚本，确保在不同环境下都能正常运行

3. **第三方库API变化**
   - **困难**：`pydantic_ai`库的API与预期不符（如`register_tool`方法不存在，需要使用`tool`方法；`run`方法需要改为`run_sync`）
   - **解决方案**：查阅库的帮助文档和源代码，调整代码以使用正确的API

4. **模型版本兼容性**
   - **困难**：加载模型时出现scikit-learn版本不兼容的警告
   - **解决方案**：确保训练和推理使用相同版本的库，并在文档中注明版本要求

### 4.2 对 AI 辅助编程的感受

使用AI辅助编程工具（如Trae IDE）的体验非常良好，主要体现在以下方面：

1. **有帮助的场景**
   - **快速生成代码框架**：能够根据需求快速生成项目结构和基础代码
   - **解决技术问题**：对于特定的技术问题，能够提供多种解决方案
   - **优化代码质量**：能够识别代码中的问题并提供改进建议
   - **学习新技术**：能够解释复杂的技术概念，帮助快速掌握新技术

2. **需要注意的地方**
   - **代码验证**：生成的代码可能存在细微错误，需要仔细验证和测试
   - **API准确性**：对于特定库的最新API可能不够了解，需要查阅官方文档确认
   - **业务逻辑**：复杂的业务逻辑需要结合人类的专业知识进行设计
   - **过度依赖**：避免过度依赖AI工具，保持独立思考和问题解决能力

### 4.3 局限与未来改进

如果有更多时间，项目还可以从以下几个方面进行改进：

1. **模型性能优化**
   - 尝试更多的特征工程方法，如特征选择、特征交叉等
   - 调参优化LightGBM模型，提高预测准确率
   - 尝试其他先进的算法，如XGBoost、CatBoost或深度学习模型

2. **应用功能扩展**
   - 添加更多的可视化图表，如客户流失风险分布、特征重要性分析等
   - 实现批量预测功能，支持导入Excel或CSV文件进行批量分析
   - 添加模型监控和更新机制，定期重新训练模型以适应新数据
   - 支持多语言界面，提高应用的可用性

3. **系统架构改进**
   - 分离前后端，使用FastAPI构建API，Streamlit作为前端
   - 实现模型服务化部署，支持RESTful API调用
   - 添加用户认证和权限管理，提高系统安全性
   - 支持多模型版本管理，方便模型迭代和回滚

4. **开发流程优化**
   - 添加更全面的单元测试和集成测试，提高代码质量
   - 实现CI/CD流水线，自动构建、测试和部署
   - 添加代码质量检查工具，如flake8、mypy等
   - 完善文档和注释，提高代码的可维护性

5. **用户体验改进**
   - 优化Streamlit界面，提高用户交互体验
   - 添加详细的使用说明和帮助文档
   - 提供更智能的用户输入提示和错误处理

通过这些改进，可以进一步提高项目的性能、可用性和可维护性，使其成为一个更完善的电信客户流失预测系统。
