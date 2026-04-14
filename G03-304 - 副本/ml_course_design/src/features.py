from pydantic import BaseModel, Field, validator
from pandera import Column, Check, DataFrameSchema
import pandera as pa
from typing import Literal, Optional

# 定义性别类型
gender_types = Literal["Male", "Female"]

# 定义Yes/No类型
yes_no_types = Literal["Yes", "No"]

# 定义服务相关类型
service_types = Literal["Yes", "No", "No internet service"]
phone_line_types = Literal["Yes", "No", "No phone service"]

# 定义互联网服务类型
internet_service_types = Literal["DSL", "Fiber optic", "No"]

# 定义合同类型
contract_types = Literal["Month-to-month", "One year", "Two year"]

# 定义支付方式类型
payment_method_types = Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]


class CustomerFeatures(BaseModel):
    """客户特征模型"""
    # 基本信息
    gender: gender_types = Field(description="性别")
    SeniorCitizen: int = Field(ge=0, le=1, description="是否为老年人 (0=No, 1=Yes)")
    Partner: yes_no_types = Field(description="是否有伴侣")
    Dependents: yes_no_types = Field(description="是否有家属")
    tenure: int = Field(ge=0, le=100, description="客户在网时长 (月)")
    
    # 电话服务
    PhoneService: yes_no_types = Field(description="是否有电话服务")
    MultipleLines: phone_line_types = Field(description="是否有多条线路")
    
    # 互联网服务
    InternetService: internet_service_types = Field(description="互联网服务类型")
    OnlineSecurity: service_types = Field(description="是否有在线安全服务")
    OnlineBackup: service_types = Field(description="是否有在线备份服务")
    DeviceProtection: service_types = Field(description="是否有设备保护服务")
    TechSupport: service_types = Field(description="是否有技术支持服务")
    StreamingTV: service_types = Field(description="是否有流媒体电视服务")
    StreamingMovies: service_types = Field(description="是否有流媒体电影服务")
    
    # 合同和账单
    Contract: contract_types = Field(description="合同类型")
    PaperlessBilling: yes_no_types = Field(description="是否使用无纸化账单")
    PaymentMethod: payment_method_types = Field(description="支付方式")
    MonthlyCharges: float = Field(ge=0, le=200, description="月费用")
    TotalCharges: float = Field(ge=0, le=10000, description="总费用")
    
    class Config:
        populate_by_name = True
        from_attributes = True


# 定义用于数据验证的DataFrame Schema
data_schema = DataFrameSchema(
    columns={
        # 输入特征
        "gender": Column(pa.String, checks=Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(pa.Int, checks=Check.isin([0, 1])),
        "Partner": Column(pa.String, checks=Check.isin(["Yes", "No"])),
        "Dependents": Column(pa.String, checks=Check.isin(["Yes", "No"])),
        "tenure": Column(pa.Int, checks=Check.ge(0)),
        "PhoneService": Column(pa.String, checks=Check.isin(["Yes", "No"])),
        "MultipleLines": Column(pa.String, checks=Check.isin(["Yes", "No", "No phone service"])),
        "InternetService": Column(pa.String, checks=Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": Column(pa.String, checks=Check.isin(["Yes", "No", "No internet service"])),
        "OnlineBackup": Column(pa.String, checks=Check.isin(["Yes", "No", "No internet service"])),
        "DeviceProtection": Column(pa.String, checks=Check.isin(["Yes", "No", "No internet service"])),
        "TechSupport": Column(pa.String, checks=Check.isin(["Yes", "No", "No internet service"])),
        "StreamingTV": Column(pa.String, checks=Check.isin(["Yes", "No", "No internet service"])),
        "StreamingMovies": Column(pa.String, checks=Check.isin(["Yes", "No", "No internet service"])),
        "Contract": Column(pa.String, checks=Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": Column(pa.String, checks=Check.isin(["Yes", "No"])),
        "PaymentMethod": Column(pa.String, checks=Check.isin([
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])),
        "MonthlyCharges": Column(pa.Float, checks=Check.ge(0)),
        "TotalCharges": Column(pa.Float, checks=Check.ge(0)),
        
        # 目标变量
        "Churn": Column(pa.Int, checks=Check.isin([0, 1])),
    },
    strict=True,
    coerce=True,
    name="customer_churn_schema"
)


if __name__ == "__main__":
    # 测试特征模型
    print("测试CustomerFeatures模型...")
    
    # 创建一个有效的特征实例
    valid_features = CustomerFeatures(
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
    
    print("有效特征实例:")
    print(valid_features)
    
    print("\n特征模型测试通过!")
