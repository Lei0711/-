import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# 使用绝对导入或直接导入
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.features import CustomerFeatures
from src.infer import ModelInferencer


class ChurnPredictionApp:
    """客户流失预测Streamlit应用"""
    
    def __init__(self):
        """初始化应用"""
        # 设置页面配置
        st.set_page_config(
            page_title="客户流失预测系统",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 初始化推理器
        self.inferencer = ModelInferencer()
        
        # 设置应用标题和说明
        self._set_app_header()
    
    def _set_app_header(self):
        """设置应用标题和说明"""
        st.title("📊 客户流失预测系统")
        st.markdown("---")
        st.write("这是一个基于机器学习的电信客户流失预测系统。输入客户信息，系统将预测该客户的流失风险并提供针对性建议。")
    
    def _create_input_form(self) -> dict:
        """创建客户信息输入表单
        
        Returns:
            输入的客户信息字典
        """
        with st.sidebar.form("customer_info_form"):
            st.header("客户信息")
            
            # 基本信息
            st.subheader("基本信息")
            gender = st.selectbox("性别", ["Male", "Female"])
            senior_citizen = st.selectbox("是否为老年人", [0, 1], format_func=lambda x: "是" if x == 1 else "否")
            partner = st.selectbox("是否有伴侣", ["Yes", "No"], format_func=lambda x: "是" if x == "Yes" else "否")
            dependents = st.selectbox("是否有家属", ["Yes", "No"], format_func=lambda x: "是" if x == "Yes" else "否")
            tenure = st.number_input("在网时长（月）", min_value=0, max_value=100, value=1)
            
            # 电话服务
            st.subheader("电话服务")
            phone_service = st.selectbox("是否有电话服务", ["Yes", "No"], format_func=lambda x: "是" if x == "Yes" else "否")
            
            if phone_service == "Yes":
                multiple_lines = st.selectbox("是否有多条线路", ["Yes", "No"])
            else:
                multiple_lines = "No phone service"
            
            # 互联网服务
            st.subheader("互联网服务")
            internet_service = st.selectbox("互联网服务类型", ["DSL", "Fiber optic", "No"])
            
            if internet_service != "No":
                online_security = st.selectbox("是否有在线安全服务", ["Yes", "No"])
                online_backup = st.selectbox("是否有在线备份服务", ["Yes", "No"])
                device_protection = st.selectbox("是否有设备保护服务", ["Yes", "No"])
                tech_support = st.selectbox("是否有技术支持服务", ["Yes", "No"])
                streaming_tv = st.selectbox("是否有流媒体电视服务", ["Yes", "No"])
                streaming_movies = st.selectbox("是否有流媒体电影服务", ["Yes", "No"])
            else:
                online_security = "No internet service"
                online_backup = "No internet service"
                device_protection = "No internet service"
                tech_support = "No internet service"
                streaming_tv = "No internet service"
                streaming_movies = "No internet service"
            
            # 合同和账单
            st.subheader("合同和账单")
            contract = st.selectbox("合同类型", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("是否使用无纸化账单", ["Yes", "No"], format_func=lambda x: "是" if x == "Yes" else "否")
            payment_method = st.selectbox(
                "支付方式", 
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )
            monthly_charges = st.number_input("月费用", min_value=0.0, max_value=200.0, value=29.85, step=0.01)
            total_charges = st.number_input("总费用", min_value=0.0, max_value=10000.0, value=29.85, step=0.01)
            
            # 提交按钮
            submit_button = st.form_submit_button("预测流失风险")
        
        # 构造特征字典
        features_dict = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        
        return features_dict, submit_button
    
    def _display_prediction_result(self, result: dict):
        """展示预测结果
        
        Args:
            result: 预测结果字典
        """
        st.markdown("---")
        st.header("预测结果")
        
        # 创建两列布局
        col1, col2 = st.columns(2)
        
        with col1:
            # 显示流失风险分数
            st.subheader("📈 流失风险评分")
            
            # 创建风险评分可视化
            risk_score = result["probability"]
            risk_percentage = risk_score * 100
            
            # 确定风险等级
            if risk_score < 0.3:
                risk_level = "低风险"
                color = "green"
            elif risk_score < 0.7:
                risk_level = "中风险"
                color = "orange"
            else:
                risk_level = "高风险"
                color = "red"
            
            # 使用进度条显示风险评分
            st.progress(risk_score)
            st.write(f"**风险等级:** <span style='color:{color}; font-weight:bold;'>{risk_level}</span>", unsafe_allow_html=True)
            st.write(f"**风险概率:** {risk_percentage:.1f}%")
            st.write(f"**预测结果:** {'⚠️ 可能流失' if result['churn'] else '✅ 不太可能流失'}")
        
        with col2:
            # 显示影响因素
            st.subheader("🔍 影响因素分析")
            
            # 检查是否有解释信息
            if "explanation" in result:
                for factor in result["explanation"]:
                    st.write(f"- {factor}")
            else:
                st.write("暂无影响因素分析")
        
        # 显示详细特征
        with st.expander("📋 详细客户信息"):
            df_features = pd.DataFrame.from_dict(result["features"], orient="index", columns=["值"])
            st.dataframe(df_features, use_container_width=True)
        
        # 显示建议
        st.subheader("💡 建议采取的行动")
        if result["churn"]:
            st.markdown("""
            - 主动联系客户，了解其需求和不满
            - 提供针对性的优惠活动，如折扣或礼品
            - 分析客户使用习惯，推荐更适合的套餐
            - 加强客户服务，提高客户满意度
            """)
        else:
            st.markdown("""
            - 继续保持良好的客户服务
            - 定期推送个性化的优惠信息
            - 关注客户使用行为变化
            - 鼓励客户升级套餐或添加新服务
            """)
    
    def _show_data_statistics(self):
        """显示数据统计信息"""
        st.markdown("---")
        st.header("📊 数据统计信息")
        
        # 创建模拟的流失数据统计
        data = {
            "合同类型": ["月付", "一年", "两年"],
            "客户数": [4200, 2100, 732],
            "流失率": [0.42, 0.18, 0.09]
        }
        
        df = pd.DataFrame(data)
        
        # 显示合同类型与流失率的关系
        fig = px.bar(df, x="合同类型", y="流失率", color="合同类型", 
                     title="不同合同类型的客户流失率",
                     labels={"流失率": "流失率(%)"}, 
                     hover_data={"客户数": True})
        fig.update_traces(hovertemplate="合同类型: %{x}<br>流失率: %{y:.1%}<br>客户数: %{customdata[0]}")
        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """运行应用"""
        # 创建输入表单
        features_dict, submit_button = self._create_input_form()
        
        # 当用户点击预测按钮时
        if submit_button:
            try:
                # 验证输入并创建特征对象
                features = CustomerFeatures(**features_dict)
                
                # 进行预测
                with st.spinner("正在预测..."):
                    result = self.inferencer.explain_prediction(features)
                
                # 展示预测结果
                self._display_prediction_result(result)
                
            except Exception as e:
                st.error(f"预测过程中发生错误: {e}")
        
        # 显示数据统计信息
        self._show_data_statistics()


if __name__ == "__main__":
    # 启动应用
    app = ChurnPredictionApp()
    app.run()
