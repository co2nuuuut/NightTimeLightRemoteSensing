import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

# ================= 配置 =================
MODEL_PATH = "../../03_Models/best_model.pkl"  # 刚才保存的冠军模型
DATA_PATH = "../../01_Processed_Data/Turkey_Earthquake_Analysis_Monthly.csv"  # GEE 导出的新数据

# ================= 分析逻辑 =================
print("🚀 开始地震影响分析...")

# 1. 加载模型
model = joblib.load(MODEL_PATH)
print("1. 模型加载成功")

# 2. 加载数据
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
print(f"2. 数据加载成功: {len(df)} 个月度记录")

# 3. 数据预处理 (必须与训练时完全一致！)
# 训练时：log_ntl = log1p(NTL_sum)
# 问题：这里的 NTL_sum 是单月的，而训练是全年的。
# 技巧：将月度数据 "年化" (Annualize) -> 假设每个月都像这个月一样，全年的量是多少？
# 公式：Annualized_Sum = Monthly_Sum * 12
df['annualized_ntl_sum'] = df['NTL_sum'] * 12

# 对数变换
df['log_ntl'] = np.log1p(df['annualized_ntl_sum'])

# 确保特征顺序与训练时一致
# 检查训练代码里的 features 列表，通常是 ['log_ntl', 'NDVI_mean', 'Precip_mean']
features = ['log_ntl', 'NDVI_mean', 'Precip_mean']
X_target = df[features]

# 4. 预测
# 预测出来的是 Log GDP
df['pred_log_gdp'] = model.predict(X_target)

# 还原为 GDP 指数 (可选，保持 Log 形态更容易看趋势)
# df['pred_gdp_index'] = np.expm1(df['pred_log_gdp'])

# 5. 去季节性 (简单移动平均法)
# 消除 "冬天亮夏天暗" 的自然波动，提取真实趋势
df['gdp_trend'] = df['pred_log_gdp'].rolling(window=3, center=True).mean()

# 6. 可视化
plt.figure(figsize=(12, 6))

# 绘制原始预测值
sns.lineplot(x='date', y='pred_log_gdp', data=df, label='Monthly Prediction (Raw)', alpha=0.4, color='gray')
# 绘制趋势线
sns.lineplot(x='date', y='gdp_trend', data=df, label='Economic Trend (3-Month Avg)', linewidth=2.5, color='blue')

# 标记地震时间 (2023年2月6日)
earthquake_date = pd.to_datetime('2023-02-06')
plt.axvline(earthquake_date, color='red', linestyle='--', label='Earthquake (Feb 2023)')

plt.title("Satellite-Predicted Economic Impact of Turkey Earthquake", fontsize=15)
plt.ylabel("Predicted Log GDP (Economic Activity Proxy)", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 保存
plt.savefig("turkey_earthquake_impact.png", dpi=300)
print("✅ 分析完成！结果图已保存为 turkey_earthquake_impact.png")
plt.show()