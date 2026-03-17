import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from statsmodels.tsa.seasonal import seasonal_decompose

# ================= 配置 =================
MODEL_PATH = "../../03_Models/best_model.pkl"
CALIBRATOR_PATH = "../../03_Models/bias_corrector.pkl"  # 新增
DATA_PATH = r"/rs-fi/01_Processed_Data/Earthquake_Provinces_Monthly_Analysis.csv"

print("🚀 正在生成最终版土耳其地震分析图...")

model = joblib.load(MODEL_PATH)
calibrator = joblib.load(CALIBRATOR_PATH)
df = pd.read_csv(DATA_PATH)

# 基础清洗
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
cols_to_drop = ['.geo', 'system:index']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

# 特征工程
df['annualized_ntl_sum'] = df['NTL_sum'] * 12
df['log_ntl'] = np.log1p(df['annualized_ntl_sum'])
features = ['log_ntl', 'NDVI_mean', 'Precip_mean']

# 重采样
df_resampled = df.set_index('date').resample('MS').mean(numeric_only=True)
df_clean = df_resampled.interpolate(method='linear').bfill().ffill().reset_index()

# 1. 原始预测
raw_pred = model.predict(df_clean[features])
# 2. 线性校正 (关键！)
df_clean['final_log_gdp'] = calibrator.predict(raw_pred.reshape(-1, 1))

# STL 分解
ts_data = df_clean.set_index('date')['final_log_gdp'].asfreq('MS')
decomposition = seasonal_decompose(ts_data, model='additive', period=12)

df_clean['Trend'] = decomposition.trend.values
df_clean['Seasonal'] = decomposition.seasonal.values
df_clean['Resid'] = decomposition.resid.values

# 绘图 (只画 Trend 和 Resid 两个最重要的)
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Trend
axes[0].plot(df_clean['date'], df_clean['Trend'], color='blue', linewidth=3, label='Economic Trend')
axes[0].axvline(pd.to_datetime('2023-02-06'), color='red', linestyle='--', linewidth=2, label='Earthquake')
axes[0].set_title('Trend Component (Calibrated)', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True)

# Resid
axes[1].scatter(df_clean['date'], df_clean['Resid'], color='red', s=20, label='Anomalies')
axes[1].axhline(0, color='black', linestyle=':')
axes[1].axvline(pd.to_datetime('2023-02-06'), color='red', linestyle='--')
axes[1].set_title('Residuals (Unexplained Shock)', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("final_turkey_stl.png", dpi=300)
print("✅ 最终土耳其分析图已保存！")
plt.show()