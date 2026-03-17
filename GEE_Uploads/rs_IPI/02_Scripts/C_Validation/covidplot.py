import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import glob
import os

# ================= 配置 =================
MODEL_PATH = "../../03_Models/best_model.pkl"
CALIBRATOR_PATH = "../../03_Models/bias_corrector.pkl"  # 新增：加载校正器
DATA_FOLDER = r"D:\虚拟c盘\大创项目\GEE_Uploads\rs-fi\Global_Validation_Fixed"

print("🚀 正在生成最终版全球 COVID-19 验证图...")

# 加载模型和校正器
model = joblib.load(MODEL_PATH)
calibrator = joblib.load(CALIBRATOR_PATH)

# 读取数据
all_files = glob.glob(os.path.join(DATA_FOLDER, "Global_Validation_Fixed_*.csv"))
df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['country_na', 'date'])

# 特征工程
# 面积系数我们这里简化处理，因为主要看趋势，校正器会自动调整截距
# 我们直接用之前的代理系数，校正器会负责把它缩放回真实范围
AREA_PROXY_FACTOR = 1000000 * 12
df['log_ntl'] = np.log1p(df['NTL_mean'] * AREA_PROXY_FACTOR)
features = ['log_ntl', 'NDVI_mean', 'Precip_mean']
df_clean = df.dropna(subset=features).copy()

# 1. 原始预测
raw_pred = model.predict(df_clean[features])
# 2. 线性校正 (关键步骤！)
# calibrator 需要 2D array，所以 reshape
df_clean['final_log_gdp'] = calibrator.predict(raw_pred.reshape(-1, 1))

# 绘图
target_countries = ['China', 'United States', 'Germany', 'India']
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, country in enumerate(target_countries):
    ax = axes[i]
    sub_df = df_clean[df_clean['country_na'] == country].copy()
    if sub_df.empty: continue

    # 趋势线
    sub_df['trend'] = sub_df['final_log_gdp'].rolling(3, center=True).mean()

    ax.plot(sub_df['date'], sub_df['final_log_gdp'], color='gray', alpha=0.3, label='Monthly (Calibrated)')
    ax.plot(sub_df['date'], sub_df['trend'], color='blue', linewidth=2.5, label='Trend')

    # 标记
    ax.axvline(pd.to_datetime('2020-01-01'), color='red', linestyle='--', alpha=0.5)

    ax.set_title(f"{country}", fontsize=14, fontweight='bold')
    ax.set_ylabel("Calibrated Log GDP Proxy")

plt.suptitle("Global Model Validation: Detecting COVID-19 Shock (Calibrated)", fontsize=18)
plt.tight_layout()
plt.savefig("final_global_validation.png", dpi=300)
print("✅ 最终全球验证图已保存！")
plt.show()