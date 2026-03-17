import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# ================= 路径配置区域 (自动适配你的目录结构) =================
# 获取当前脚本所在的目录 (02_Scripts/D_Application)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录 (rs-fi) - 往上跳两级
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

# 1. 训练数据路径 (已处理的年度数据)
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, '01_Processed_Data', 'final_training_data.csv')

# 2. 月度原始数据文件夹 (GEE 导出的 csv 文件夹)
MONTHLY_RAW_FOLDER = os.path.join(PROJECT_ROOT, '00_Raw_Data', 'Target_Monthly_VNP46A2')

# 3. 结果图片保存路径
OUTPUT_FIG_PATH = os.path.join(PROJECT_ROOT, '04_Results_Figures', 'final_turkey_earthquake_impact.png')

# 4. 结果CSV保存路径
OUTPUT_CSV_PATH = os.path.join(PROJECT_ROOT, '01_Processed_Data', 'Turkey_Earthquake_Analysis_Result.csv')

# ================= 参数配置 =================
TARGET_COUNTRY_NAME = "TURKEY"  # 土耳其
TARGET_EVENT_DATE = "2023-02-06"  # 地震日期

print(f"🚀 开始执行土耳其地震分析...")
print(f"   项目根目录: {PROJECT_ROOT}")

# ================= 步骤 1: 训练“月度专用”模型 =================
print("\n[Step 1] 训练基于 NTL_mean 的轻量级模型...")

if not os.path.exists(TRAIN_DATA_PATH):
    print(f"❌ 错误：找不到训练数据: {TRAIN_DATA_PATH}")
    exit()

df_train = pd.read_csv(TRAIN_DATA_PATH)

# 特征工程 (使用 NTL_mean 适配月度数据)
df_train['log_gdp'] = np.log1p(df_train['GDP'])
df_train['log_ntl_mean'] = np.log1p(df_train['NTL_mean'])

# 训练特征
features = ['log_ntl_mean', 'NDVI_mean', 'Precip_mean']
target = 'log_gdp'

df_clean = df_train.dropna(subset=features + [target])
X_train = df_clean[features]
y_train = df_clean[target]

# 训练 RF
rf_monthly = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_monthly.fit(X_train, y_train)
print(f"   -> 模型训练完成 (R²: {rf_monthly.score(X_train, y_train):.4f})")

# ================= 步骤 2: 加载并清洗月度数据 =================
print("\n[Step 2] 加载月度测试数据...")

monthly_files = glob.glob(os.path.join(MONTHLY_RAW_FOLDER, "*.csv"))
print(f"   -> 在 {os.path.basename(MONTHLY_RAW_FOLDER)} 中发现 {len(monthly_files)} 个文件")

df_list = []
for f in monthly_files:
    try:
        temp = pd.read_csv(f)
        # 统一列名大小写，防止 target 和 TARGET 不一致
        temp.columns = [c.lower() for c in temp.columns]

        # 选取需要的列 (根据GEE导出情况，这里做宽容处理)
        # 假设列名可能是 ntl_mean 或 mean
        cols_map = {
            'ntl_mean': 'NTL_mean', 'mean': 'NTL_mean',  # 兼容
            'ndvi_mean': 'NDVI_mean',
            'precip_mean': 'Precip_mean',
            'country_na': 'country_na',
            'date': 'date'
        }

        # 重命名
        temp = temp.rename(columns=cols_map)

        # 只要包含关键列就保留
        if set(['NTL_mean', 'country_na', 'date']).issubset(temp.columns):
            df_list.append(temp)
    except Exception as e:
        print(f"   ⚠️ 读取失败 {os.path.basename(f)}: {e}")

if not df_list:
    print("❌ 错误：没有加载到有效数据，请检查 00_Raw_Data 文件夹下的月度数据。")
    exit()

df_monthly = pd.concat(df_list, ignore_index=True)
df_monthly['date'] = pd.to_datetime(df_monthly['date'])
df_monthly['country_na'] = df_monthly['country_na'].astype(str).str.upper().str.strip()

# ================= 步骤 3: 筛选土耳其并预测 =================
print(f"\n[Step 3] 筛选 {TARGET_COUNTRY_NAME} 并预测经济活力...")

# 筛选
turkey_data = df_monthly[df_monthly['country_na'] == TARGET_COUNTRY_NAME].copy()

if len(turkey_data) == 0:
    print(f"❌ 错误：未找到 {TARGET_COUNTRY_NAME} 的数据。")
    print(f"   数据中包含的国家示例: {df_monthly['country_na'].unique()[:10]}")
    exit()

# 构造特征
turkey_data['log_ntl_mean'] = np.log1p(turkey_data['NTL_mean'])

# 预测
turkey_data['pred_log_gdp'] = rf_monthly.predict(turkey_data[features])

# 去季节性平滑 (3个月滑动平均)
turkey_data = turkey_data.sort_values('date')
turkey_data['trend'] = turkey_data['pred_log_gdp'].rolling(window=3, center=True).mean()

# 保存中间结果到 Processed Data
turkey_data.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"   -> 中间数据已保存至: {os.path.basename(OUTPUT_CSV_PATH)}")

# ================= 步骤 4: 可视化与分析 =================
print(f"\n[Step 4] 生成分析图表...")

plt.figure(figsize=(12, 6))

# 1. 原始预测 (浅色)
plt.plot(turkey_data['date'], turkey_data['pred_log_gdp'],
         label='Raw Prediction (Monthly)', color='lightgray', linestyle='--', marker='o', alpha=0.5)

# 2. 平滑趋势 (深色)
plt.plot(turkey_data['date'], turkey_data['trend'],
         label='Economic Trend (3-Month Smooth)', color='#1f77b4', linewidth=2.5)

# 3. 地震线
event_date = pd.to_datetime(TARGET_EVENT_DATE)
plt.axvline(x=event_date, color='#d62728', linestyle='-', linewidth=2, label='Earthquake (Feb 6, 2023)')

# 4. 图表装饰
plt.title(f"Economic Resilience Analysis: Turkey Earthquake Impact", fontsize=16, fontweight='bold')
plt.ylabel("Satellite-Derived Economic Index (Log Scale)", fontsize=12)
plt.xlabel("Time", fontsize=12)
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3, linestyle='--')

# 聚焦时间轴 (2022-2023)
plt.xlim(pd.to_datetime("2022-01-01"), pd.to_datetime("2023-12-31"))

# 保存图片
plt.tight_layout()
plt.savefig(OUTPUT_FIG_PATH, dpi=300)
print(f"📊 图表已保存至: 04_Results_Figures/{os.path.basename(OUTPUT_FIG_PATH)}")
plt.show()

# ================= 5. 定量冲击计算 =================
# 定义震前震后窗口 (例如震前5个月 vs 震后5个月)
pre_window = turkey_data[
    (turkey_data['date'] >= "2022-09-01") & (turkey_data['date'] < "2023-02-01")
    ]['trend'].mean()

post_window = turkey_data[
    (turkey_data['date'] >= "2023-02-01") & (turkey_data['date'] < "2023-07-01")
    ]['trend'].mean()

if pd.notna(pre_window) and pd.notna(post_window):
    impact = (post_window - pre_quake) / pre_quake * 100
    print(f"\n📉 [冲击评估报告]")
    print(f"   震前基准 (2022.09-2023.01): {pre_window:.4f}")
    print(f"   震后水平 (2023.02-2023.06): {post_window:.4f}")
    print(f"   经济活动指数变动: {impact:.2f}%")
else:
    print("\n⚠️ 数据不足，无法计算定量冲击 (可能是滑动平均导致的头尾缺失)。")