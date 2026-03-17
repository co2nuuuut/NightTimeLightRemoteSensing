import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import glob
import os

# ================= 配置区域 =================
BASE_DIR = r"D:\虚拟c盘\大创项目\GEE_Uploads\rs-fi"
MODEL_PATH = os.path.join(BASE_DIR, "03_Models", "best_model.pkl")
CALIBRATOR_PATH = os.path.join(BASE_DIR, "03_Models", "bias_corrector.pkl")
DATA_FOLDER = os.path.join(BASE_DIR, "00_Raw_Data", "Global_Validation_Fixed")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "01_Processed_Data", "final_training_data.csv")

print("🚀 开始最终验证：季节性调整 (SA) + 指数化 (Index)...")

# ================= 1. 内置官方 GDP 同比数据 (用于重建真实指数) =================
# 来源: OECD / 国家统计局 (YoY %)
OFFICIAL_YOY = {
    'CHN': {
        '2019Q1': 6.0, '2019Q2': 6.0, '2019Q3': 5.8, '2019Q4': 5.8,
        '2020Q1': -6.8, '2020Q2': 3.2, '2020Q3': 4.9, '2020Q4': 6.5,
        '2021Q1': 18.3, '2021Q2': 7.9, '2021Q3': 4.9, '2021Q4': 4.0,
        '2022Q1': 4.8, '2022Q2': 0.4, '2022Q3': 3.9, '2022Q4': 2.9,
        '2023Q1': 4.5, '2023Q2': 6.3
    },
    'USA': {
        '2019Q1': 2.3, '2019Q2': 2.1, '2019Q3': 2.3, '2019Q4': 2.6,
        '2020Q1': 0.6, '2020Q2': -9.1, '2020Q3': -2.9, '2020Q4': -2.3,
        '2021Q1': 0.5, '2021Q2': 12.2, '2021Q3': 4.9, '2021Q4': 5.5,
        '2022Q1': 3.5, '2022Q2': 1.9, '2022Q3': 1.7, '2022Q4': 0.7
    },
    'DEU': {
        '2019Q1': 1.0, '2019Q2': 0.0, '2019Q3': 0.8, '2019Q4': 0.4,
        '2020Q1': -2.2, '2020Q2': -11.3, '2020Q3': -4.0, '2020Q4': -3.6,
        '2021Q1': -2.7, '2021Q2': 10.4, '2021Q3': 2.9, '2021Q4': 1.2,
        '2022Q1': 4.0, '2022Q2': 1.7, '2022Q3': 1.3, '2022Q4': 0.3
    },
    'IND': {
        '2019Q1': 5.0, '2019Q2': 4.5, '2019Q3': 4.4, '2019Q4': 3.3,
        '2020Q1': 3.0, '2020Q2': -23.9, '2020Q3': -6.6, '2020Q4': 0.7,
        '2021Q1': 1.6, '2021Q2': 20.1, '2021Q3': 8.4, '2021Q4': 5.4,
        '2022Q1': 4.1, '2022Q2': 13.5, '2022Q3': 6.3, '2022Q4': 4.4
    }
}

# ================= 2. 卫星数据处理 =================
print("1. 加载并校正卫星数据...")
model = joblib.load(MODEL_PATH)
calibrator = joblib.load(CALIBRATOR_PATH)

# 计算面积系数
df_train = pd.read_csv(TRAIN_DATA_PATH)
df_train['country_na'] = df_train['country_na'].astype(str).str.upper().str.strip()
df_train['area_factor'] = df_train['NTL_sum'] / df_train['NTL_mean']
country_area_map = df_train.groupby('country_na')['area_factor'].median().to_dict()

# 读取 CSV
all_files = glob.glob(os.path.join(DATA_FOLDER, "Global_Validation_Fixed_*.csv"))
df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
df['date'] = pd.to_datetime(df['date'])
df['country_na'] = df['country_na'].astype(str).str.upper().str.strip()
iso_map = {"CHINA": "CHN", "UNITED STATES": "USA", "GERMANY": "DEU", "INDIA": "IND"}
df['iso_code'] = df['country_na'].map(iso_map)
df = df.dropna(subset=['iso_code'])

# 映射面积 & 特征工程
df['area_factor'] = df['country_na'].map(country_area_map)
df = df.dropna(subset=['area_factor'])
df['calc_ntl_sum'] = df['NTL_mean'] * df['area_factor']
df['log_ntl'] = np.log1p(df['calc_ntl_sum'])
features = ['log_ntl', 'NDVI_mean', 'Precip_mean']

# 预测
df_clean = df.dropna(subset=features).copy()
raw_pred = model.predict(df_clean[features])
df_clean['pred_log_gdp'] = calibrator.predict(raw_pred.reshape(-1, 1))

# ================= 3. 核心算法：季节性调整 + 指数化 =================
print("2. 执行季节性调整 (Seasonal Adjustment)...")
final_results = []

for iso in ['CHN', 'USA', 'DEU', 'IND']:
    # 1. 提取月度数据并排序
    sub = df_clean[df_clean['iso_code'] == iso].set_index('date').sort_index()
    if sub.empty: continue

    # 2. 季节性调整 (SA)
    # 使用12个月滑动平均来剔除季节性，提取纯趋势
    # 这是经济学处理 Time Series 的标准做法
    sub['gdp_sa'] = sub['pred_log_gdp'].rolling(window=12, center=True, min_periods=6).mean()

    # 3. 季度聚合 (取SA后的均值)
    sub_q = sub['gdp_sa'].resample('Q').mean().reset_index()
    sub_q['quarter'] = sub_q['date'].dt.to_period('Q').astype(str)

    # 4. 卫星指数化 (Satellite Index)
    # 以 2019 平均值为基准 (Index=100)
    base_val = sub_q[sub_q['date'].dt.year == 2019]['gdp_sa'].mean()
    if pd.isna(base_val): continue

    # Log差值转指数公式: Index = 100 * exp(Log_Current - Log_Base)
    sub_q['sat_index'] = 100 * np.exp(sub_q['gdp_sa'] - base_val)

    # 5. 官方 GDP 指数化 (Official Index)
    # 根据 YoY 数据倒推指数
    yoy_data = OFFICIAL_YOY.get(iso, {})
    official_index = {}
    # 假设 2018 全年为 100 (虚拟基准)，递推算出 2019-2023
    # 这里简化处理：直接设定 2019 平均为 100，根据 YoY 调整每一季的相对位置
    # 简单算法：True_Index_t = True_Index_{t-4} * (1 + YoY/100)
    # 我们先生成一个假序列，最后再归一化到 2019=100

    # 为了画图简单，我们采用“对齐法”：
    # 直接把 YoY 数据拿来，生成一个累积增长曲线
    quarters = sorted(yoy_data.keys())
    dummy_index = [100.0] * len(quarters)  # 占位

    # 这是一个简化的展示逻辑：
    # 只要看 "趋势形态" 是否一致即可。
    # 我们把真实 YoY 转换为 "趋势分" (Trend Score)

    # 更严谨的做法：
    # 我们只画 2019Q1 之后。
    # 设 2019Q1 真值为 100。
    # 2020Q1 真值 = 100 * (1 + YoY_2020Q1/100)
    # 这种方法虽然有误差，但能看清“坑”在哪里。

    # 重新构建真实指数 dataframe
    true_idx_list = []
    # 初始化：假设 2019 四个季度平均 100，根据 YoY 大致反推
    # 这里直接手动构造几个关键点用于画图对比 (归一化后的)
    # 为了代码简洁，我们直接读取 YoY 字典，并假设 2018Qx 为基准
    # 实际上，只要看“拐点”对不对得上。

    sub_q['country'] = iso
    final_results.append(sub_q)

df_final = pd.concat(final_results)

# ================= 4. 绘图 (双轴对比法) =================
# 左轴：卫星指数 (100基准)
# 右轴：官方 YoY (看跌幅深度)
# 这样可以避免指数推算的累积误差，直接对比“卫星的走势”和“官方的增长率”

sns.set(style="white")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, iso in enumerate(['CHN', 'USA', 'DEU', 'IND']):
    ax1 = axes[i]
    data = df_final[df_final['country'] == iso].dropna()

    if data.empty: continue

    # 绘制卫星指数 (蓝色实线) - 代表经济总量趋势
    color1 = 'tab:blue'
    ax1.plot(data['date'], data['sat_index'], color=color1, linewidth=3, label='Satellite Economic Index (SA)')
    ax1.set_ylabel('Economic Activity Index (2019=100)', color=color1, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # 绘制官方 YoY (红色虚线) - 代表增长动力
    ax2 = ax1.twinx()
    color2 = 'tab:red'

    # 准备 YoY 数据
    yoy_dict = OFFICIAL_YOY.get(iso, {})
    dates = []
    vals = []
    for q_str, val in yoy_dict.items():
        # 2019Q1 -> 2019-03-31
        dt = pd.to_datetime(
            q_str.replace('Q1', '-03-31').replace('Q2', '-06-30').replace('Q3', '-09-30').replace('Q4', '-12-31'))
        dates.append(dt)
        vals.append(val)

    # 排序
    sorted_pairs = sorted(zip(dates, vals))
    d_sorted, v_sorted = zip(*sorted_pairs)

    ax2.plot(d_sorted, v_sorted, color=color2, linestyle='--', marker='s', linewidth=2, alpha=0.7,
             label='Official GDP YoY %')
    ax2.set_ylabel('Official GDP Growth (%)', color=color2, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color2)

    # 添加 0 轴线给 YoY
    ax2.axhline(0, color='red', linewidth=0.5, alpha=0.5)

    # 标记 COVID
    ax1.axvline(pd.to_datetime('2020-01-01'), color='gray', linestyle=':', label='COVID-19 Start')

    # 计算相关性 (虽然一个是指数一个是增速，但看波峰波谷是否对应)
    # 我们主要看视觉吻合度：
    # 当红线(YoY)暴跌时，蓝线(Index)应该转头向下。

    ax1.set_title(f"{iso}: Satellite Index vs Official Growth", fontsize=14, fontweight='bold')

plt.tight_layout()
save_path = os.path.join(BASE_DIR, "04_Results_Figures", "final_sa_index_validation.png")
plt.savefig(save_path, dpi=300)
print(f"\n✅ 最终验证图已保存至: {save_path}")
plt.show()