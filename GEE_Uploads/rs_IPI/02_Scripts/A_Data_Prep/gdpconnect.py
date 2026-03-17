import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================
SATELLITE_DATA_PATH = "../../01_Processed_Data/merged_annual_temp.csv"
# 使用你刚才确认正确的长路径
GDP_DATA_PATH = r"/rs-fi/00_Raw_Data/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_254306\API_NY.GDP.MKTP.CD_DS2_en_csv_v2_254306.csv"

# ================= 核心逻辑 =================
print("🚀 开始 Stage 2：关联 GDP 数据 (基于名称匹配版)")

# --- 1. 读取卫星数据 ---
print("1. 读取卫星数据...")
if not os.path.exists(SATELLITE_DATA_PATH):
    print(f"❌ 错误：找不到文件 {SATELLITE_DATA_PATH}")
    exit()

df_sat = pd.read_csv(SATELLITE_DATA_PATH)
print(f"   -> 卫星数据: {len(df_sat)} 行")

# --- 2. 读取 GDP 数据 (跳过元数据) ---
print("2. 读取 GDP 数据...")
if not os.path.exists(GDP_DATA_PATH):
    print(f"❌ 错误：找不到文件 {GDP_DATA_PATH}")
    exit()

# 定位表头
skip_rows = 0
with open(GDP_DATA_PATH, 'r', encoding='utf-8', errors='ignore') as f:
    for i, line in enumerate(f):
        if "Country Code" in line and "1960" in line:
            skip_rows = i
            break

# 读取
df_gdp_raw = pd.read_csv(GDP_DATA_PATH, skiprows=skip_rows)
print(f"   -> GDP数据读取成功 (跳过前 {skip_rows} 行)")

# --- 3. 数据清洗与转换 ---
print("3. 转换 GDP 格式...")

# 宽表变长表
year_cols = [c for c in df_gdp_raw.columns if c.isdigit()]
df_gdp_melt = df_gdp_raw.melt(
    id_vars=['Country Name', 'Country Code'],
    value_vars=year_cols,
    var_name='year',
    value_name='GDP'
)

# 类型转换
df_gdp_melt['year'] = pd.to_numeric(df_gdp_melt['year'], errors='coerce')
df_gdp_melt['GDP'] = pd.to_numeric(df_gdp_melt['GDP'], errors='coerce')
df_gdp_melt = df_gdp_melt.dropna(subset=['year', 'GDP'])
df_gdp_melt['year'] = df_gdp_melt['year'].astype(int)

# --- 4. 名称标准化与映射 (关键修复步骤) ---
print("4. 执行名称标准化与匹配...")


# 标准化函数: 转大写，去空格
def normalize_name(df, col_name):
    df[col_name] = df[col_name].astype(str).str.upper().str.strip()


normalize_name(df_sat, 'country_na')
normalize_name(df_gdp_melt, 'Country Name')

# 手动修正字典 (解决常见名称不一致)
# 左边是卫星数据(LSIB)的名字，右边是世界银行(WB)的名字
name_map = {
    "TURKEY": "TURKIYE",
    "RUSSIA": "RUSSIAN FEDERATION",
    "IRAN": "IRAN, ISLAMIC REP.",
    "VENEZUELA": "VENEZUELA, RB",
    "SYRIA": "SYRIAN ARAB REPUBLIC",
    "VIETNAM": "VIET NAM",
    "BOLIVIA": "BOLIVIA",
    "LAOS": "LAO PDR",
    "CONGO, DEMOCRATIC REPUBLIC OF THE": "CONGO, DEM. REP.",
    "CONGO, REPUBLIC OF THE": "CONGO, REP.",
    "EGYPT": "EGYPT, ARAB REP.",
    "SOUTH KOREA": "KOREA, REP.",
    "NORTH KOREA": "KOREA, DEM. PEOPLE'S REP.",
    "YEMEN": "YEMEN, REP.",
    "GAMBIA, THE": "GAMBIA, THE",  # 保持一致
    "BAHAMAS, THE": "BAHAMAS, THE",
    "SLOVAKIA": "SLOVAK REPUBLIC",
    "KYRGYZSTAN": "KYRGYZ REPUBLIC",
    "MACEDONIA": "NORTH MACEDONIA",
    "CZECH REPUBLIC": "CZECHIA"
}

# 应用修正
df_sat['country_na'] = df_sat['country_na'].replace(name_map)

# --- 5. 合并数据 ---
# 改用 'country_na' (卫星) 和 'Country Name' (GDP) 进行连接
merged_data = pd.merge(
    df_sat,
    df_gdp_melt,
    left_on=['country_na', 'year'],
    right_on=['Country Name', 'year'],
    how='inner'
)

# --- 6. 保存 ---
if len(merged_data) > 0:
    output_file = "../../01_Processed_Data/final_training_data.csv"
    # 去除重复列
    cols_to_keep = ['country_na', 'country_co', 'year', 'GDP', 'NTL_mean', 'NTL_sum', 'NDVI_mean', 'Precip_mean']
    # 仅保留存在的列
    final_cols = [c for c in cols_to_keep if c in merged_data.columns]
    final_df = merged_data[final_cols]

    final_df.to_csv(output_file, index=False)
    print(f"\n✅ 匹配成功！")
    print(f"   最终样本数: {len(final_df)}")
    print(f"   匹配到的国家数: {final_df['country_na'].nunique()}")
    print(f"   已保存为: {output_file}")
    print("\n👉 完美！请立刻运行 Stage 3 进行训练！")
else:
    print("\n❌ 依然为空。请手动检查以下名称示例，看看差异在哪里：")
    print(f"   卫星数据名称: {sorted(df_sat['country_na'].unique())[:10]}")
    print(f"   GDP 数据名称: {sorted(df_gdp_melt['Country Name'].unique())[:10]}")