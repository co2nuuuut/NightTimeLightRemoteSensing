import pandas as pd
import os

# ================== 📂 路径配置 ==================
BASE_DIR = r"D:\虚拟c盘\大创项目\GEE_Uploads\rs-fi"
RAW_FILE = os.path.join(BASE_DIR, "00_Raw_Data", "testgdp.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "01_Processed_Data", "quarterly_gdp_true.csv")
# =================================================

print("🚀 步骤 1/2: 清洗 OECD 原始数据 (SDMX-CSV 适配版)...")

if not os.path.exists(RAW_FILE):
    print(f"❌ 错误：找不到文件 {RAW_FILE}")
    exit()

# 1. 读取数据
try:
    df = pd.read_csv(RAW_FILE)
except:
    df = pd.read_csv(RAW_FILE, header=0)

# 2. 精准映射列名 (针对你的文件结构)
col_map = {}
# 直接根据你提供的列名进行硬编码映射，不再盲猜
if 'REF_AREA' in df.columns:
    col_map['REF_AREA'] = 'country_info'
elif 'Reference area' in df.columns:
    col_map['Reference area'] = 'country_info'

if 'TIME_PERIOD' in df.columns:
    col_map['TIME_PERIOD'] = 'quarter'
elif 'Time period' in df.columns:
    col_map['Time period'] = 'quarter'

if 'OBS_VALUE' in df.columns:
    col_map['OBS_VALUE'] = 'value'
elif 'Value' in df.columns:
    col_map['Value'] = 'value'

print(f"   -> 映射关系: {col_map}")

# 应用映射
df = df.rename(columns=col_map)

# 检查是否成功
required = ['country_info', 'quarter', 'value']
if not all(c in df.columns for c in required):
    print(f"❌ 错误：列名匹配失败，当前列: {df.columns.tolist()}")
    # 尝试打印前几列帮助排查
    print("请检查你的CSV表头是否包含 REF_AREA, TIME_PERIOD, OBS_VALUE")
    exit()

# 3. 数据转换
# 转数值
df['value'] = pd.to_numeric(df['value'], errors='coerce')
df = df.dropna(subset=['value'])

# 判断是否需要 +100 (增长率转指数)
mean_val = df['value'].mean()
if mean_val < 50:
    print("   -> 检测为增长率数据，正在转换为指数 (100 + value)...")
    df['gdp_index'] = 100 + df['value']
else:
    print("   -> 检测为指数数据，直接使用...")
    df['gdp_index'] = df['value']

# 4. 国家代码处理 (智能判断)
print("   -> 正在处理国家代码...")
sample_country = str(df['country_info'].iloc[0])
print(f"   -> 样本国家标识: '{sample_country}'")

# 定义映射字典 (G20)
name_to_code = {
    "China": "CHN", "People's Republic of China": "CHN",
    "United States": "USA", "Germany": "DEU", "India": "IND",
    "United Kingdom": "GBR", "France": "FRA", "Italy": "ITA",
    "Brazil": "BRA", "Türkiye": "TUR", "Turkey": "TUR",
    "Indonesia": "IDN", "Japan": "JPN", "Korea": "KOR",
    "Canada": "CAN", "Australia": "AUS", "Mexico": "MEX",
    "Russia": "RUS", "Russian Federation": "RUS", "South Africa": "ZAF",
    "Saudi Arabia": "SAU", "Argentina": "ARG", "European Union": "EUU"
}

# 如果样本长度为3且全是大写字母，大概率已经是 ISO 代码了
if len(sample_country) == 3 and sample_country.isupper():
    print("   -> 检测到列中已经是国家代码，直接使用。")
    df['country_code'] = df['country_info']
else:
    print("   -> 检测到列中是国家全名，正在进行映射...")
    df['country_code'] = df['country_info'].map(name_to_code)

# 过滤掉无法识别的国家 (如 OECD 总计数据)
df_clean = df.dropna(subset=['country_code'])

# 筛选时间 (2019Q1 以后)
df_clean = df_clean[df_clean['quarter'] >= '2019-Q1']

# 5. 保存
final_cols = ['country_code', 'quarter', 'gdp_index']
# 去重 (防止同一国家同一季度有多行数据)
df_clean = df_clean.drop_duplicates(subset=['country_code', 'quarter'])
df_clean[final_cols].to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ 清洗完成！已保存至: {OUTPUT_FILE}")
print(f"   包含国家 ({len(df_clean['country_code'].unique())}个): {df_clean['country_code'].unique()}")
print(f"   数据条数: {len(df_clean)}")