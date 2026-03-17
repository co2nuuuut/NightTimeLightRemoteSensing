import pandas as pd
import io
import requests

print("🚀 开始下载 OECD 季度 GDP 数据 (分库修正版)...")

# 通用下载函数
def download_oecd_data(dataset_id, countries, description):
    # VIXOBSA: Volume index, seasonally adjusted (GDP指数，剔除通胀和季节性，适合看趋势)
    url = f"https://stats.oecd.org/sdmx-json/data/{dataset_id}/{countries}.B1_GE.VIXOBSA.Q/all?contentType=csv"
    print(f"   -> 正在请求 {description} 数据...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        print(f"      ⚠️ {description} 下载失败 (可能部分国家数据缺失): {e}")
        return pd.DataFrame()

# ================= 1. 分组下载 =================

# 组1: OECD 成员国 (发达经济体)
# 包含: 美国, 德国, 英国, 法国, 意大利, 土耳其, 韩国, 日本, 加拿大, 澳大利亚, 墨西哥
oecd_countries = "USA+DEU+GBR+FRA+ITA+TUR+KOR+JPN+CAN+AUS+MEX"
df_oecd = download_oecd_data("QNA", oecd_countries, "OECD成员国")

# 组2: 非 OECD 成员国 (主要发展中国家)
# 包含: 中国, 印度, 印度尼西亚, 巴西, 俄罗斯, 南非
non_oecd_countries = "CHN+IND+IDN+BRA+RUS+ZAF"
df_non_oecd = download_oecd_data("QNA_NON_OECD", non_oecd_countries, "非OECD成员国")

# ================= 2. 合并与清洗 =================
if df_oecd.empty and df_non_oecd.empty:
    print("\n❌ 错误：所有下载均失败。请检查网络连接。")
    exit()

print("   -> 正在合并数据...")
df_raw = pd.concat([df_oecd, df_non_oecd], ignore_index=True)

# 清洗列
# OECD CSV 格式通常为: LOCATION, TIME, Value
df = df_raw[['LOCATION', 'TIME', 'Value']].copy()
df.columns = ['country_code', 'quarter', 'gdp_index']

# 筛选时间 (2019年以后)
df = df[df['quarter'] >= '2019-Q1']

print(f"   -> 获取到 {len(df)} 条季度记录，涵盖 {df['country_code'].nunique()} 个国家")
print(f"   -> 国家列表: {df['country_code'].unique()}")

# ================= 3. 保存 =================
output_file = "quarterly_gdp_true.csv"
df.to_csv(output_file, index=False)
print(f"\n✅ 成功！真实季度 GDP 指数已保存为: {output_file}")