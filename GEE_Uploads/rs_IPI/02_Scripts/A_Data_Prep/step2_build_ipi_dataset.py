import pandas as pd
import glob
import os
import requests
import io
import time

# ================= 路径配置 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
RS_DATA_FOLDER = os.path.join(PROJECT_ROOT, '00_Raw_Data', 'Step1_Training_RS_Data_Split')
OUTPUT_CSV_PATH = os.path.join(PROJECT_ROOT, '01_Processed_Data', 'final_monthly_training_data.csv')

# ================= 核心：IPI 代码映射 (FRED Series ID) =================
# 备注：我把美国的换成了更通用的 INDPRO
FRED_MAPPING = {
    'United States': 'INDPRO',  # 美国工业产出 (最权威指标)
    'Turkey': 'TURPROINDMISMEI',  # 土耳其
    'China': 'CHNCPIALLMINMEI',  # 中国 (注意：FRED上中国的IPI可能缺失，暂且试试CPI或跳过)
    'Germany': 'DEUPROINDMISMEI',
    'United Kingdom': 'GBRPROINDMISMEI',
    'Japan': 'JPNPROINDMISMEI',
    'South Korea': 'KORPROINDMISMEI',
    'Russia': 'RUSPROINDMISMEI',
    'India': 'INDPROINDMISMEI',
    'France': 'FRAPROINDMISMEI',
    'Italy': 'ITAPROINDMISMEI',
    'Canada': 'CANPROINDMISMEI',
    'Brazil': 'BRAPROINDMISMEI',
    'South Africa': 'ZAFPROINDMISMEI',
    'Australia': 'AUSPROINDMISMEI',
    'Mexico': 'MEXPROINDMISMEI',
    'Indonesia': 'IDNPROINDMISMEI',
    'Spain': 'ESPPROINDMISMEI',
    'Netherlands': 'NLDPROINDMISMEI',
}

# 伪装浏览器头，防止被反爬拦截
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

print("🚀 [Step 2 Fix] 开始构建数据集 (原生 Requests 版)...")

# --- 1. 合并遥感数据 ---
print("\n1. 合并遥感数据...")
csv_files = glob.glob(os.path.join(RS_DATA_FOLDER, "*.csv"))
df_list = []
for f in csv_files:
    try:
        temp = pd.read_csv(f)
        temp.columns = [c.strip() for c in temp.columns]
        # 简单清洗列名
        temp = temp.rename(columns={'ntl_mean': 'NTL_mean', 'mean': 'NTL_mean'})
        # 只要有国家和日期列就行
        if 'country_na' in temp.columns:
            df_list.append(temp)
    except:
        pass

if not df_list:
    print("❌ 错误：遥感数据文件夹为空！")
    exit()

df_rs = pd.concat(df_list, ignore_index=True)
df_rs['country_na'] = df_rs['country_na'].astype(str).str.strip()
print(f"   -> 遥感数据就绪: {len(df_rs)} 行")

# --- 2. 强力下载 IPI ---
print("\n2. 下载 IPI 数据 (超时重试机制)...")
ipi_list = []

for country, series_id in FRED_MAPPING.items():
    # 检查我们的遥感数据里有没有这个国家，没有就跳过，省流量
    if country not in df_rs['country_na'].unique():
        continue

    print(f"   -> 下载 {country} [{series_id}]...", end="")

    # 构造 FRED 直接下载链接
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    success = False
    error_msg = ""

    # 重试 3 次
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)  # 10秒超时
            if r.status_code == 200:
                # 成功获取
                content = r.content.decode('utf-8')
                df_temp = pd.read_csv(io.StringIO(content))

                # 清洗
                df_temp.columns = ['DATE', 'IPI']
                df_temp['DATE'] = pd.to_datetime(df_temp['DATE'])
                df_temp['year'] = df_temp['DATE'].dt.year
                df_temp['month'] = df_temp['DATE'].dt.month
                df_temp['country_na'] = country

                ipi_list.append(df_temp)
                success = True
                print(" ✅")
                break
            else:
                error_msg = f"HTTP {r.status_code}"
        except Exception as e:
            error_msg = str(e)
            time.sleep(1)  # 歇一秒再试

    if not success:
        print(f" ❌ 失败: {error_msg}")

# --- 3. 合并与保存 ---
if ipi_list:
    df_ipi = pd.concat(ipi_list, ignore_index=True)

    # 合并
    df_final = pd.merge(df_rs, df_ipi, on=['country_na', 'year', 'month'], how='inner')

    if len(df_final) > 0:
        df_final.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\n✅ 成功！最终训练集已生成: {OUTPUT_CSV_PATH}")
        print(f"   样本量: {len(df_final)}")
        print(f"   包含国家: {df_final['country_na'].unique()}")
    else:
        print("\n❌ 合并后数据为空 (可能是年份对不上)。")
else:
    print("\n❌ 所有下载均失败。请使用方案 B (手动下载)。")