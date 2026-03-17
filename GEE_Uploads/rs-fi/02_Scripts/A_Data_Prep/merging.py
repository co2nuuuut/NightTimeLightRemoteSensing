import pandas as pd
import glob
import os

# ================= 配置区域 =================
# 请确保这里的路径是你电脑上的真实路径
ANNUAL_FOLDER_PATH = r"../../00_Raw_Data/Global_Annual_VNP46A2"
MONTHLY_FOLDER_PATH = r"../../00_Raw_Data/Target_Monthly_VNP46A2"
# (如果你的文件夹名不是 Annual/Monthly，请自行修改)

# ================= 第一步：合并【全球年度训练数据】 =================
print(f"📂 正在扫描年度数据: {ANNUAL_FOLDER_PATH}")
annual_files = glob.glob(os.path.join(ANNUAL_FOLDER_PATH, "Global_Annual*.csv"))

df_annual_list = []
for f in annual_files:
    try:
        temp_df = pd.read_csv(f)
        if 'system:index' in temp_df.columns:
            temp_df = temp_df.drop(columns=['system:index', '.geo'], errors='ignore')
        df_annual_list.append(temp_df)
    except Exception as e:
        print(f"   ⚠️ 跳过文件: {e}")

if df_annual_list:
    df_annual = pd.concat(df_annual_list, ignore_index=True)
    print(f"✅ 年度合并成功！形状: {df_annual.shape}")
    # 【修改点】用 print 代替 display
    print(df_annual.head(3))
else:
    print("❌ 未找到年度文件")

# ================= 第二步：合并【重点国家月度数据】 =================
print(f"\n📂 正在扫描月度数据: {MONTHLY_FOLDER_PATH}")
monthly_files = glob.glob(os.path.join(MONTHLY_FOLDER_PATH, "Target*.csv"))

df_monthly_list = []
for f in monthly_files:
    try:
        temp_df = pd.read_csv(f)
        if 'system:index' in temp_df.columns:
            temp_df = temp_df.drop(columns=['system:index', '.geo'], errors='ignore')
        df_monthly_list.append(temp_df)
    except Exception as e:
        print(f"   ⚠️ 跳过文件: {e}")

if df_monthly_list:
    df_monthly = pd.concat(df_monthly_list, ignore_index=True)
    if 'date' in df_monthly.columns:
        df_monthly['date'] = pd.to_datetime(df_monthly['date'])

    print(f"✅ 月度合并成功！形状: {df_monthly.shape}")
    # 【修改点】用 print 代替 display
    print(df_monthly.head(3))
else:
    print("❌ 未找到月度文件")

# ================= 保存临时结果 =================
# 为了方便下一步操作，我们将合并好的数据保存到本地
if df_annual_list:
    df_annual.to_csv("merged_annual_temp.csv", index=False)
    print("\n💾 已保存年度合并数据到: merged_annual_temp.csv")

if df_monthly_list:
    df_monthly.to_csv("merged_monthly_temp.csv", index=False)
    print("💾 已保存月度合并数据到: merged_monthly_temp.csv")