import os
import subprocess
import re

# --- 您需要修改的参数 ---

# 1. 您在第一步中存放所有 .tif 文件的本地文件夹路径
#    请注意：Windows用户路径中的反斜杠'\'需要写成'\\'或'/'
local_data_folder = "D:/虚拟c盘/HNTL_v2"

# 2. 您在第二步中创建的 GEE 影像集的完整路径
#    请将 'YOUR_GEE_USERNAME' 替换为您的GEE用户名
gee_asset_collection_path = "projects/awaran-20130924/assets/H-NTL-v2-Annual"

# --- 脚本主程序 (通常无需修改) ---

print(f"开始上传文件从 '{local_data_folder}' 到 GEE 资产 '{gee_asset_collection_path}'...")

# 获取文件夹中所有的 .tif 文件
files_to_upload = [f for f in os.listdir(local_data_folder) if f.endswith('.tif')]
files_to_upload.sort()  # 按年份排序

for filename in files_to_upload:
    # 从文件名中提取年份 (例如从 'H_NTL_v2...2022.tif' 中提取 '2022')
    match = re.search(r'\d{4}', filename)

    if match:
        year_str = match.group(0)  # 提取找到的年份
    else:
        print(f"  - 警告: 无法在文件名 '{filename}' 中找到四位数的年份，已跳过。")
        continue


    # 构造 GEE 资产的ID，例如 'HNTL_v2_2022'
    asset_id = f"HNTL_v2_{year_str}"
    full_asset_path = f"{gee_asset_collection_path}/{asset_id}"

    # 构造 system:time_start 属性，格式为 YYYY-MM-DD
    time_start_property = f"{year_str}-01-01"

    # 构造完整的本地文件路径
    local_file_path = os.path.join(local_data_folder, filename)

    # 构建 earthengine upload 命令
    # --asset_id 是目标路径
    # --time_start 是影像的起始时间
    # -p 是创建所有必需的父文件夹/集合
    upload_command = [
        "earthengine", "upload", "image",
        f"--asset_id={full_asset_path}",
        f"--time_start={time_start_property}",
        "-p",  # 如果集合不存在，会自动创建
        local_file_path
    ]

    print(f"\n正在准备上传: {filename}")
    print(f"  -> GEE 目标ID: {full_asset_path}")
    print(f"  -> 设置 time_start 为: {time_start_property}")

    # 执行命令
    try:
        subprocess.run(upload_command, check=True)
        print(f"  ✅ 命令已成功提交！请稍后在 GEE Code Editor 的 'Tasks' 标签页中查看上传进度。")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ 命令执行失败: {e}")
    except FileNotFoundError:
        print("  ❌ 错误: 'earthengine' 命令未找到。请确保您已正确安装 google-cloud-sdk 和 earthengine-api。")
        break

print("\n所有上传任务均已提交。")