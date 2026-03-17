import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import re
import os
import glob
import numpy as np
import json
from statsmodels.tsa.seasonal import seasonal_decompose

# --- 以下为核心配置和函数，大部分来自您的原始代码 ---

# 设置 Matplotlib 后端，确保在无图形界面的服务器上也能运行
matplotlib.use('Agg')
# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')


def set_chinese_font():
    """设置 Matplotlib 以正确显示中文字符。"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        print("✅ 绘图字体已设置为 'SimHei' (黑体)。")
    except Exception as e:
        print(f"⚠️ 警告：未能找到 'SimHei' 字体。将使用默认字体。({e})")
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False


def create_and_save_plot(row_data, output_filename):
    """
    【修改版】:
    此函数基于您原始的 create_resilience_plot_base64 函数。
    - 核心的数据处理和绘图逻辑保持不变。
    - 最终输出从 base64 字符串改为了直接保存为 PNG 文件。
    """
    # --- 1. 数据提取与预处理 ---
    time_series_data = {}
    for col_name, value in row_data.items():
        match = re.match(r'NTL_(\d{4})_(\d{2})', col_name)
        if match:
            year, week_str = match.groups()
            if week_str == '00' or int(week_str) > 53: continue
            try:
                time_stamp = pd.to_datetime(f'{year}-{week_str}-1', format='%G-%V-%u')
            except ValueError:
                continue
            if pd.notna(value) and value > 0:
                time_series_data[time_stamp] = float(value)
            else:
                time_series_data[time_stamp] = np.nan

    s = pd.Series(time_series_data).sort_index()
    s_interpolated = s.interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(
        method='ffill')

    if s_interpolated.isnull().all() or len(s_interpolated) < 104:
        print(f"  -> 🔴 错误: {row_data.get('location_name', 'N/A')} 的有效周数据不足104个，无法进行分析。")
        return False

    # --- 2. 时间序列分析 ---
    s_smoothed = s_interpolated.ewm(alpha=0.7, adjust=False).mean()
    decomposition = seasonal_decompose(s_smoothed, model='additive', period=52)
    trend_component = decomposition.trend.interpolate(method='linear', limit_direction='both').fillna(
        method='bfill').fillna(method='ffill')
    earthquake_date = pd.Timestamp(row_data['date'])
    pre_earthquake_trend = trend_component[trend_component.index < earthquake_date]
    baseline_value = pre_earthquake_trend.iloc[-1] if not pre_earthquake_trend.empty else None

    # --- 3. 绘图 ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.plot(s_smoothed.index, s_smoothed.values, color='lightgray', linestyle='-', linewidth=1.5,
            label='原始周度数据 (波动)')
    ax.plot(trend_component.index, trend_component.values, color='black', marker='.', markersize=5, linestyle='-',
            label='城市发展趋势 (已校正季节性)')
    ax.axvline(x=earthquake_date, color='red', linestyle='--', linewidth=2.5,
               label=f'地震发生 ({earthquake_date.date()})')
    if baseline_value is not None:
        ax.axhline(y=baseline_value, color='green', linestyle=':', linewidth=2, label='震前发展水平基准线')

    zoom_window_weeks = 52
    start_date = earthquake_date - pd.DateOffset(weeks=zoom_window_weeks)
    end_date = earthquake_date + pd.DateOffset(weeks=zoom_window_weeks)
    ax.set_xlim(start_date, end_date)

    zoomed_data = trend_component[start_date:end_date]
    if not zoomed_data.empty:
        y_min = zoomed_data.min() * 0.95
        y_max = zoomed_data.max() * 1.05
        ax.set_ylim(y_min, y_max)

    # --- 4. 设置图表样式与标签 ---
    ax.set_title(f'地震冲击响应分析: {row_data.get("location_name")}', fontsize=16)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('周度夜光趋势值', fontsize=12)
    ax.legend(fontsize=10)
    fig.autofmt_xdate()
    plt.tight_layout()

    # --- 5. 保存图像文件 (这是与原始函数的主要区别) ---
    try:
        fig.savefig(output_filename, format='png', dpi=150)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"  -> 🔴 错误：保存文件 '{output_filename}' 时发生错误: {e}")
        return False


def find_and_generate_specific_plot(directory_path, target_location_name, output_filename):
    """
    主逻辑函数：
    1. 读取指定目录下的所有CSV文件并合并。
    2. 在合并后的数据中查找特定地点的地震记录。
    3. 如果找到，则调用绘图函数为其生成并保存图像。
    """
    set_chinese_font()

    # 检查路径是否存在
    if not os.path.isdir(directory_path):
        print(f"🔴 错误：指定的路径 '{directory_path}' 不是一个有效的文件夹。")
        return

    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    if not csv_files:
        print(f"🔴 错误：在文件夹 '{directory_path}' 中没有找到任何 .csv 文件。")
        return

    print(f"🔍 正在从 {len(csv_files)} 个CSV文件中加载数据...")
    try:
        all_data_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        # 为防止重复记录，保留第一次出现的记录
        all_data_df.drop_duplicates(subset=['location_name', 'date'], keep='first', inplace=True)
    except Exception as e:
        print(f"🔴 错误：读取或合并CSV文件时出错: {e}")
        return

    print(f"✅ 数据加载完成，共计 {len(all_data_df)} 条独立地震记录。")
    print(f"🔎 正在查找目标事件: '{target_location_name}'...")

    # 查找匹配的行
    target_row_df = all_data_df[all_data_df['location_name'] == target_location_name]

    if target_row_df.empty:
        print("\n-----------------------------------------------------")
        print(f"⚠️ 未能找到与 '{target_location_name}' 匹配的地震记录。")
        print("请检查：")
        print("  1. location_name 是否与CSV文件中的完全一致（包括空格和大小写）。")
        print("  2. CSV文件是否确实包含这条记录。")
        print("-----------------------------------------------------")
    else:
        # 即使有多个匹配项，也只处理第一个
        target_row_series = target_row_df.iloc[0]
        print(f"🎯 已找到目标事件！开始生成图表...")

        success = create_and_save_plot(target_row_series, output_filename)

        print("\n-----------------------------------------------------")
        if success:
            print(f"🎉 任务完成！图表已成功保存为 '{output_filename}'。")
        else:
            print("❌ 生成图表时发生错误，请查看上面的日志信息。")
        print("-----------------------------------------------------")


# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 请在这里配置您的CSV文件所在的文件夹路径
    csv_directory_path = r"D:\虚拟c盘\GEE_Uploads\EURm\M65"

    # 2. 这是您想从CSV文件中查找并绘图的地震事件的'location_name'
    target_earthquake_name = "168 km SW of Mawu, China"

    # 3. 这是您希望保存的图片文件名
    output_image_filename = "Mawu_China_earthquake_analysis.png"

    # 运行主函数
    find_and_generate_specific_plot(csv_directory_path, target_earthquake_name, output_image_filename)