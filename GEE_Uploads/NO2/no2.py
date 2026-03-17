# =============================================================================
#
#  地震韧性分析脚本 (V2.1 - 纯NO2数据绘图版)
#
#  作者: [您的名字]
#  日期: 2025-11-10
#
#  功能:
#  1. 读取GEE导出的NO2周度数据。
#  2. (无校正) 直接使用时间序列分解方法分析趋势。
#  3. 为每次地震生成基于NO2趋势的韧性分析图。
#  4. 将所有结果汇总到一个交互式Folium地图中。
#
# =============================================================================

import pandas as pd
import folium
import matplotlib
import matplotlib.pyplot as plt
import re
import base64
import os
import glob
import numpy as np
import json
from statsmodels.tsa.seasonal import seasonal_decompose

# --- 初始设置 ---
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')


def set_chinese_font():
    """设置绘图时支持中文的字体。"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        print("✅ 绘图字体已设置为 'SimHei' (黑体)。")
    except Exception as e:
        print(f"⚠️ 警告：未能找到 'SimHei' 字体，中文可能显示为方框。错误: {e}")
    plt.rcParams['axes.unicode_minus'] = False


def create_resilience_plot_base64(row_data):
    """
    【V2.1 - 适配NO2数据】:
    - 读取以 "NO2_" 开头的列。
    - 使用与NTL版本相同的季节性分解方法进行趋势分析。
    - 更新图表的标题和标签以反映NO2数据。
    """
    time_series_data = {}

    # --- 1. 数据提取 (核心修改点) ---
    for col_name, value in row_data.items():
        # 将正则表达式从 'NTL_' 修改为 'NO2_'
        match = re.match(r'NO2_(\d{4})_(\d{2})', col_name)
        if match:
            year, week_str = match.groups()
            if week_str == '00' or int(week_str) > 53: continue
            try:
                # 使用标准的ISO周日期格式进行转换
                time_stamp = pd.to_datetime(f'{year}-{week_str}-1', format='%G-%V-%u')
            except ValueError:
                continue
            # 同样处理无效值
            if pd.notna(value) and value > 0:
                time_series_data[time_stamp] = float(value)
            else:
                time_series_data[time_stamp] = np.nan

    if not time_series_data:
        print(f"  -> 跳过 (无NO2数据): {row_data.get('location_name', 'N/A')}")
        return None

    s = pd.Series(time_series_data).sort_index()
    # 插值处理缺失值
    s_interpolated = s.interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(
        method='ffill')

    # 检查是否有足够的数据进行季节性分解（至少需要2个周期，即104周）
    if s_interpolated.isnull().all() or len(s_interpolated) < 104:
        print(f"  -> 跳过 (有效周数据不足104个): {row_data.get('location_name', 'N/A')}")
        return None

    # 使用指数加权移动平均进行平滑，以减少短期噪声
    s_smoothed = s_interpolated.ewm(span=8, adjust=False).mean()

    # --- 2. 时间序列分解 ---
    # 使用STL分解方法，周期为52周（一年）
    decomposition = seasonal_decompose(s_smoothed, model='additive', period=52)
    trend_component = decomposition.trend.interpolate(method='linear', limit_direction='both').fillna(
        method='bfill').fillna(method='ffill')

    earthquake_date = pd.Timestamp(row_data['date'])

    # 计算震前基准线
    pre_earthquake_trend = trend_component[trend_component.index < earthquake_date]
    baseline_value = None
    if not pre_earthquake_trend.empty:
        baseline_value = pre_earthquake_trend.iloc[-1]

    # --- 3. 绘图部分 ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # 绘制原始(平滑后)的NO2数据作为淡灰色背景
    ax.plot(s_smoothed.index, s_smoothed.values, color='lightgray', linestyle='-', linewidth=1.5,
            label='原始周度NO2数据 (含季节性波动)')

    # 绘制分解出的趋势线作为黑色主角
    ax.plot(trend_component.index, trend_component.values, color='black', marker='.', markersize=5, linestyle='-',
            label='城市活动趋势 (已移除季节性)')

    # 绘制地震线和基准线
    ax.axvline(x=earthquake_date, color='red', linestyle='--', linewidth=2.5,
               label=f'地震发生 ({earthquake_date.date()})')

    if baseline_value is not None:
        ax.axhline(y=baseline_value, color='green', linestyle=':', linewidth=2, label='震前活动水平基准线')

    # 聚焦于地震前后各一年的时间窗口
    zoom_window_weeks = 52
    start_date = earthquake_date - pd.DateOffset(weeks=zoom_window_weeks)
    end_date = earthquake_date + pd.DateOffset(weeks=zoom_window_weeks)
    ax.set_xlim(start_date, end_date)

    # 自动调整Y轴范围以获得更好的视觉效果
    zoomed_data = trend_component[start_date:end_date]
    if not zoomed_data.empty:
        y_min = zoomed_data.min() * 0.95
        y_max = zoomed_data.max() * 1.05
        ax.set_ylim(y_min, y_max)

    # 更新标题和标签 (核心修改点)
    ax.set_title(f'地震冲击响应分析 (基于NO2浓度): {row_data.get("location_name")}', fontsize=16)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('周度NO2浓度趋势值 (mol/m^2)', fontsize=12)
    ax.legend(fontsize=10)
    fig.autofmt_xdate()
    plt.tight_layout()

    from io import BytesIO
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png', dpi=120)
    plt.close(fig)
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    return encoded


def create_resilience_map(directory_path, output_filename='urban_earthquake_resilience_NO2_uncalibrated.html'):
    """主函数，驱动整个分析和地图生成流程。"""
    set_chinese_font()
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

    if not csv_files:
        print(f"🔴 错误：在文件夹 '{directory_path}' 中没有找到任何 .csv 文件。")
        return

    print(f"找到 {len(csv_files)} 个CSV文件，正在合并处理...")
    all_data_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    all_data_df = all_data_df.drop_duplicates(subset=['location_name', 'date'])
    print(f"数据合并完成，共计 {len(all_data_df)} 条地震记录。开始逐个分析并生成地图...")

    m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB positron')

    processed_count = 0
    for index, row in all_data_df.iterrows():
        b64_image = create_resilience_plot_base64(row)

        if not b64_image: continue

        processed_count += 1
        location_name = row.get('location_name', 'N/A')
        magnitude = row.get('magnitude', 0)
        date = row.get('date', 'N/A')

        # 兼容两种可能的地理坐标格式
        lon, lat = None, None
        if '.geo' in row and pd.notna(row['.geo']):
            try:
                geo_data = json.loads(row['.geo'])
                lon, lat = geo_data['coordinates']
            except Exception:
                pass

        if lon is None and all(k in row for k in ['longitude', 'latitude']):
            lon, lat = row['longitude'], row['latitude']

        if lon is None:
            print(f"  -> 严重警告：无法解析第 {index} 行 ({location_name}) 的地理坐标。")
            continue

        html = f"""<h4>{location_name}</h4><b>震级:</b> {magnitude} | <b>日期:</b> {date}<br><hr><img src="data:image/png;base64,{b64_image}">"""
        iframe = folium.IFrame(html, width=1250, height=650)
        popup = folium.Popup(iframe, max_width=1250)

        # 修改标记点颜色以作区分
        folium.CircleMarker(
            location=[lat, lon], radius=float(magnitude) * 1.5,
            popup=popup, tooltip=f"{location_name} (M{magnitude})",
            color='darkblue', fill=True, fill_color='blue', fill_opacity=0.6
        ).add_to(m)

    m.save(output_filename)

    print("\n-----------------------------------------------------")
    print(f"🎉 地图已成功生成并保存为 '{output_filename}'。")
    print(f"本次共对 {processed_count} 次有效地震事件进行了韧性分析和绘图。")
    print("-----------------------------------------------------")


if __name__ == "__main__":
    # --- 【【【 用户需要修改的路径 】】】 ---
    # 指向您存放从GEE下载的NO2 CSV文件的文件夹
    csv_directory_path = r"drive-download-20251110T111232Z-1-001"

    # --- 程序开始 ---
    if not os.path.isdir(csv_directory_path):
        print(f"🔴 错误：指定的路径 '{csv_directory_path}' 不存在或不是一个文件夹。")
    else:
        create_resilience_map(csv_directory_path)