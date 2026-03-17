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

matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')


def set_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        print("✅ 绘图字体已设置为 'SimHei' (黑体)。")
    except:
        print("⚠️ 警告：未能找到 'SimHei' 字体。")
    plt.rcParams['axes.unicode_minus'] = False


def create_resilience_plot_base64(row_data):
    """
    【最终版 V2】:
    - 增加淡灰色的原始数据线作为背景参考。
    - 在分解出的“趋势”数据上进行分析。
    - 聚焦于地震前后一年的时间窗口。
    - 绘制基于“趋势”的震前基准线。
    """
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
        print(f"  -> 跳过: {row_data.get('location_name', 'N/A')} (有效周数据不足104个)")
        return None

    s_smoothed = s_interpolated.ewm(alpha=0.7, adjust=False).mean()

    decomposition = seasonal_decompose(s_smoothed, model='additive', period=52)
    trend_component = decomposition.trend.interpolate(method='linear', limit_direction='both').fillna(
        method='bfill').fillna(method='ffill')

    earthquake_date = pd.Timestamp(row_data['date'])

    pre_earthquake_trend = trend_component[trend_component.index < earthquake_date]
    baseline_value = None
    if not pre_earthquake_trend.empty:
        baseline_value = pre_earthquake_trend.iloc[-1]

    # --- 绘图部分 ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # ==================================================================
    #  【核心视觉修改】
    # ==================================================================
    # 1. 新增：绘制原始数据线 (设为淡灰色背景)
    ax.plot(s_smoothed.index, s_smoothed.values, color='lightgray', linestyle='-', linewidth=1.5,
            label='原始周度数据 (波动)')

    # 2. 绘制分解趋势线 (保持为黑色主角)
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

    ax.set_title(f'地震冲击响应分析: {row_data.get("location_name")}', fontsize=16)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('周度夜光趋势值', fontsize=12)
    ax.legend(fontsize=10)
    fig.autofmt_xdate()
    plt.tight_layout()

    from io import BytesIO
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png', dpi=120)
    plt.close(fig)
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    return encoded


def create_resilience_map(directory_path, output_filename='urban_earthquake_resilience_final.html'):
    set_chinese_font()
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    # ... (此函数的其余部分无需修改) ...
    if not csv_files:
        print(f"🔴 错误：在文件夹 '{directory_path}' 中没有找到任何 .csv 文件。")
        return
    print(f"找到 {len(csv_files)} 个CSV文件，正在合并处理...")
    all_data_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True).drop_duplicates(
        subset=['location_name', 'date'])
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
        try:
            geo_data = json.loads(row['.geo'])
            lon, lat = geo_data['coordinates']
        except Exception as e:
            print(f"  -> 严重警告：无法解析第 {index} 行的地理坐标。错误: {e}")
            continue
        html = f"""<h4>{location_name}</h4><b>震级:</b> {magnitude} | <b>日期:</b> {date}<br><hr><img src="data:image/png;base64,{b64_image}">"""
        iframe = folium.IFrame(html, width=1250, height=650)
        popup = folium.Popup(iframe, max_width=1250)
        folium.CircleMarker(
            location=[lat, lon], radius=float(magnitude) * 1.5,
            popup=popup, tooltip=f"{location_name} (M{magnitude})",
            color='darkred', fill=True, fill_color='red', fill_opacity=0.6
        ).add_to(m)
    m.save(output_filename)
    print("\n-----------------------------------------------------")
    print(f"🎉 地图已成功生成并保存为 '{output_filename}'。")
    print(f"本次共对 {processed_count} 次有效地震事件进行了韧性分析和绘图。")
    print("-----------------------------------------------------")


if __name__ == "__main__":
    csv_directory_path = r"D:\虚拟c盘\GEE_Uploads\EURm\M65"
    if not os.path.isdir(csv_directory_path):
        print(f"🔴 错误：指定的路径 '{csv_directory_path}' 不存在。")
    else:
        create_resilience_map(csv_directory_path)