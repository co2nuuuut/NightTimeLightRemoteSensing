# =============================================================================
#
#  地震韧性分析脚本 (V2.2 - 适配LST数据最终版)
#
#  功能:
#  1. 读取本地文件夹中，由GEE导出的MODIS LST周度数据CSV文件。
#  2. (无校正) 使用时间序列分解方法分析城市热环境趋势。
#  3. 为每次地震生成基于LST趋势的韧性分析图。
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
# 使用 'Agg' 后端，这样脚本可以在没有图形界面的服务器上运行而不会报错
matplotlib.use('Agg')
# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')


def set_chinese_font():
    """
    尝试设置支持中文的字体。
    在Windows上通常是 'SimHei'。在其他操作系统上可能需要更改。
    """
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        print("✅ 绘图字体已设置为 'SimHei' (黑体)。")
    except Exception as e:
        print(f"⚠️ 警告：未能找到 'SimHei' 字体。中文可能无法正常显示。错误: {e}")
    # 解决保存图像时负号'-'显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False


def create_resilience_plot_base64(row_data):
    """
    为单次地震事件创建韧性分析图，并返回图像的Base64编码字符串。

    【适配LST数据】:
    - 读取以 "LST_" 开头的列。
    - 使用与原版相同的季节性分解方法进行趋势分析。
    - 更新图表的标题和标签以反映地表温度数据。
    """
    time_series_data = {}

    # --- 1. 从行数据中提取时间序列 ---
    for col_name, value in row_data.items():
        # 正则表达式匹配 'LST_YYYY_WW' 格式的列名
        match = re.match(r'LST_(\d{4})_(\d{2})', col_name)
        if match:
            year, week_str = match.groups()
            # 忽略无效的周数
            if week_str == '00' or int(week_str) > 53: continue
            try:
                # 将年和周数转换为日期格式 (每周的第一天，即周一)
                time_stamp = pd.to_datetime(f'{year}-{week_str}-1', format='%G-%V-%u')
            except ValueError:
                continue
            # GEE导出的无效值通常是-999
            if pd.notna(value) and value > -999:
                time_series_data[time_stamp] = float(value)
            else:
                time_series_data[time_stamp] = np.nan

    if not time_series_data:
        print(f"  -> 跳过 (无有效的LST数据): {row_data.get('location_name', 'N/A')}")
        return None

    s = pd.Series(time_series_data).sort_index()
    # 使用时间插值来填充缺失值，使时间序列连续
    s_interpolated = s.interpolate(method='time', limit_direction='both').fillna(method='bfill').fillna(method='ffill')

    # 至少需要两年数据（104周）才能进行有效的季节性分解
    if s_interpolated.isnull().all() or len(s_interpolated) < 104:
        print(f"  -> 跳过 (有效周数据不足104个): {row_data.get('location_name', 'N/A')}")
        return None

    # 对数据进行平滑处理，以减少短期噪声的干扰
    s_smoothed = s_interpolated.ewm(span=8, adjust=False).mean()

    # --- 2. 时间序列分析：季节性分解 ---
    # 将时间序列分解为趋势、季节性和残差三个部分
    decomposition = seasonal_decompose(s_smoothed, model='additive', period=52)  # 周期设为52周
    trend_component = decomposition.trend.interpolate(method='linear', limit_direction='both').fillna(
        method='bfill').fillna(method='ffill')

    earthquake_date = pd.Timestamp(row_data['date'])

    # 将地震发生前的最后一个趋势点作为恢复的基准线
    pre_earthquake_trend = trend_component[trend_component.index < earthquake_date]
    baseline_value = None
    if not pre_earthquake_trend.empty:
        baseline_value = pre_earthquake_trend.iloc[-1]

    # --- 3. 开始绘图 ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # 绘制平滑后的原始数据线作为背景参考
    ax.plot(s_smoothed.index, s_smoothed.values, color='lightgray', linestyle='-', linewidth=1.5,
            label='原始周度地表温度 (含季节性波动)')

    # 绘制分解出的趋势线，这是我们分析的核心
    ax.plot(trend_component.index, trend_component.values, color='black', marker='.', markersize=5, linestyle='-',
            label='城市热环境趋势 (已移除季节性)')

    # 标记地震发生时间
    ax.axvline(x=earthquake_date, color='red', linestyle='--', linewidth=2.5,
               label=f'地震发生 ({earthquake_date.date()})')

    # 绘制震前基准线
    if baseline_value is not None:
        ax.axhline(y=baseline_value, color='green', linestyle=':', linewidth=2, label='震前热环境水平基准线')

    # 将图表缩放到地震前后各一年的时间窗口
    zoom_window_weeks = 52
    start_date = earthquake_date - pd.DateOffset(weeks=zoom_window_weeks)
    end_date = earthquake_date + pd.DateOffset(weeks=zoom_window_weeks)
    ax.set_xlim(start_date, end_date)

    # 自动调整Y轴的显示范围，使其更美观
    zoomed_data = trend_component[start_date:end_date]
    if not zoomed_data.empty:
        y_min = zoomed_data.min() - 2  # 上下留出2摄氏度的缓冲
        y_max = zoomed_data.max() + 2
        ax.set_ylim(y_min, y_max)

    # 设置图表的标题和坐标轴标签
    ax.set_title(f'地震冲击响应分析 (基于地表温度): {row_data.get("location_name")}', fontsize=16)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('周度地表温度趋势值 (°C)', fontsize=12)
    ax.legend(fontsize=10)
    fig.autofmt_xdate()  # 自动格式化X轴的日期标签，避免重叠
    plt.tight_layout()

    # --- 4. 将图像转换为Base64编码 ---
    from io import BytesIO
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png', dpi=120)
    plt.close(fig)  # 关闭图像，释放内存
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    return encoded


def create_resilience_map(directory_path, output_filename='urban_earthquake_resilience_LST.html'):
    """
    主函数，循环处理所有CSV文件并生成最终的交互式地图。
    """
    set_chinese_font()

    # 查找指定目录下的所有CSV文件
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

    if not csv_files:
        print(f"🔴 错误：在文件夹 '{directory_path}' 中没有找到任何 .csv 文件。")
        return

    print(f"找到 {len(csv_files)} 个CSV文件，正在合并处理...")

    # 合并所有CSV文件为一个DataFrame，并按地震位置和日期去除重复项
    all_data_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    all_data_df = all_data_df.drop_duplicates(subset=['location_name', 'date'])

    print(f"数据合并完成，共计 {len(all_data_df)} 条地震记录。开始逐个分析并生成地图...")

    # 初始化Folium地图
    m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB positron')

    processed_count = 0
    # 遍历DataFrame的每一行
    for index, row in all_data_df.iterrows():
        # 为当前行（地震事件）生成分析图
        b64_image = create_resilience_plot_base64(row)

        # 如果绘图失败（如数据不足），则跳过
        if not b64_image: continue

        processed_count += 1
        location_name = row.get('location_name', 'N/A')
        magnitude = row.get('magnitude', 0)
        date = row.get('date', 'N/A')

        # 解析地理坐标
        lon, lat = None, None
        # 优先使用GEE导出的'.geo'字段
        if '.geo' in row and pd.notna(row['.geo']):
            try:
                geo_data = json.loads(row['.geo'])
                lon, lat = geo_data['coordinates']
            except Exception as e:
                print(f"  -> 警告：无法解析第 {index} 行的 .geo 字段。错误: {e}")
                pass

        # 如果'.geo'字段无效，则尝试使用'longitude'和'latitude'列
        if lon is None and all(k in row for k in ['longitude', 'latitude']):
            lon, lat = row['longitude'], row['latitude']

        if lon is None:
            print(f"  -> 严重警告：无法解析第 {index} 行 ({location_name}) 的地理坐标，跳过此条目。")
            continue

        # 创建嵌入在地图弹窗中的HTML内容
        html = f"""<h4>{location_name}</h4><b>震级:</b> {magnitude} | <b>日期:</b> {date}<br><hr><img src="data:image/png;base64,{b64_image}">"""
        iframe = folium.IFrame(html, width=1250, height=650)
        popup = folium.Popup(iframe, max_width=1250)

        # 在地图上添加一个圆形标记
        folium.CircleMarker(
            location=[lat, lon],
            radius=float(magnitude) * 1.5,  # 圆点大小与震级相关
            popup=popup,
            tooltip=f"{location_name} (M{magnitude})",
            color='darkpurple',
            fill=True,
            fill_color='purple',
            fill_opacity=0.6
        ).add_to(m)

    # 保存地图到HTML文件
    m.save(output_filename)

    print("\n-----------------------------------------------------")
    print(f"🎉 地图已成功生成并保存为 '{output_filename}'。")
    print(f"本次共对 {processed_count} 次有效地震事件进行了韧性分析和绘图。")
    print("-----------------------------------------------------")


if __name__ == "__main__":
    # --- 【【【 用户需要修改的路径 】】】 ---
    # 请将此路径修改为您在本地存放LST CSV文件的文件夹的绝对路径
    # 例如: r"C:\Users\YourUser\Downloads\urban_earthquake_LST_weekly"
    csv_directory_path = r"urban_earthquake_LST_weekly"

    # --- 程序开始 ---
    if not os.path.isdir(csv_directory_path):
        print(f"🔴 错误：指定的路径 '{csv_directory_path}' 不存在或不是一个文件夹。")
    else:
        create_resilience_map(csv_directory_path)