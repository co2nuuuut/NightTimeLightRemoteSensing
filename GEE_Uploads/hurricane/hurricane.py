# =============================================================================
# city_centric_hurricane_analyzer.py
#
# 一个以城市/地区为中心的本地分析程序。它通过聚类将地理位置相近的飓风
# 事件归为一组，然后为每个地区生成一张包含多次冲击事件的综合时间序列图，
# 并将结果展示在一个以城市为标记的交互式地图上。
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
from sklearn.cluster import DBSCAN
from math import radians, sin, cos, sqrt, atan2

# --- 配置绘图后端和样式 ---
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# 1. 配置区域 - 【请务必修改这里的路径!】
# =============================================================================

# CSV数据所在的文件夹
DATA_DIRECTORY = r"D:\虚拟c盘\GEE_Uploads\hurricane\drive-download-20251024T064751Z-1-001"

# DBSCAN聚类参数
# eps: 两个样本被视为在邻域内的最大距离（单位：度）。5度约等于555公里，适合区域性风暴聚类。
# min_samples: 一个核心点邻域内所需的最小样本数。设置为2表示至少2次风暴才能构成一个受灾区。
CLUSTER_EPSILON = 5.0
CLUSTER_MIN_SAMPLES = 2

# 输出的HTML地图文件名
OUTPUT_HTML_FILE = "City_Centric_Hurricane_Impact_Map.html"


def set_chinese_font():
    """尝试设置中文字体以正确显示图表标题。"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        print("✅ 绘图字体已设置为 'SimHei' (黑体)。")
    except:
        print("⚠️ 警告：未能找到 'SimHei' 字体。")
    plt.rcParams['axes.unicode_minus'] = False


def cluster_events_by_location(df):
    """使用DBSCAN算法根据飓风起始点对事件进行地理空间聚类。"""
    print("\n--- 开始对飓风事件进行地理空间聚类以识别受灾地区 ---")

    coords = []
    indices = []
    for index, row in df.iterrows():
        try:
            geo_data = json.loads(row['.geo'])
            start_lon, start_lat = geo_data['coordinates'][0]
            coords.append([start_lat, start_lon])
            indices.append(index)
        except (json.JSONDecodeError, IndexError, KeyError):
            continue

    if not coords:
        print("🔴 错误：无法从任何记录中解析地理坐标。")
        return None

    # DBSCAN需要弧度单位来进行haversine距离计算
    X = np.radians(np.array(coords))

    # 使用haversine度量进行聚类
    db = DBSCAN(eps=np.radians(CLUSTER_EPSILON), min_samples=CLUSTER_MIN_SAMPLES, metric='haversine').fit(X)

    # 将聚类结果添加回DataFrame
    cluster_labels = pd.Series(db.labels_, index=indices, name='cluster_id')
    df_clustered = df.join(cluster_labels)

    # 过滤掉未被聚类的事件 (标签为-1的是噪声点)
    df_clustered = df_clustered[df_clustered['cluster_id'] != -1]

    num_clusters = len(df_clustered['cluster_id'].unique())
    print(f"✅ 聚类完成！识别出 {num_clusters} 个频繁受灾地区（簇）。")

    return df_clustered


def create_consolidated_plot_base64(representative_row, all_events_in_cluster):
    """
    为整个地区（簇）创建一个包含多次冲击事件的综合图表。
    使用代表性事件（最强的飓风）的时间序列作为背景。
    """
    # 1. 从代表性事件中提取和处理时间序列数据
    time_series_data = {}
    for col_name, value in representative_row.items():
        match = re.match(r'NTL_(\d{4})_(\d{2})', col_name)
        if match:
            year, week_str = match.groups()
            if week_str == '00' or int(week_str) > 53: continue
            try:
                time_stamp = pd.to_datetime(f'{year}-{week_str}-1', format='%G-%V-%u')
            except ValueError:
                continue
            time_series_data[time_stamp] = float(value) if pd.notna(value) and value > 0 else np.nan

    if not time_series_data: return None

    s = pd.Series(time_series_data).sort_index()
    s_interpolated = s.interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(
        method='ffill')
    if s_interpolated.isnull().all() or len(s_interpolated) < 104: return None
    s_smoothed = s_interpolated.ewm(alpha=0.7, adjust=False).mean()
    decomposition = seasonal_decompose(s_smoothed, model='additive', period=52)
    trend_component = decomposition.trend.interpolate(method='linear', limit_direction='both').fillna(
        method='bfill').fillna(method='ffill')

    # 2. 绘图
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(s_smoothed.index, s_smoothed.values, color='lightgray', linestyle='-', linewidth=1.5,
            label='原始周度数据 (平滑后)')
    ax.plot(trend_component.index, trend_component.values, color='black', marker='.', markersize=5, linestyle='-',
            label='地区发展趋势 (基于最强冲击)')

    # 3. 在一张图上标记所有相关事件
    # 先对事件按日期排序
    all_events_in_cluster = sorted(all_events_in_cluster, key=lambda x: pd.Timestamp(x['start_date']))

    for event in all_events_in_cluster:
        event_date = pd.Timestamp(event['start_date'])
        is_representative = (event['sid'] == representative_row['sid'])

        ax.axvline(
            x=event_date,
            color='red' if is_representative else 'royalblue',
            linestyle='--' if is_representative else ':',
            linewidth=2.5 if is_representative else 1.5,
            label=f"{event['name']} ({event_date.date()}) - {'代表性事件' if is_representative else '其他事件'}"
        )

    # 4. 设置标题和格式
    # 确定一个聚焦的时间窗口，以最早和最晚的事件为中心
    first_event_date = pd.Timestamp(all_events_in_cluster[0]['start_date'])
    last_event_date = pd.Timestamp(all_events_in_cluster[-1]['start_date'])
    ax.set_xlim(first_event_date - pd.DateOffset(years=2), last_event_date + pd.DateOffset(years=2))

    cluster_name = f"地区 (中心约: {representative_row['cluster_center'][0]:.1f}°N, {representative_row['cluster_center'][1]:.1f}°E)"
    ax.set_title(f"多重飓风冲击对同一地区的综合影响分析: {cluster_name}", fontsize=16)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('周度夜光趋势值', fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    fig.autofmt_xdate()
    plt.tight_layout()

    # 5. 转换为Base64
    from io import BytesIO
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png', dpi=120)
    plt.close(fig)
    return base64.b64encode(tmpfile.getvalue()).decode('utf-8')


def main():
    """主程序：加载数据 -> 聚类 -> 逐个地区分析 -> 生成地图"""
    set_chinese_font()

    # 1. 加载并合并所有CSV数据
    csv_files = glob.glob(os.path.join(DATA_DIRECTORY, 'urban_hurricane_*.csv'))
    if not csv_files:
        print(f"🔴 错误：在 '{DATA_DIRECTORY}' 中没有找到任何 'urban_hurricane_*.csv' 文件。")
        return
    print(f"找到 {len(csv_files)} 个CSV文件，正在合并...")
    master_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True).drop_duplicates(subset=['sid'])

    # 2. 对事件进行地理聚类
    df_clustered = cluster_events_by_location(master_df)
    if df_clustered is None or df_clustered.empty:
        print("🔴 聚类后无有效数据，程序终止。")
        return

    # 3. 初始化地图
    map_center = [df_clustered['lat'].mean(),
                  df_clustered['lon'].mean()] if 'lat' in df_clustered.columns and 'lon' in df_clustered.columns else [
        20, 120]
    m = folium.Map(location=map_center, zoom_start=3, tiles='CartoDB positron')

    processed_clusters = 0
    # 4. 按簇（地区）进行分组处理
    for cluster_id, cluster_df in df_clustered.groupby('cluster_id'):
        print(f"\n--- 正在分析地区簇 {cluster_id} (包含 {len(cluster_df)} 次飓风事件) ---")

        # a. 找到该簇的代表性事件（最强的那个）
        representative_event = cluster_df.loc[cluster_df['max_category'].idxmax()]

        # b. 计算该簇的地理中心
        cluster_coords = []
        for index, row in cluster_df.iterrows():
            try:
                geo_data = json.loads(row['.geo'])
                cluster_coords.append(geo_data['coordinates'][0])
            except:
                continue
        if not cluster_coords: continue
        center_lon, center_lat = np.mean(np.array(cluster_coords), axis=0)

        # 将中心点存入代表性事件中，方便绘图函数使用
        representative_event['cluster_center'] = (center_lat, center_lon)

        # c. 生成综合分析图
        all_events_list = cluster_df.to_dict('records')
        b64_image = create_consolidated_plot_base64(representative_event, all_events_list)
        if not b64_image:
            print("  -> 生成图像失败，跳过此地区。")
            continue

        processed_clusters += 1

        # d. 准备弹出窗口的HTML
        hurricane_names = ", ".join(cluster_df['name'].unique())
        html = f"""
        <h4>地区综合分析 (中心: {center_lat:.2f}, {center_lon:.2f})</h4>
        <b>影响该地区的强飓风包括:</b> {hurricane_names}<br><hr>
        <img src="data:image/png;base64,{b64_image}">
        """
        iframe = folium.IFrame(html, width=1250, height=700)
        popup = folium.Popup(iframe, max_width=1250)

        # e. 在地图上添加地区标记
        folium.Marker(
            location=[center_lat, center_lon],
            popup=popup,
            tooltip=f"地区: {len(cluster_df)} 次强风暴<br>点击查看综合分析",
            icon=folium.Icon(color='darkred', icon='cloud')
        ).add_to(m)

    # 5. 保存最终的地图文件
    m.save(OUTPUT_HTML_FILE)
    print("\n-----------------------------------------------------")
    print(f"🎉 地图已成功生成并保存为 '{OUTPUT_HTML_FILE}'。")
    print(f"本次共分析了 {processed_clusters} 个频繁受灾的地区。")
    print("-----------------------------------------------------")


if __name__ == "__main__":
    main()