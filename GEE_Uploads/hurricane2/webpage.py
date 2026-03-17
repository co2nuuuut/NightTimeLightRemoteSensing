# =============================================================================
# visualize_hurricane_impact.py
#
# 【最终修正版】 - 解决了 SettingWithCopyWarning 警告。
#
# 使用方法:
# 1. 将此脚本与您所有的 city_hurricane_NTL_weekly_analysis_batch_... .csv 文件
#    放在同一个文件夹中。
# 2. 修改脚本末尾的 `csv_directory_path` 变量，使其指向您存放CSV文件的文件夹路径。
# 3. 运行此脚本 (python visualize_hurricane_impact.py)。
# 4. 将在同一文件夹下生成一个名为 'hurricane_impact_resilience_map.html' 的文件。
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

# --- 设置Matplotlib后端和字体 ---
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')


def set_chinese_font():
    """尝试设置中文字体以在图表中正确显示中文。"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        print("✅ 绘图字体已设置为 'SimHei' (黑体)。")
    except Exception as e:
        print(f"⚠️ 警告：未能找到 'SimHei' 字体，图表中的中文可能显示为方框。错误: {e}")
    plt.rcParams['axes.unicode_minus'] = False


def create_hurricane_plot_base64(row_data):
    """
    为单行数据（一个飓风-城市事件）创建韧性分析图表。
    【飓风定制版 V2】 - 修正 SettingWithCopyWarning
    """

    # 1. --- 数据解析与准备 ---
    time_series_df = pd.DataFrame(columns=['NTL', 'H_dist', 'H_cat'])

    for col_name, value in row_data.items():
        match = re.match(r'D_(\d{4})_(\d{2})_(NTL|H_dist|H_cat)', col_name)
        if match:
            year, week_str, data_type = match.groups()
            if week_str == '00' or int(week_str) > 53: continue
            try:
                time_stamp = pd.to_datetime(f'{year}-{week_str}-1', format='%G-%V-%u')
            except ValueError:
                continue
            time_series_df.loc[time_stamp, data_type] = float(value)

    if time_series_df.empty:
        return None

    # 2. --- 夜光数据(NTL)预处理 ---
    # =======================【核心修正点 1】=======================
    # 使用 .loc 直接在原始DataFrame上操作，避免SettingWithCopyWarning
    time_series_df.loc[time_series_df['NTL'] <= 0, 'NTL'] = np.nan
    ntl_series = time_series_df['NTL']
    # =========================================================

    ntl_interpolated = ntl_series.interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(
        method='ffill')

    if ntl_interpolated.isnull().all() or len(ntl_interpolated.dropna()) < 104:
        print(f"  -> 跳过: {row_data.get('city_name', 'N/A')} (有效周数据不足104个)")
        return None

    ntl_smoothed = ntl_interpolated.ewm(alpha=0.7, adjust=False).mean()
    decomposition = seasonal_decompose(ntl_smoothed, model='additive', period=52)
    trend_component = decomposition.trend.interpolate(method='linear', limit_direction='both').fillna(
        method='bfill').fillna(method='ffill')

    # 3. --- 飓风数据(H_dist, H_cat)预处理 ---
    # =======================【核心修正点 2】=======================
    # 在切片时使用 .copy()，明确表示我们正在创建一个新的DataFrame副本进行操作
    hurricane_data = time_series_df[['H_dist', 'H_cat']].copy()
    hurricane_data[hurricane_data < 0] = np.nan
    # =========================================================

    # 4. --- 确定关键日期和基准线 ---
    hurricane_date = pd.Timestamp(row_data['h_impact_time'])
    pre_hurricane_trend = trend_component[trend_component.index < hurricane_date]
    baseline_value = pre_hurricane_trend.iloc[-1] if not pre_hurricane_trend.empty else None

    # 5. --- 绘图 (此部分逻辑不变) ---
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 7))

    ax1.plot(ntl_smoothed.index, ntl_smoothed.values, color='lightgray', linestyle='-', linewidth=1.5,
             label='原始周度夜光 (平滑后)')
    ax1.plot(trend_component.index, trend_component.values, color='black', marker='.', markersize=4, linestyle='-',
             label='城市发展趋势')

    ax1.axvline(x=hurricane_date, color='red', linestyle='--', linewidth=2.5,
                label=f'飓风最接近 ({hurricane_date.date()})')
    if baseline_value is not None:
        ax1.axhline(y=baseline_value, color='green', linestyle=':', linewidth=2, label='风前发展水平基准')

    zoom_window_weeks = 52
    start_date = hurricane_date - pd.DateOffset(weeks=zoom_window_weeks)
    end_date = hurricane_date + pd.DateOffset(weeks=zoom_window_weeks)
    ax1.set_xlim(start_date, end_date)

    zoomed_data = trend_component[start_date:end_date]
    if not zoomed_data.empty:
        y_min = zoomed_data.min() * 0.9
        y_max = zoomed_data.max() * 1.1
        ax1.set_ylim(y_min, y_max)

    ax1.set_title(f'飓风 "{row_data.get("h_name")}" 对 {row_data.get("city_name")} 的冲击响应分析', fontsize=16)
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('周度夜光趋势值 (Trend)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()

    valid_hurricane_data = hurricane_data.dropna()
    if not valid_hurricane_data.empty:
        scatter = ax2.scatter(
            valid_hurricane_data.index,
            valid_hurricane_data['H_dist'],
            c=valid_hurricane_data['H_cat'],
            s=valid_hurricane_data['H_cat'] * 40 + 20,
            cmap='viridis_r',
            alpha=0.7,
            label='飓风实时状态'
        )
        cbar = plt.colorbar(scatter, ax=ax2, pad=0.08)
        cbar.set_label('飓风实时等级 (S-S Scale)', rotation=270, labelpad=20)

    ax2.set_ylabel('飓风中心距离 (km)', fontsize=12, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.invert_yaxis()

    lines, labels = ax1.get_legend_handles_labels()
    if not valid_hurricane_data.empty:
        proxy_scatter = matplotlib.lines.Line2D([0], [0], linestyle="none", c='blue', marker='o', markersize=10,
                                                label='飓风实时状态 (见颜色条)')
        ax1.legend(handles=lines + [proxy_scatter], fontsize=10, loc='best')
    else:
        ax1.legend(handles=lines, fontsize=10, loc='best')

    fig.autofmt_xdate()
    plt.tight_layout(rect=[0, 0, 0.95, 1])

    # 6. --- 将图像编码为Base64字符串 ---
    from io import BytesIO
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png', dpi=120)
    plt.close(fig)
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    return encoded


def create_hurricane_map(directory_path, output_filename='hurricane_impact_resilience_map.html'):
    """
    主函数：合并所有CSV数据，为每个事件生成图表，并创建一个交互式Folium地图。
    """
    set_chinese_font()

    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    if not csv_files:
        print(f"🔴 错误：在文件夹 '{directory_path}' 中没有找到任何 .csv 文件。")
        return

    print(f"✅ 找到 {len(csv_files)} 个CSV文件，正在合并处理...")

    all_data_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    all_data_df.drop_duplicates(subset=['event_id'], inplace=True)

    print(f"✅ 数据合并与去重完成，共计 {len(all_data_df)} 条唯一的“飓风-城市”事件。")
    print("--- 开始逐个分析并生成地图标记点 (这可能需要一些时间)... ---")

    m = folium.Map(location=[25, 120], zoom_start=4, tiles='CartoDB positron')

    processed_count = 0
    for index, row in all_data_df.iterrows():
        b64_image = create_hurricane_plot_base64(row)

        if not b64_image:
            continue

        processed_count += 1
        if processed_count % 10 == 0:
            print(f"  -> 已处理 {processed_count} / {len(all_data_df)} 个事件...")

        city_name = row.get('city_name', 'N/A')
        h_name = row.get('h_name', 'NOT_NAMED')
        h_max_cat = row.get('h_max_cat', 0)
        h_impact_time = row.get('h_impact_time', 'N/A')
        h_min_dist_km = row.get('h_min_dist_km', 'N/A')

        try:
            geo_data = json.loads(row['.geo'])
            lon, lat = geo_data['coordinates']
        except (KeyError, json.JSONDecodeError, TypeError):
            try:
                lat = row['city_lat']
                lon = row['city_lon']
            except KeyError:
                print(f"  -> 严重警告：无法解析第 {index} 行的地理坐标，跳过此点。")
                continue

        html = f"""
        <h4>{city_name} vs. 飓风 {h_name}</h4>
        <b>最大强度:</b> {h_max_cat} 级<br>
        <b>最接近日期:</b> {h_impact_time}<br>
        <b>最近距离:</b> {h_min_dist_km} km
        <hr>
        <img src="data:image/png;base64,{b64_image}">
        """

        iframe = folium.IFrame(html, width=1300, height=750)
        popup = folium.Popup(iframe, max_width=1300)

        folium.CircleMarker(
            location=[lat, lon],
            radius=float(h_max_cat) * 1.8,
            popup=popup,
            tooltip=f"{city_name} 受到 飓风 {h_name} 影响",
            color='darkblue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        ).add_to(m)

    m.save(output_filename)

    print("\n-----------------------------------------------------")
    print(f"🎉 地图已成功生成并保存为 '{output_filename}'。")
    print(f"   本次共对 {processed_count} / {len(all_data_df)} 个有效事件进行了韧性分析和绘图。")
    print("-----------------------------------------------------")


if __name__ == "__main__":
    # =======================【请修改这里】=======================
    # 请将此路径修改为您存放所有 'city_hurricane_NTL_weekly_analysis_...' CSV文件的文件夹路径
    csv_directory_path = r"data2"
    # =========================================================

    if not os.path.isdir(csv_directory_path):
        print(f"🔴 错误：指定的路径 '{csv_directory_path}' 不是一个有效的文件夹。")
        print("   请确保路径正确，并且不要包含文件名。")
    else:
        create_hurricane_map(csv_directory_path)