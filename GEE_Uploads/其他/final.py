import pandas as pd
import folium
import matplotlib
import matplotlib.pyplot as plt
import re
import json
import base64
import os
import glob
import numpy as np
from sklearn.linear_model import LinearRegression  # 引入线性回归模型

# 切换 matplotlib 后端
matplotlib.use('Agg')


def set_chinese_font():
    """为 matplotlib 设置中文字体。"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        print("绘图字体已设置为 'SimHei' (黑体)。")
    except Exception:
        print("警告：未能找到 'SimHei' 字体。")


def create_itsa_plot_base64(row_data):
    """
    核心分析函数：执行中断时间序列分析 (ITSA) 并绘图
    1. 提取月度时间序列数据。
    2. 对震前数据进行线性回归，建立趋势模型。
    3. 将趋势线延伸，构建“反事实”情景。
    4. 绘制实际数据与反事实情景的对比图。
    """
    time_series_data = {}
    for col_name, value in row_data.items():
        # 正则表达式匹配 'NTL_Pre_M12', 'NTL_Post_M0' 等格式
        match = re.match(r'NTL_(Pre|Post)_M(\d+)', col_name)
        if match and pd.notna(value) and value != -999:
            period, month = match.groups()
            # 将 'Pre' 转换为负数月份，'Post' 转换为正数月份
            relative_month = -int(month) if period == 'Pre' else int(month)
            time_series_data[relative_month] = float(value)

    if len(time_series_data) < 5:  # 数据点过少，无法进行有意义的分析
        return None

    s = pd.Series(time_series_data).sort_index()

    # --- 中断时间序列分析 (ITSA) ---
    pre_event_data = s[s.index < 0]

    # 1. 建立震前趋势模型 (至少需要3个震前数据点才能拟合)
    if len(pre_event_data) < 3:
        print(f"警告: {row_data.get('location_name', 'N/A')} 的震前有效数据不足3个，无法进行趋势拟合，已跳过。")
        return None

    X_pre = pre_event_data.index.values.reshape(-1, 1)
    y_pre = pre_event_data.values

    model = LinearRegression()
    model.fit(X_pre, y_pre)

    # 2. 构建“反事实”情景：将模型预测延伸到整个时间段
    X_full = s.index.values.reshape(-1, 1)
    counterfactual = model.predict(X_full)

    # --- 绘图：展示ITSA结果 ---
    fig, ax = plt.subplots(figsize=(7, 4))

    # 将实际月度数据点作为散点图绘制
    ax.plot(s.index, s.values, marker='o', linestyle='-', color='gray', alpha=0.7, label='实际月度灯光')

    # 绘制震前拟合的趋势线
    ax.plot(X_pre, model.predict(X_pre), linestyle='-', color='blue', linewidth=2, label='震前趋势')

    # 绘制延伸的“反事实”线 (代表如果没发生地震，灯光本应有的趋势)
    post_indices = s.index[s.index >= 0]
    counterfactual_post = pd.Series(counterfactual, index=s.index)[s.index >= 0]
    ax.plot(post_indices, counterfactual_post, linestyle='--', color='blue', alpha=0.8, label='“反事实”预测')

    # 标注地震发生时刻
    ax.axvline(x=-0.5, color='red', linestyle='--', linewidth=2, label='地震发生')

    # 填充实际值与反事实之间的差距，直观展示“灯光损失”
    post_event_actual = s[s.index >= 0]
    ax.fill_between(post_event_actual.index, post_event_actual.values, counterfactual_post.values,
                    where=post_event_actual.values < counterfactual_post.values,
                    color='red', alpha=0.2, label='灯光损失')

    ax.set_title('地震影响的中断时间序列分析', fontsize=12)
    ax.set_xlabel('相对于地震的月份', fontsize=10)
    ax.set_ylabel('月平均 NTL 指数', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # 将图表保存到内存并编码
    from io import BytesIO
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    plt.close(fig)
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    return encoded


def create_map_from_directory(directory_path, magnitude_thresholds, colors):
    """
    主函数：读取CSV，进行ITSA分析，并创建带有分级颜色标记和图例的最终地图。
    """
    set_chinese_font()

    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    if not csv_files:
        print(f"错误：在文件夹 '{directory_path}' 中没有找到任何 .csv 文件。")
        return

    print(f"找到 {len(csv_files)} 个CSV文件，正在合并处理...")
    all_data_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    print(f"数据合并完成，共计 {len(all_data_df)} 条地震记录。开始逐个分析并生成地图...")

    m = folium.Map(location=[20, 0], zoom_start=2)

    for index, row in all_data_df.iterrows():
        # 调用全新的ITSA分析与绘图函数
        b64_image = create_itsa_plot_base64(row)

        if not b64_image:
            continue

        try:
            geo_json = json.loads(row['.geo'])
            if geo_json['type'] == 'Point':
                coords = geo_json['coordinates']
                location = [coords[1], coords[0]]
            else:
                continue
        except (TypeError, json.JSONDecodeError, KeyError, IndexError):
            print(f"警告：无法解析第 {index} 行的地理坐标，已跳过。")
            continue

        location_name = row.get('location_name', 'N/A')
        magnitude = row.get('magnitude', 'N/A')
        date = row.get('date', 'N/A')

        try:
            mag_value = float(magnitude)
        except (ValueError, TypeError):
            mag_value = 0

        marker_color = colors[-1]  # 默认为最低级别的颜色
        for i, threshold in enumerate(magnitude_thresholds):
            if mag_value >= threshold:
                marker_color = colors[i]
                break

        html = f"""
        <h4>{location_name}</h4>
        <b>震级:</b> {magnitude}<br>
        <b>日期:</b> {date}<br><hr>
        <img src="data:image/png;base64,{b64_image}">
        """
        iframe = folium.IFrame(html, width=750, height=450)
        popup = folium.Popup(iframe)

        folium.CircleMarker(
            location=location, radius=float(mag_value) * 1.5,
            popup=popup, tooltip=f"{location_name} (M{magnitude})",
            color=marker_color, fill=True, fill_color=marker_color
        ).add_to(m)

    # ... (图例代码可以保持不变，或根据需要调整) ...
    legend_html = """..."""  # 此处省略
    # m.get_root().html.add_child(folium.Element(legend_html))

    output_filename = 'earthquake_ITSA_map.html'
    m.save(output_filename)
    print("\n处理完成！")
    print(f"地图已成功生成并保存为 '{output_filename}'。")


# --- 程序入口 ---
if __name__ == "__main__":
    # !!! 重要：请将这里的路径修改为您月度数据CSV文件所在的实际文件夹路径 !!!
    csv_directory_path = r'D:\虚拟c盘\drive-download-20251019T162621Z-1-001'  # 请替换为您的路径

    # 定义震级分级和颜色
    mag_thresholds = [8.0, 7.5]
    mag_colors = ['darkred', 'orange', 'gray']  # >=8.0, 7.5-7.9, <7.5

    create_map_from_directory(csv_directory_path, mag_thresholds, mag_colors)