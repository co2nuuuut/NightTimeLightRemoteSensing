import pandas as pd
import folium
import matplotlib
import matplotlib.pyplot as plt
import re
import json
import base64
import os
import glob  # 用于查找文件

# 切换 matplotlib 后端，避免在服务器或无 GUI 环境下出错
matplotlib.use('Agg')


def set_chinese_font():
    """为 matplotlib 设置中文字体以正确显示。"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        print("绘图字体已设置为 'SimHei' (黑体)。")
    except Exception:
        print("警告：未能找到 'SimHei' 字体。图表中的中文可能无法正常显示。")
        try:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False
            print("绘图字体已尝试设置为 'Microsoft YaHei' (微软雅黑)。")
        except Exception:
            print("警告：也未能找到 'Microsoft YaHei' 字体。")


def create_ntl_weekly_plot_base64(row_data):
    """
    根据单行地震数据生成 NTL 每周时间序列图，并返回其 Base64 编码字符串。
    (此函数无需修改，已适配周度数据)
    """
    time_series_data = {}
    for col_name, value in row_data.items():
        match = re.match(r'NTL_(Pre|Post)_W(\d+)', col_name)

        if match and pd.notna(value) and value != -999:
            period, week = match.groups()
            relative_week = -int(week) if period == 'Pre' else int(week)
            time_series_data[relative_week] = float(value)

    if not time_series_data:
        return None

    sorted_weeks = sorted(time_series_data.keys())
    sorted_ntl_values = [time_series_data[week] for week in sorted_weeks]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(sorted_weeks, sorted_ntl_values, marker='o', linestyle='-', label='周中位数夜间灯光 (NTL)')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=1.5, label='地震发生周 (Week 0)')

    ax.set_title('地震前后每周夜间灯光变化', fontsize=12)
    ax.set_xlabel('相对于地震的周数', fontsize=10)
    ax.set_ylabel('NTL 指数', fontsize=10)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    from io import BytesIO
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    plt.close(fig)

    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    return encoded


def create_map_from_directory(directory_path):
    """
    主函数：读取指定目录下所有的CSV文件，合并数据，并创建交互式地图。
    (此函数无需修改)
    """
    set_chinese_font()

    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))
    if not csv_files:
        print(f"错误：在文件夹 '{directory_path}' 中没有找到任何 .csv 文件。")
        return

    print(f"找到 {len(csv_files)} 个CSV文件，将进行合并处理...")
    all_data_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    print(f"数据合并完成，共计 {len(all_data_df)} 条地震记录。")

    m = folium.Map(location=[20, 0], zoom_start=2)

    for index, row in all_data_df.iterrows():
        b64_image = create_ntl_weekly_plot_base64(row)
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
        admin_region = row.get('Admin_Region', 'N/A')

        html = f"""
        <h4>{location_name}</h4>
        <b>行政区:</b> {admin_region}<br>
        <b>震级:</b> {magnitude}<br>
        <b>日期:</b> {date}<br><hr>
        <img src="data:image/png;base64,{b64_image}">
        """

        iframe = folium.IFrame(html, width=650, height=450)
        popup = folium.Popup(iframe)

        folium.CircleMarker(
            location=location,
            radius=float(row.get('magnitude', 5)) * 1.5,
            popup=popup,
            tooltip=f"{location_name} (M{magnitude})",
            color='crimson',
            fill=True,
            fill_color='crimson'
        ).add_to(m)

    # 更新输出文件名以反映新数据
    output_filename = 'earthquakes_weekly_stable_masked_map.html'
    m.save(output_filename)
    print("\n处理完成！")
    print(f"地图已成功生成并保存为 '{output_filename}'。")
    print("请在您的网页浏览器中打开该文件查看交互式地图和图表。")


# --- 程序入口 ---
if __name__ == "__main__":
    # !!! 唯一修改点：更新为您的新文件夹路径 !!!
    csv_directory_path = r'D:\虚拟c盘\GEE_Uploads\drive-download-20251019T072805Z-1-001'

    create_map_from_directory(csv_directory_path)