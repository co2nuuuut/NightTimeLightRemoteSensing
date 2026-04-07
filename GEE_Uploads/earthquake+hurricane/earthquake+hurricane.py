import pandas as pd
import folium
import os


def create_hazards_impact_map(eq_csv, cities_csv, output_filename='global_hazards_map.html'):
    print("开始生成灾害影响散点地图...")

    # 初始化 Folium 地图 (同款底图)
    m = folium.Map(location=[25, 110], zoom_start=4, tiles='CartoDB positron')

    eq_group = folium.FeatureGroup(name='破坏性地震中心 (M6.5+)')
    hc_group = folium.FeatureGroup(name='受飓风影响城市点')

    # ==========================================
    # 1. 绘制受飓风影响的城市 (蓝色圆点，如您的截图)
    # ==========================================
    if os.path.exists(cities_csv):
        df_cities = pd.read_csv(cities_csv)
        print(f"正在绘制 {len(df_cities)} 个受飓风影响的城市点...")

        for index, row in df_cities.iterrows():
            # 使用 CSV 中实际的列名: city, hurricane, category
            c_name = row['city']
            h_name = row['hurricane']
            cat = row['category']

            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                # 这里对应修改 tooltip 里的字段名
                tooltip=f"<b>城市:</b> {c_name}<br><b>遭受飓风:</b> {h_name} (Cat {cat})",
                color='darkblue',
                weight=1.5,
                fill=True,
                fill_color='blue',
                fill_opacity=0.5
            ).add_to(hc_group)
    else:
        print(f"⚠️ 警告: 未找到 {cities_csv}")
    # ==========================================
    # 2. 绘制地震数据 (红色圆点)
    # ==========================================
    if os.path.exists(eq_csv):
        df_eq = pd.read_csv(eq_csv)
        print(f"正在绘制 {len(df_eq)} 个地震散点...")

        for index, row in df_eq.iterrows():
            # 只有 magnitude, longitude, latitude 这三列
            mag = float(row['magnitude'])
            radius_size = (mag - 6.0) * 4

            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius_size,
                # 移除了 row['date']，防止报错
                tooltip=f"<b>地震事件:</b> M {mag}",
                color='darkred',
                weight=1.5,
                fill=True,
                fill_color='red',
                fill_opacity=0.5
            ).add_to(eq_group)
    else:
        print(f"⚠️ 警告: 未找到 {eq_csv}")


    # ==========================================
    # 3. 添加到地图并保存
    # ==========================================
    hc_group.add_to(m)
    eq_group.add_to(m)
    folium.LayerControl(position='topright').add_to(m)

    m.save(output_filename)
    print(f"🎉 地图已生成并保存为 '{output_filename}'。双击即可查看！")


if __name__ == "__main__":
    # 请确保这两个文件在同一目录下
    EARTHQUAKE_CSV = 'earthquakes_M65.csv'  # 上一轮对话获取的地震数据
    IMPACTED_CITIES_CSV = 'impacted_cities.csv'  # 刚才第一步提取的城市数据
    import pandas as pd

    df = pd.read_csv('impacted_cities.csv')
    print("当前CSV的列名是:", df.columns.tolist())
    df_eq = pd.read_csv('earthquakes_M65.csv')
    print("地震CSV的真实列名是:", df_eq.columns.tolist())
    create_hazards_impact_map(EARTHQUAKE_CSV, IMPACTED_CITIES_CSV, 'global_hazards_impact_map.html')