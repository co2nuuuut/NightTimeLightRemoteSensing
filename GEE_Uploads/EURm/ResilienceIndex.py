import os
import glob
import re
import json
import base64
import numpy as np
import pandas as pd
import folium
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from statsmodels.tsa.seasonal import seasonal_decompose
from io import BytesIO

matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')


def set_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        print("✅ 绘图字体已设置为 'SimHei' (黑体)。")
    except:
        print("⚠️ 警告：未能找到 'SimHei' 字体。")
    plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 第一部分：多指标综合评价算法 (TOPSIS + CRITIC + 熵权)
# ==========================================
def calculate_comprehensive_resilience_index(X, D):
    N, M = X.shape
    X_unified = X.copy()

    # I: 统一指标方向
    for j in range(M):
        if D[j] == -1:
            X_unified[:, j] = -X_unified[:, j]
    X_unified = X_unified - np.min(X_unified, axis=0) + 1e-8

    # II: 计算组合权重
    P = X_unified / np.sum(X_unified, axis=0, keepdims=True)
    Ej = -np.sum(P * np.log(P + 1e-8), axis=0) / np.log(N)
    We = (1 - Ej) / np.sum(1 - Ej)

    sigma = np.std(X, axis=0, ddof=1)
    R = np.corrcoef(X, rowvar=False)
    conflict = np.sum(1 - R, axis=1)
    C = sigma * conflict
    Wc = C / np.sum(C)

    CV_e = np.std(We) / (np.mean(We) + 1e-8)
    CV_c = np.std(Wc) / (np.mean(Wc) + 1e-8)
    lambda_e = CV_e / (CV_e + CV_c)
    lambda_c = 1 - lambda_e
    combined_weights = lambda_e * We + lambda_c * Wc
    combined_weights = combined_weights / np.sum(combined_weights)

    # III: 改进TOPSIS
    Vp = np.max(X_unified, axis=0)
    Vn = np.min(X_unified, axis=0)
    X_weighted = X_unified * combined_weights
    Vp_weighted = Vp * combined_weights
    Vn_weighted = Vn * combined_weights

    def cosine_distance(vec, ideal):
        return 1 - (np.dot(vec, ideal) / ((np.linalg.norm(vec) + 1e-8) * (np.linalg.norm(ideal) + 1e-8)))

    S_plus = np.array([cosine_distance(X_weighted[i], Vp_weighted) for i in range(N)])
    S_minus = np.array([cosine_distance(X_weighted[i], Vn_weighted) for i in range(N)])
    C_score = S_minus / (S_plus + S_minus + 1e-8)

    # IV: 分位数校准
    Q_05, Q_95 = np.percentile(C_score, 5), np.percentile(C_score, 95)
    RI = np.clip((C_score - Q_05) / (Q_95 - Q_05 + 1e-8), 0, 1)

    return RI, combined_weights


# ==========================================
# 第二部分：时序预处理、特征提取与制图
# ==========================================
def process_timeseries_and_plot(row_data):
    """
    预处理时间序列，计算下降率和恢复率，并生成 Base64 图片
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

            # --- 核心预处理：处理 -999 和异常零值 ---
            if pd.notna(value) and value != -999:
                time_series_data[time_stamp] = float(value)
            else:
                time_series_data[time_stamp] = np.nan

    s = pd.Series(time_series_data).sort_index()
    # 线性插值补全
    s_interpolated = s.interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(
        method='ffill')

    if s_interpolated.isnull().all() or len(s_interpolated) < 104:
        return np.nan, np.nan, None

    # 指数加权平滑与季节性分解
    s_smoothed = s_interpolated.ewm(alpha=0.7, adjust=False).mean()
    decomposition = seasonal_decompose(s_smoothed, model='additive', period=52)
    trend = decomposition.trend.interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(
        method='ffill')

    earthquake_date = pd.Timestamp(row_data['date'])

    # --- 特征提取：寻找震前基准、震后谷底、震后恢复期 ---
    pre_trend = trend[trend.index < earthquake_date]
    if pre_trend.empty: return np.nan, np.nan, None

    # 震前基准
    ntl_pre = pre_trend.iloc[-1]
    date_pre = pre_trend.index[-1]

    # 震后3个月 (约12周) 内寻找最低点 ( Drop阶段 )
    drop_window = trend[(trend.index >= earthquake_date) & (trend.index <= earthquake_date + pd.DateOffset(weeks=12))]
    if drop_window.empty: return np.nan, np.nan, None
    ntl_min = drop_window.min()
    date_min = drop_window.idxmin()

    # 震后1年 (约52周) 作为恢复考察期结束点
    rec_window = trend[(trend.index > date_min) & (trend.index <= earthquake_date + pd.DateOffset(weeks=52))]
    if rec_window.empty:
        ntl_post, date_post = ntl_min, date_min
    else:
        ntl_post = rec_window.iloc[-1]
        date_post = rec_window.index[-1]

    # --- 计算下降率和恢复率 ---
    base_val = max(ntl_pre, 1e-5)  # 防除以0
    drop_rate = (ntl_pre - ntl_min) / base_val
    recovery_rate = (ntl_post - ntl_min) / base_val

    # ================= 绘图部分 =================
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.plot(s_smoothed.index, s_smoothed.values, color='lightgray', linestyle='-', linewidth=1.5,
            label='原始周度数据 (波动)')
    ax.plot(trend.index, trend.values, color='black', marker='', linestyle='-', linewidth=2, label='城市发展趋势')

    ax.axvline(x=earthquake_date, color='red', linestyle='--', linewidth=2,
               label=f'地震发生 ({earthquake_date.date()})')
    ax.axhline(y=ntl_pre, color='green', linestyle=':', linewidth=2, label='震前发展水平基准线')

    # 标注特征点以供验证提取算法的准确性
    ax.scatter(date_pre, ntl_pre, color='green', s=100, zorder=5, label='震前状态 (Pre)')
    ax.scatter(date_min, ntl_min, color='blue', s=100, zorder=5, label='震后谷底 (Min)')
    ax.scatter(date_post, ntl_post, color='orange', s=100, zorder=5, label='恢复期末 (Post)')

    zoom_window_weeks = 52
    start_date = earthquake_date - pd.DateOffset(weeks=zoom_window_weeks)
    end_date = earthquake_date + pd.DateOffset(weeks=zoom_window_weeks)
    ax.set_xlim(start_date, end_date)

    zoomed_data = trend[start_date:end_date]
    if not zoomed_data.empty:
        ax.set_ylim(zoomed_data.min() * 0.95, zoomed_data.max() * 1.05)

    ax.set_title(
        f'韧性指标特征分析: {row_data.get("location_name")}\n下降率: {drop_rate:.2%} | 恢复率: {recovery_rate:.2%}',
        fontsize=15)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('周度夜光趋势值', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    fig.autofmt_xdate()
    plt.tight_layout()

    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png', dpi=100)
    plt.close(fig)
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    return drop_rate, recovery_rate, encoded


# ==========================================
# 第三部分：主控流程 (批量读取 -> 提特征 -> 算指数 -> 画地图)
# ==========================================
def run_integrated_pipeline(directory_path, output_html='urban_resilience_map_integrated.html'):
    set_chinese_font()
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

    if not csv_files:
        print(f"🔴 错误：在 '{directory_path}' 中没有找到CSV文件。")
        return

    print(f"找到 {len(csv_files)} 个CSV文件，正在合并...")
    df_raw = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    df_raw = df_raw.drop_duplicates(subset=['location_name', 'date'])
    print(f"合并完成，总计 {len(df_raw)} 条独立地震记录。")

    print("\n>>> 正在逐行提取时间序列、分析下降率与恢复率、生成趋势图...")
    drop_rates, rec_rates, b64_images = [], [], []

    for index, row in df_raw.iterrows():
        dr, rr, b64 = process_timeseries_and_plot(row)
        drop_rates.append(dr)
        rec_rates.append(rr)
        b64_images.append(b64)
        if (index + 1) % 20 == 0:
            print(f"已处理 {index + 1}/{len(df_raw)} 条...")

    # 保存至 DataFrame
    df_raw['Drop_Rate'] = drop_rates
    df_raw['Recovery_Rate'] = rec_rates
    df_raw['Base64_Plot'] = b64_images

    # 过滤掉时序缺失导致提取失败的数据
    df_valid = df_raw.dropna(subset=['Drop_Rate', 'Recovery_Rate']).copy()
    print(f"\n特征提取完毕，有效样本数：{len(df_valid)} / {len(df_raw)}")

    if len(df_valid) == 0:
        print("🔴 错误：没有有效的数据可以进行韧性计算！")
        return

    print("\n>>> 开始进行综合韧性指数 (RI) 评价计算...")
    # 构建输入特征矩阵 X
    # 此处指标: 1. 夜光下降率(越小越好->负向)  2. 夜光恢复率(越大越好->正向)
    # 也可以加入其它标量，如 Magnitude(-1), Population(1) 等，这里专注用你的两个指标
    X = df_valid[['Drop_Rate', 'Recovery_Rate']].values
    D = np.array([-1, 1])  # 下降率是负向指标(-1), 恢复率是正向指标(1)

    RI, weights = calculate_comprehensive_resilience_index(X, D)
    df_valid['Resilience_Index'] = RI

    print("\n--- 指标权重计算结果 ---")
    print(f"夜光下降率: {weights[0]:.4f}")
    print(f"夜光恢复率: {weights[1]:.4f}")

    print("\n>>> 正在生成交互式可视化地图...")
    m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB positron')
    colormap = cm.get_cmap('RdYlGn')  # 红-黄-绿 渐变色盘

    for index, row in df_valid.iterrows():
        b64_image = row['Base64_Plot']
        if not b64_image: continue

        loc_name = row.get('location_name', 'N/A')
        mag = row.get('magnitude', 0)
        date = row.get('date', 'N/A')
        ri = row['Resilience_Index']
        dr = row['Drop_Rate']
        rr = row['Recovery_Rate']

        try:
            geo_data = json.loads(row['.geo'])
            lon, lat = geo_data['coordinates']
        except Exception:
            continue

        # 映射RI颜色 (RI越接近1，颜色越绿；越接近0，颜色越红)
        hex_color = mcolors.to_hex(colormap(ri))

        html = f"""
        <div style="font-family: sans-serif; min-width:600px;">
            <h3 style="margin-bottom: 5px;">{loc_name}</h3>
            <b>震级:</b> {mag} | <b>发生日期:</b> {date}<br>
            <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <b style="font-size: 16px; color: {hex_color};">⭐ 综合韧性指数 (RI): {ri:.4f}</b><br>
                <b>📉 阶段 1: 夜光下降率 (抵抗力):</b> {dr:.2%}<br>
                <b>📈 阶段 2: 夜光恢复率 (恢复力):</b> {rr:.2%}
            </div>
            <img src="data:image/png;base64,{b64_image}" style="width:100%;">
        </div>
        """

        iframe = folium.IFrame(html, width=800, height=500)  # 根据图表尺寸自适应
        popup = folium.Popup(iframe, max_width=850)

        # 将地图上的圆圈颜色设为计算出的韧性颜色！
        folium.CircleMarker(
            location=[lat, lon],
            radius=float(mag) * 1.5,
            popup=popup,
            tooltip=f"{loc_name} (RI: {ri:.3f})",
            color='black', weight=1,  # 黑色细边框
            fill=True,
            fill_color=hex_color,  # 内部填充使用映射颜色
            fill_opacity=0.8
        ).add_to(m)

    m.save(output_html)

    # 额外把计算好的表格结果保存一份
    csv_out = os.path.join(directory_path, "Calculated_Resilience_Results.csv")
    df_valid[['location_name', 'date', 'magnitude', 'Drop_Rate', 'Recovery_Rate', 'Resilience_Index']].to_csv(csv_out,
                                                                                                              index=False,
                                                                                                              encoding='utf-8-sig')

    print("\n-----------------------------------------------------")
    print(f"🎉 任务全部完成！")
    print(f"1. 带有图表和综合指数的交互地图已保存至: {output_html}")
    print(f"2. 指标特征提取数据表已保存至: {csv_out}")
    print("-----------------------------------------------------")


if __name__ == "__main__":
    csv_directory_path = r"D:\虚拟c盘\大创项目\GEE_Uploads\EURm\M65"
    if not os.path.isdir(csv_directory_path):
        print(f"🔴 错误：指定的路径 '{csv_directory_path}' 不存在。请修改为您电脑上的实际路径。")
    else:
        run_integrated_pipeline(csv_directory_path)