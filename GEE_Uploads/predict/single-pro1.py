import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import glob
import numpy as np
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')


# ==========================================
# 100% 复制同学代码里的 KNN 核心清洗函数
# ==========================================
def ts_knn_impute(series, window_size=4, k=5, epsilon=1e-5):
    arr = series.values.copy()
    n = len(arr)
    out = arr.copy()
    valid_windows = []
    valid_centers = []
    for i in range(window_size, n - window_size):
        win = arr[i - window_size: i + window_size + 1]
        if not np.isnan(win).any():
            features = np.delete(win, window_size)
            valid_windows.append(features)
            valid_centers.append(win[window_size])

    valid_windows = np.array(valid_windows)
    valid_centers = np.array(valid_centers)

    for i in range(n):
        if np.isnan(arr[i]):
            if len(valid_windows) < k: continue
            target_win = []
            for j in range(i - window_size, i + window_size + 1):
                if j == i: continue
                if 0 <= j < n and not np.isnan(arr[j]):
                    target_win.append(arr[j])
                else:
                    target_win.append(np.nanmean(arr))
            target_win = np.array(target_win)
            distances = np.sqrt(np.sum((valid_windows - target_win) ** 2, axis=1))
            nearest_idx = np.argsort(distances)[:k]
            weights = 1 / (distances[nearest_idx] + epsilon)
            out[i] = np.sum(weights * valid_centers[nearest_idx]) / np.sum(weights)

    result_series = pd.Series(out, index=series.index)
    return result_series.interpolate(method='linear', limit_direction='both').bfill().ffill()


def process_and_plot_exact_friend_logic(row_data, output_filename):
    print(f"\nProcessing: {row_data.get('location_name', 'Unknown')}")

    # ---------------------------------------------------------
    # 第一步：完全照搬同学的数据提取逻辑
    # ---------------------------------------------------------
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

            if pd.notna(value) and value > 0 and value != -999:
                time_series_data[time_stamp] = float(value)
            else:
                time_series_data[time_stamp] = np.nan

    s = pd.Series(time_series_data).sort_index()
    if len(s.dropna()) < 52:
        print("Error: Insufficient data points.")
        return False

    # ---------------------------------------------------------
    # 第二步：完全照搬同学的预处理逻辑 (KNN -> Rolling -> EWM -> Decompose)
    # ---------------------------------------------------------
    s_imputed = ts_knn_impute(s, window_size=4, k=5)
    s_rolling = s_imputed.rolling(window=4, center=True).mean().bfill().ffill()
    s_smoothed = s_rolling.ewm(alpha=0.7, adjust=False).mean()

    decomposition = seasonal_decompose(s_smoothed, model='additive', period=52)
    trend_component = decomposition.trend.interpolate(method='linear', limit_direction='both').bfill().ffill()

    # 数据分割
    earthquake_date = pd.Timestamp(row_data['date'])
    pre_earthquake_trend = trend_component[trend_component.index < earthquake_date].dropna()
    post_earthquake_dates = trend_component[trend_component.index >= earthquake_date].index

    if pre_earthquake_trend.empty or len(post_earthquake_dates) == 0:
        return False

    # 提取状态点 - Pre (绿点)
    date_pre = pre_earthquake_trend.index[-1]
    ntl_pre = pre_earthquake_trend.iloc[-1]

    # ---------------------------------------------------------
    # 第三步：完全照搬同学的 Holt-Winters 预测参数 (无阻尼)
    # ---------------------------------------------------------
    train_values = np.clip(pre_earthquake_trend.values.astype(np.float64), 1e-4, None)
    forecast_series = None

    try:
        hw_model = ExponentialSmoothing(
            train_values,
            trend='add',
            seasonal='add',
            seasonal_periods=52,
            initialization_method='estimated'
        ).fit()
        forecast_vals = hw_model.forecast(steps=len(post_earthquake_dates))
        forecast_series = pd.Series(forecast_vals, index=post_earthquake_dates)
    except Exception as e:
        print(f"Holt-Winters failed: {e}")

    # ---------------------------------------------------------
    # 第四步：计算其余状态点 (Min 和 Post)
    # ---------------------------------------------------------
    # Min (蓝点): 震后 12 周内最低点
    post_window_12 = trend_component[(trend_component.index >= earthquake_date) &
                                     (trend_component.index <= earthquake_date + pd.Timedelta(weeks=12))]
    if post_window_12.empty: return False
    date_min = post_window_12.idxmin()
    ntl_min = post_window_12.min()

    # Post (橙点): 谷底之后的交点，若无则取1年
    date_post, ntl_post = None, None
    if forecast_series is not None:
        trend_after_min = trend_component[trend_component.index > date_min]
        forecast_after_min = forecast_series[forecast_series.index > date_min]

        diff = trend_after_min - forecast_after_min
        intersections = diff[diff >= 0]  # 寻找实际曲线跨越预测曲线的点
        if not intersections.empty:
            date_post = intersections.index[0]
            ntl_post = trend_component.loc[date_post]

    if date_post is None:
        one_year_later = earthquake_date + pd.Timedelta(weeks=52)
        if one_year_later in trend_component.index:
            date_post = one_year_later
        else:
            post_trend = trend_component[trend_component.index >= earthquake_date]
            date_post = post_trend.index[np.argmin(np.abs(post_trend.index - one_year_later))]
        ntl_post = trend_component.loc[date_post]

    # ---------------------------------------------------------
    # 第五步：绘图 (全英文，避免中文方块)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(14, 7), dpi=120)

    # 完全按同学的样式画线
    ax.plot(s_smoothed.index, s_smoothed.values, color='#cccccc', linewidth=1.5, label='Smoothed Raw Data')
    ax.plot(trend_component.index, trend_component.values, color='black', marker='.', markersize=4, linewidth=2,
            label='Actual Trend (De-seasonalized)')
    ax.axvline(x=earthquake_date, color='#e53935', linestyle='-', linewidth=3, alpha=0.8,
               label=f'Earthquake ({earthquake_date.date()})')
    ax.axhline(y=ntl_pre, color='#43a047', linestyle='-.', linewidth=2.5, label='Static Baseline (Pre-level)')

    if forecast_series is not None and not forecast_series.empty:
        concat_index = pre_earthquake_trend.index[-1:].append(forecast_series.index)
        concat_values = np.append(pre_earthquake_trend.values[-1], forecast_series.values)
        ax.plot(concat_index, concat_values, color='#1e88e5', linestyle='--', linewidth=2.5,
                label='Dynamic Baseline (Holt-Winters)')

    # 添加三个状态点散点
    ax.scatter(date_pre, ntl_pre, color='green', s=150, zorder=10, label='Pre-disaster Point')
    ax.scatter(date_min, ntl_min, color='blue', s=150, zorder=10, label='Min Point (Bottom)')
    ax.scatter(date_post, ntl_post, color='orange', s=150, zorder=10, label='Post Point (Recovery)')

    # 限制视图范围
    start_date = earthquake_date - pd.DateOffset(weeks=52)
    end_date = earthquake_date + pd.DateOffset(weeks=104)
    ax.set_xlim(start_date, end_date)

    zoomed_data = trend_component[start_date:end_date]
    if not zoomed_data.empty:
        y_min = zoomed_data.min() * 0.85
        y_max = zoomed_data.max() * 1.15
        if forecast_series is not None:
            y_max = max(y_max, forecast_series.max() * 1.15)
        ax.set_ylim(y_min, y_max)

    ax.set_title(f"Resilience Analysis: {row_data.get('location_name')} (M{row_data.get('magnitude')})", fontsize=16,
                 fontweight='bold')
    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('Weekly NTL Radiance', fontsize=13)
    ax.legend(fontsize=11, loc='best', frameon=True, shadow=True)

    fig.autofmt_xdate(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(output_filename)
    print(f"Plot saved successfully: {output_filename}")
    plt.close()


if __name__ == "__main__":
    csv_dir = r"D:\虚拟c盘\大创项目\GEE_Uploads\EURm\M65"
    target_location = "73 km ENE of Namie, Japan"

    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        print("No CSV files found.")
    else:
        df_all = pd.concat([pd.read_csv(f) for f in csv_files])
        row = df_all[df_all['location_name'] == target_location]

        if not row.empty:
            process_and_plot_exact_friend_logic(row.iloc[0], "resilience_pure_HW.png")
        else:
            print("Target location not found.")