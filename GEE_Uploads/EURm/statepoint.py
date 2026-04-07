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


def process_and_plot_academic(row_data, output_filename):
    print(f"\nProcessing: {row_data.get('location_name', 'Unknown')}")

    # 1. 数据提取
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
    if len(s.dropna()) < 52: return False

    # 2. 预处理与平滑
    s_imputed = ts_knn_impute(s, window_size=4, k=5)
    s_rolling = s_imputed.rolling(window=4, center=True).mean().bfill().ffill()
    s_smoothed = s_rolling.ewm(alpha=0.7, adjust=False).mean()

    decomposition = seasonal_decompose(s_smoothed, model='additive', period=52)
    trend_component = decomposition.trend.interpolate(method='linear', limit_direction='both').bfill().ffill()

    # 3. 截取震前数据
    earthquake_date = pd.Timestamp(row_data['date'])
    pre_earthquake_trend = trend_component[trend_component.index < earthquake_date].dropna()
    post_earthquake_dates = trend_component[trend_component.index >= earthquake_date].index

    if pre_earthquake_trend.empty or len(post_earthquake_dates) == 0: return False

    # ---> NTL_{pre}: 震前基准
    date_pre = pre_earthquake_trend.index[-1]
    ntl_pre = pre_earthquake_trend.iloc[-1]

    # 4. Holt-Winters 预测 ---> \hat{y}_{t+h}
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

    # 5. 特征点提取 ---> NTL_{min} 和 NTL_{post}

    # NTL_{min}: 震后 12 周内最低点
    post_window_12 = trend_component[(trend_component.index >= earthquake_date) &
                                     (trend_component.index <= earthquake_date + pd.Timedelta(weeks=12))]
    if post_window_12.empty: return False
    date_min = post_window_12.idxmin()
    ntl_min = post_window_12.min()

    # NTL_{post}: 预测曲线与真实曲线首次重合值
    date_post, ntl_post = None, None
    if forecast_series is not None:
        trend_after_min = trend_component[trend_component.index > date_min]
        forecast_after_min = forecast_series[forecast_series.index > date_min]

        diff = trend_after_min - forecast_after_min
        intersections = diff[diff >= 0]
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

    # ==========================================
    # 6. 绘图 (严格应用论文公式与符号)
    # ==========================================
    fig, ax = plt.subplots(1, 1, figsize=(14, 7), dpi=150)

    # [1] 原始 NTL 数据
    ax.plot(s_smoothed.index, s_smoothed.values, color='#cccccc', linewidth=1.5,
            label=r'Smoothed NTL Data')

    # [2] 真实 NTL 观测曲线 (T_t)
    ax.plot(trend_component.index, trend_component.values, color='black', marker='.', markersize=4, linewidth=2,
            label=r'True NTL Curve ($T_t$)')

    # [3] 地震发生时间
    ax.axvline(x=earthquake_date, color='#e53935', linestyle='-', linewidth=3, alpha=0.8,
               label=rf'Earthquake Event')

    # [4] 震前水平静态参考线 (可选)
    ax.axhline(y=ntl_pre, color='#43a047', linestyle='-.', linewidth=2.5, alpha=0.4)

    # [5] 反事实预测基准线 (\hat{y}_{t+h})
    if forecast_series is not None and not forecast_series.empty:
        concat_index = pre_earthquake_trend.index[-1:].append(forecast_series.index)
        concat_values = np.append(pre_earthquake_trend.values[-1], forecast_series.values)
        ax.plot(concat_index, concat_values, color='#1e88e5', linestyle='--', linewidth=2.5,
                label=r'Counterfactual Baseline ($\hat{y}_{t+h}$)')

    # [6] 三个核心状态点散点
    ax.scatter(date_pre, ntl_pre, color='green', s=160, zorder=10,
               label=r'$NTL_{pre}$ (Pre-event Baseline)')
    ax.scatter(date_min, ntl_min, color='blue', s=160, zorder=10,
               label=r'$NTL_{min}$ (Response Minimum)')
    ax.scatter(date_post, ntl_post, color='orange', s=160, zorder=10,
               label=r'$NTL_{post}$ (Recovery State)')

    # 视图范围调整
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

    # 标题与坐标轴
    ax.set_title(f"Resilience Point Extraction: {row_data.get('location_name')} (M{row_data.get('magnitude')})",
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Date (t)', fontsize=13)
    ax.set_ylabel('NTL Intensity', fontsize=13)

    # 图例设置 (将渲染数学公式)
    ax.legend(fontsize=12, loc='best', frameon=True, shadow=True, edgecolor='black')

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
            process_and_plot_academic(row.iloc[0], "statepoint.png")
        else:
            print("Target location not found.")