import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import matplotlib.patches as mpatches

# ===== 1. 全局参数与样式设置 (核心可调部分) =====
plt.rcParams.update({
    'font.sans-serif': ['Arial', 'DejaVu Sans'],  # 更通用的无衬线字体
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.titlesize': 13,  # 标题字体大小
    'axes.labelsize': 11,  # 坐标轴标签字体大小
    'legend.fontsize': 9,  # 图例字体大小
})

# ---- 关键参数设定 (按需调整) ----
PLOT_PARAMS = {
    'impact_color': '#FFF7BC',        # 改进的荧光黄/淡琥珀色，更柔和
    'event_line_color': '#E41A1C',    # 红色，用于事件线
    'raw_line_color': '#D9D9D9',      # 原始数据线颜色
    'trend_line_color': '#262626',    # 趋势线颜色
    'pre_color': '#4DAF4A',           # Pre点颜色 (绿色)
    'min_color': '#377EB8',           # Min点颜色 (蓝色)
    'post_color': '#FF7F00',          # Post点颜色 (橙色)
    'marker_size': 100,               # 关键点大小
    'trend_line_width': 2.5,          # 趋势线宽度
    'event_line_style': '--',         # 事件线样式
    'grid_alpha': 0.2,                # 网格线透明度
}

FEATURE_PARAMS = {
    'pre_window': 8,       # 震前基准计算窗口（周）
    'response_window': 12, # 震后响应窗口（周）
    'recovery_week': 52,   # 恢复期（周）
}

# ===== 2. 模拟数据生成 (无需改动) =====
def generate_simulated_data(weeks=80, t0=20):
    """生成模拟 NTL, LST, NO2 时序数据"""
    np.random.seed(42)  # 确保结果可复现
    t = np.arange(weeks)
    impact = np.zeros(weeks)
    impact[t0:] = -0.5 * np.exp(-(t[t0:] - t0) / 10)
    ntl = 0.8 + impact + np.random.normal(0, 0.03, weeks)
    lst = 0.3 - impact * 0.8 + np.random.normal(0, 0.04, weeks)
    no2 = 0.4 - impact * 0.6 + np.random.normal(0, 0.05, weeks)
    return pd.DataFrame({
        'Week': t,
        'NTL': ntl,
        'LST': lst,
        'NO2': no2
    })


# ===== 3. 综合指数计算 (无需改动) =====
def calculate_comprehensive_index(df, weights):
    """计算综合韧性指数"""
    for col in ['NTL', 'LST', 'NO2']:
        df[f'{col}_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    df['LST_norm'] = 1 - df['LST_norm']
    df['NO2_norm'] = 1 - df['NO2_norm']
    df['ENV_Index'] = 0.5 * df['LST_norm'] + 0.5 * df['NO2_norm']
    df['Composite_Index'] = (df['NTL_norm'] * weights['NTL'] +
                             df['ENV_Index'] * weights['ENV'])
    return df


# ===== 4. 核心特征提取与绘图 (重点优化部分) =====
def plot_resilience_trend_enhanced(df, t0=20, params=PLOT_PARAMS, feat_params=FEATURE_PARAMS):
    """
    绘制优化后的时序趋势与韧性特征图
    """
    # 平滑处理
    window = max(9, int(len(df) / 8) // 2 * 2 + 1)
    df['Smooth_Index'] = savgol_filter(df['Composite_Index'],
                                        window_length=window,
                                        polyorder=3)
    # 核心特征点计算
    pre_val = df.loc[t0 - feat_params['pre_window']:t0 - 1, 'Smooth_Index'].mean()
    pre_time = t0 - 1
    response_df = df.loc[t0:t0 + feat_params['response_window']]
    min_val = response_df['Smooth_Index'].min()
    min_time = response_df['Smooth_Index'].idxmin()
    post_time = min(t0 + feat_params['recovery_week'], df.index.max())
    post_val = df.loc[post_time, 'Smooth_Index']
    DrR = (pre_val - min_val) / pre_val
    ReR = (post_val - min_val) / pre_val

    # ===== 创建图形 =====
    fig, ax = plt.subplots(figsize=(10, 5.5))  # 更优的宽高比

    # 1. 冲击阶段背景（改进的荧光黄）
    ax.axvspan(t0, t0 + feat_params['response_window'],
               facecolor=params['impact_color'],
               alpha=0.4,  # 提高透明度，使下方曲线更清晰
               edgecolor=None,
               label='冲击阶段')

    # 2. 原始波动线
    ax.plot(df['Week'], df['Composite_Index'],
            color=params['raw_line_color'],
            linewidth=0.8,
            alpha=0.7,
            label='原始波动',
            zorder=1)

    # 3. 平滑趋势线
    ax.plot(df['Week'], df['Smooth_Index'],
            color=params['trend_line_color'],
            linewidth=params['trend_line_width'],
            label='平滑趋势',
            zorder=2)

    # 4. 地震事件线
    ax.axvline(x=t0,
               color=params['event_line_color'],
               linestyle=params['event_line_style'],
               linewidth=2,
               label=f'地震 (t={t0})',
               zorder=3)

    # 5. 三个核心特征点（带边框的散点）
    for (time, val, color, label) in [
        (pre_time, pre_val, params['pre_color'], 'Pre (震前基准)'),
        (min_time, min_val, params['min_color'], 'Min (响应谷底)'),
        (post_time, post_val, params['post_color'], 'Post (恢复期末)')
    ]:
        ax.scatter(time, val,
                   color=color,
                   edgecolors='white',
                   linewidths=1.5,
                   s=params['marker_size'],
                   zorder=5)
        # 改进的标注
        ax.annotate(label.split()[0],
                    xy=(time, val),
                    xytext=(0, 15 if label.startswith('Min') else -15),  # Min点标注在上方
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    fontweight='bold',
                    color=color)

    # 6. 韧性指标文本框
    bbox_text = f'$DrR$ = {DrR:.3f}\n$ReR$ = {ReR:.3f}'
    ax.text(0.02, 0.97,
            bbox_text,
            transform=ax.transAxes,
            fontsize=10,
            fontfamily='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white",
                      edgecolor="#CCCCCC",
                      alpha=0.9))

    # ===== 图表美化 =====
    ax.set_title('典型城市灾害冲击下的时序趋势与韧性特征提取',
                 fontsize=13,
                 fontweight='bold',
                 pad=15)
    ax.set_xlabel('时间 (周)',
                  fontsize=11,
                  labelpad=8)
    ax.set_ylabel('综合韧性指数',
                  fontsize=11,
                  labelpad=8)
    ax.set_xlim(df['Week'].min() - 2, df['Week'].max() + 2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.grid(True, which='major', axis='both',
            linestyle='-', linewidth=0.5,
            alpha=params['grid_alpha'])

    # 改进的图例（去除重复项，更清晰）
    from collections import OrderedDict
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    # 手动控制图例项
    legend_handles = [
        mpatches.Patch(facecolor=params['impact_color'], alpha=0.4, label='冲击阶段'),
        plt.Line2D([0], [0], color=params['event_line_color'],
                   linestyle=params['event_line_style'], linewidth=2, label=f'地震 (t={t0})'),
        plt.Line2D([0], [0], color=params['raw_line_color'],
                   linewidth=0.8, alpha=0.7, label='原始波动'),
        plt.Line2D([0], [0], color=params['trend_line_color'],
                   linewidth=params['trend_line_width'], label='平滑趋势'),
    ]
    ax.legend(handles=legend_handles,
              loc='upper center',
              bbox_to_anchor=(0.5, -0.15),
              ncol=4,
              frameon=False,
              handlelength=2.5)

    plt.tight_layout()
    plt.show()

    # 打印计算结果
    print("=" * 40)
    print("韧性特征计算结果:")
    print(f"  震前基准 (NTL_pre 近似): {pre_val:.4f}")
    print(f"  响应谷底 (NTL_min 近似): {min_val:.4f}")
    print(f"  恢复期末 (NTL_post 近似): {post_val:.4f}")
    print(f"  下降率 (DrR): {DrR:.4f}")
    print(f"  恢复率 (ReR): {ReR:.4f}")
    print("=" * 40)


# ===== 5. 主程序 =====
if __name__ == "__main__":
    # 生成数据
    df_raw = generate_simulated_data(weeks=80, t0=20)

    # 设置权重 (可在此处调整NTL与环境的权重)
    weights = {'NTL': 0.6, 'ENV': 0.4}

    # 计算综合指数
    df_comp = calculate_comprehensive_index(df_raw, weights)

    # 绘制优化后的图表
    plot_resilience_trend_enhanced(df_comp, t0=20)