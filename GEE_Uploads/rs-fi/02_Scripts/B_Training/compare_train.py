import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- 导入五大算法 ---
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# ================= 配置区域 =================
DATA_PATH = "../../01_Processed_Data/final_training_data.csv"
BEST_MODEL_PATH = "../../03_Models/best_model.pkl"

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']  # 防止中文乱码(如果有)
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 数据加载与预处理 =================
print("🚀 阶段一：数据准备...")

df = pd.read_csv(DATA_PATH)
print(f"   原始样本量: {len(df)}")

# --- 特征工程 (对数变换) ---
# 目标变量 Y
df['log_gdp'] = np.log1p(df['GDP'])

# 特征变量 X
# 自动判断是用 NTL_sum 还是 NTL_mean
ntl_col = 'NTL_sum' if 'NTL_sum' in df.columns else 'NTL_mean'
print(f"   使用夜光特征列: {ntl_col}")

df['log_ntl'] = np.log1p(df[ntl_col])
# NDVI 和 Precip 保持原样 (或者根据分布也可以取对数，这里暂时保持原样)
feature_cols = ['log_ntl', 'NDVI_mean', 'Precip_mean']
target_col = 'log_gdp'

# 清洗空值
df_clean = df.dropna(subset=feature_cols + [target_col])
X = df_clean[feature_cols]
y = df_clean[target_col]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   训练集: {len(X_train)}, 测试集: {len(X_test)}")

# ================= 2. 定义模型群 =================
print("\n🚀 阶段二：初始化五大算法...")

# 注意：SVR 和 MLP 对数据尺度敏感，必须先标准化 (StandardScaler)
# 我们使用 make_pipeline 将标准化和模型打包在一起

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        random_state=42, n_jobs=-1, verbosity=0
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        random_state=42, n_jobs=-1, verbose=-1
    ),

    "SVR (Support Vector)": make_pipeline(
        StandardScaler(),
        SVR(C=1.0, epsilon=0.1, kernel='rbf')
    ),

    "MLP (Deep Learning)": make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(100, 50),  # 两个隐藏层
            max_iter=1000,  # 增加迭代次数防止不收敛
            activation='relu',
            solver='adam',
            random_state=42
        )
    )
}

# ================= 3. 循环训练与评估 =================
print("\n🚀 阶段三：开始大比武 (Training & Evaluation)...")

results = []
predictions = {}  # 存储每个模型的预测结果以便画图

best_score = -np.inf
best_model_name = ""
best_model_obj = None

for name, model in models.items():
    start_time = time.time()
    print(f"   Running {name}...", end="")

    # 训练
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    # 评估
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    cost_time = time.time() - start_time
    print(f" Done! ({cost_time:.2f}s) -> R²={r2:.4f}")

    # 记录结果
    results.append({
        "Model": name,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "Time(s)": cost_time
    })

    # 寻找最佳模型
    if r2 > best_score:
        best_score = r2
        best_model_name = name
        best_model_obj = model

# ================= 4. 结果展示 =================
print("\n" + "=" * 40)
print("🏆 最终成绩单 (按 R² 排序)")
print("=" * 40)

results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
print(results_df)

# 保存最佳模型
print(f"\n💾 正在保存冠军模型: [{best_model_name}]")
joblib.dump(best_model_obj, BEST_MODEL_PATH)

# ================= 5. 可视化对比 =================
print("\n📊 正在生成对比图表...")

# 创建画布：左边是柱状图，右边是散点图矩阵
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3)  # 2行3列

# --- 图1：R2 分数对比 (柱状图) ---
ax_bar = fig.add_subplot(gs[0, 0])
sns.barplot(x="R2", y="Model", data=results_df, palette="viridis", ax=ax_bar)
ax_bar.set_title("Model Comparison (R² Score)", fontsize=14)
ax_bar.set_xlim(0, 1)
for i, v in enumerate(results_df["R2"]):
    ax_bar.text(v + 0.01, i, f"{v:.3f}", va='center')

# --- 图2-6：各模型 真实值 vs 预测值 (散点图) ---
# 我们画出前5个模型的散点图
plot_positions = [gs[0, 1], gs[0, 2], gs[1, 0], gs[1, 1], gs[1, 2]]

for idx, model_name in enumerate(models.keys()):
    if idx >= 5: break

    ax = fig.add_subplot(plot_positions[idx])
    y_p = predictions[model_name]

    # 散点
    ax.scatter(y_test, y_p, alpha=0.4, s=10, color='blue')
    # 对角线 (完美预测线)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)

    score = results_df[results_df['Model'] == model_name]['R2'].values[0]
    ax.set_title(f"{model_name} (R²={score:.3f})")
    ax.set_xlabel("Actual Log GDP")
    ax.set_ylabel("Predicted")

plt.tight_layout()
plt.savefig("model_comparison_results.png", dpi=300)
print("✅ 图表已保存为: model_comparison_results.png")
plt.show()