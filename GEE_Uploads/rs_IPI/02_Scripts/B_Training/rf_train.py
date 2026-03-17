import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib  # 用于保存模型

# ================= 配置区域 =================
DATA_PATH = "../../01_Processed_Data/final_training_data.csv"
MODEL_SAVE_PATH = "gdp_prediction_model.pkl"

# ================= 1. 加载与预处理 =================
print("🚀 开始 Stage 3：模型训练...")

# 读取数据
df = pd.read_csv(DATA_PATH)
print(f"1. 数据加载成功: {len(df)} 行样本")

# 检查数据分布 (可选)
# print(df.describe())

# --- 关键步骤：对数变换 (Log Transformation) ---
# GDP 和 夜光总量通常呈幂律分布（跨度极大），直接训练效果不好。
# 取对数可以将数据拉伸成接近正态分布，极大提升线性关系。
print("2. 执行特征工程 (对数变换)...")

# 处理目标变量 Y
df['log_gdp'] = np.log1p(df['GDP'])

# 处理特征变量 X
# 优先使用 NTL_sum (总量对总量相关性最高)，如果只有 mean 就用 mean
if 'NTL_sum' in df.columns:
    df['log_ntl'] = np.log1p(df['NTL_sum'])
    print("   -> 使用夜光总量 (NTL_sum) 作为核心特征")
else:
    df['log_ntl'] = np.log1p(df['NTL_mean'])
    print("   -> 使用夜光均值 (NTL_mean) 作为核心特征")

# NDVI 和 Precip 通常不需要取对数，直接使用
# 我们构建特征列表
features = ['log_ntl', 'NDVI_mean', 'Precip_mean']
target = 'log_gdp'

# 去除空值 (以防万一)
df_clean = df.dropna(subset=features + [target])
print(f"   -> 清洗后剩余样本: {len(df_clean)}")

# ================= 2. 划分训练集与测试集 =================
X = df_clean[features]
y = df_clean[target]

# 80% 训练，20% 测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"3. 数据划分: 训练集 {len(X_train)}, 测试集 {len(X_test)}")

# ================= 3. 训练随机森林 =================
print("4. 正在训练随机森林回归模型 (Random Forest)...")

rf_model = RandomForestRegressor(
    n_estimators=200,   # 树的数量
    max_depth=None,     # 深度不限
    min_samples_split=2,
    random_state=42,
    n_jobs=-1           # 使用所有CPU核心加速
)

rf_model.fit(X_train, y_train)
print("   -> 训练完成！")

# ================= 4. 模型评估 =================
print("5. 评估模型表现...")

# 预测
y_pred = rf_model.predict(X_test)

# 计算指标
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "="*30)
print(f"🌟 模型评估结果 (Test Set):")
print(f"   R² (决定系数): {r2:.4f}  <-- 越接近 1 越好 (通常 >0.85 即为优秀)")
print(f"   MAE (平均绝对误差): {mae:.4f}")
print(f"   RMSE (均方根误差): {rmse:.4f}")
print("="*30 + "\n")

# ================= 5. 可视化与保存 =================

# 保存模型
joblib.dump(rf_model, MODEL_SAVE_PATH)
print(f"💾 模型已保存至: {MODEL_SAVE_PATH}")

# 绘制 真实值 vs 预测值 散点图
plt.figure(figsize=(10, 6))
# 绘制散点
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', s=15, label='Samples')
# 绘制完美预测线 (对角线)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')

plt.title(f'Satellite Predictions vs Actual GDP (Log Scale)\nR² = {r2:.3f}', fontsize=14)
plt.xlabel('Actual Log GDP (World Bank)', fontsize=12)
plt.ylabel('Predicted Log GDP (Satellite Model)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 保存图片
plt.savefig('model_performance.png', dpi=300)
print("📊 性能评估图已保存为: model_performance.png")
plt.show()

# 打印特征重要性
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
print("\n🔍 特征重要性排名:")
for f in range(X.shape[1]):
    print(f"   {f+1}. {features[indices[f]]}: {importances[indices[f]]:.4f}")