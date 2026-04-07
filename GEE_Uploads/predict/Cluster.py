import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 基础配置 =====================
# 学术期刊高对比度配色（8种，彻底解决索引报错）
journal_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 中文显示
plt.rcParams["axes.unicode_minus"] = False    # 负号显示
plt.rcParams["figure.dpi"] = 100
plt.rcParams['font.size'] = 11

# ===================== 2. 读取数据 =====================
file_path = r"D:\城市聚类\Calculated_Resilience_Results_total.csv"
df = pd.read_csv(file_path, encoding="gbk")
# 聚类特征：恢复力指数 + 发展指数（删除缺失值）
df_cluster = df[["Resilience_Index", "Development_Index"]].dropna()
# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster)

# ===================== 3. 轮廓系数法：自动计算最优聚类数K（核心修改） =====================
k_range = range(2, 8)
sil_score = []
# 遍历计算每个K值的轮廓系数
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    sil_score.append(silhouette_score(X_scaled, labels))

# 选择轮廓系数最大的K作为最优聚类数
best_k = k_range[np.argmax(sil_score)]

# 绘制轮廓系数评估图
plt.figure(figsize=(8, 5))
plt.plot(k_range, sil_score, "o-", color="#000000", linewidth=2.5, markersize=10)
plt.axvline(x=best_k, color="red", linestyle="--", linewidth=2, label=f"最优K={best_k}")
plt.title("轮廓系数法 - 最优聚类数确定", fontweight="bold", fontsize=14)
plt.xlabel("聚类数量 K", fontsize=12)
plt.ylabel("轮廓系数", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("轮廓系数最优K.png", dpi=300, bbox_inches="tight")
plt.show()

# ===================== 4. K-means聚类（使用轮廓系数确定的最优K） =====================
kmeans_model = KMeans(n_clusters=best_k, random_state=42)
cluster_labels = kmeans_model.fit_predict(X_scaled)
# 将聚类结果写入原数据
df["Cluster"] = cluster_labels
# 还原质心坐标（原始数据尺度）
centers = scaler.inverse_transform(kmeans_model.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=["城市综合韧性指数", "城市发展水平指数"])

# ===================== 5. 结合指标意义为聚类打标签 =====================
def get_cluster_label(center):
    resilience_dev = df_cluster["Resilience_Index"].mean()
    develop_dev = df_cluster["Development_Index"].mean()
    
    r = center[0]
    d = center[1]
    
    if r >= resilience_dev and d >= develop_dev:
        return "高发展水平稳定型"
    elif r >= resilience_dev and d < develop_dev:
        return "低发展水平稳定型"
    elif r < resilience_dev and d >= develop_dev:
        return "高发展水平脆弱型"
    else:
        return "待提升型（低恢复力+低发展）"

# 为每个簇生成标签
cluster_names = [get_cluster_label(center) for center in centers]
# 将标签映射到数据
label_map = {i: cluster_names[i] for i in range(best_k)}
df["Cluster_Label"] = df["Cluster"].map(label_map)

# ===================== 6. 输出聚类结果 =====================
print("="*60)
print(f"📊 轮廓系数法自动确定最优聚类数：K = {best_k}")
print(f"🎯 最优轮廓系数：{max(sil_score):.4f}")
print("="*60)
print("📈 各簇质心（原始指标值）：")
print(centers_df.round(4))
print("="*60)
print("📦 各簇样本数量与类型标签：")
result = df["Cluster_Label"].value_counts().reset_index()
result.columns = ["城市类型标签", "样本数量"]
print(result)
print("="*60)

# ===================== 7. 期刊风格聚类可视化 =====================
plt.figure(figsize=(10, 6))
for i in range(best_k):
    mask = df["Cluster"] == i
    plt.scatter(
        df.loc[mask, "Resilience_Index"],
        df.loc[mask, "Development_Index"],
        c=journal_colors[i],
        s=70,
        alpha=0.9,
        label=f"簇{i+1}：{cluster_names[i]}"
    )
# 绘制质心
plt.scatter(centers[:, 0], centers[:, 1], 
           c="black", marker="*", s=400, 
           edgecolors="gold", linewidth=2, label="簇质心")

# plt.title("城市韧性与发展水平聚类", fontweight="bold", fontsize=14)
plt.xlabel("城市综合韧性指数", fontsize=14)
plt.ylabel("城市发展水平指数", fontsize=14)
plt.legend(loc="best", framealpha=0.9)
plt.grid(alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("期刊风格_聚类结果图.png", dpi=300, bbox_inches="tight")
plt.show()

# ===================== 8. 保存带标签的结果文件 =====================
output_path = file_path.replace(".csv", "_带聚类标签.csv")
df.to_csv(output_path, index=False, encoding="gbk")
print(f"✅ 带分类标签的数据已保存至：\n{output_path}")