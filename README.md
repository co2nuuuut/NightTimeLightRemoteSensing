# 基于卫星遥感的全球城市韧性分析方法及其系统开发

---

## 📖 项目简介

本项目旨在构建一套基于卫星遥感的**全球城市韧性量化分析框架**，融合夜间灯光遥感（NTL）、地表温度（LST）、大气 NO₂ 浓度及社会经济等多源数据，以"演进式韧性"理论为核心，分析城市在地震、飓风等自然灾害冲击下的响应与恢复模式。

**核心目标：**
1. 构建基于 EURm（Enhanced Urban Resilience Metric）指标的城市韧性量化模型
2. 建立城市聚类模型，揭示全球城市韧性分异规律
3. 开发集数据处理、模型分析与智能决策支持于一体的 WebGIS 系统平台
4. 集成大语言模型，开发"城市韧性智能体"

---

## 🗂️ 仓库结构

```
大创项目/
│
├── GEE_Uploads/                   # 核心分析代码与数据（Google Earth Engine 导出）
│   ├── EURm/                      # 地震韧性 EURm 模型
│   │   ├── EURm.py                # 批量城市地震韧性分析脚本
│   │   ├── single.py              # 单城市韧性分析脚本
│   │   ├── M65/                   # M6.5+ 地震 NTL 周度数据
│   │   └── *.html                 # 交互式韧性可视化地图（5个）
│   │
│   ├── hurricane/                 # 飓风韧性分析（初版）
│   │   ├── hurricane.py           # 基于 DBSCAN 聚类的飓风冲击分析
│   │   └── *.html                 # 飓风影响可视化地图（3个）
│   │
│   ├── hurricane2/                # 飓风韧性分析（扩展版，136批次）
│   │   ├── webpage.py             # 扩展版飓风分析与可视化
│   │   ├── data1/, data2/         # 飓风 NTL 周度数据
│   │   └── hurricane_impact_resilience_map.html
│   │
│   ├── LST/                       # 地表温度（LST）韧性分析
│   │   ├── lst.py                 # MODIS LST 周度数据分析脚本
│   │   ├── urban_earthquake_LST_weekly/  # LST 周度数据
│   │   └── urban_earthquake_resilience_LST.html
│   │
│   ├── NO2/                       # 大气 NO₂ 韧性分析
│   │   ├── no2.py                 # TROPOMI NO₂ 数据分析脚本
│   │   ├── drive-download-.../    # NO₂ 原始数据
│   │   └── urban_earthquake_resilience_NO2_uncalibrated.html
│   │
│   ├── rs-fi/                     # 遥感 → GDP 预测流程（完整 ML Pipeline）
│   │   ├── 00_Raw_Data/           # 原始 NTL 与 GDP 数据
│   │   ├── 01_Processed_Data/     # 预处理后数据
│   │   ├── 02_Scripts/            # 分析脚本
│   │   │   ├── A_Data_Prep/       # 数据准备
│   │   │   ├── B_Training/        # 模型训练（XGBoost / 随机森林）
│   │   │   ├── C_Validation/      # 模型验证
│   │   │   └── D_Application/     # 应用推断
│   │   ├── 03_Models/             # 训练好的模型文件（.pkl）
│   │   └── 04_Results_Figures/    # 分析图表（15+张）
│   │
│   ├── rs_IPI/                    # 遥感 → 工业增加值预测流程
│   │   ├── 00_Raw_Data/
│   │   ├── 01_Processed_Data/
│   │   └── 02_Scripts/
│   │
│   ├── 其他/                       # 全球地震可视化脚本
│   │   # final.py、picss.py 等
│   │
│   └── hassemap.py                # 辅助地图工具脚本
│
├── NTL/                           # 夜间灯光遥感文献资料
│   ├── 夜光遥感总结报告与研究方案.md  # 10篇核心文献综述与研究方案
│   ├── 夜光遥感论文汇总/             # 分类整理的参考文献
│   └── *.jpg                      # 参考图像资料
│
├── 实验结果/                        # 各模块实验输出结果
│   ├── EURm (1).ipynb             # EURm 分析 Jupyter Notebook
│   ├── hurricanev1.ipynb          # 飓风分析 Jupyter Notebook
│   ├── Taiwan-hualian-2024.4.2/   # 台湾花莲地震（2024.4.2）案例
│   ├── city_hurricane_NTL_weekly_analysis/  # 飓风 NTL 周度数据（39批次）
│   ├── urban_earthquake_LST_weekly/         # 地震 LST 周度分析结果
│   ├── urban_earthquake_NO2_weekly_M65_pop50k/ # NO₂ 周度分析结果
│   └── test/                      # 测试数据与脚本
│
├── earthquake/                    # 地震原始数据下载（USGS）
│   ├── drive-download-2025101*/   # 多批次地震数据
│   └── weekly/                    # 周度汇总数据
│
├── 中期进度汇总.md                  # 项目中期进度报告（2026.03.16）
├── 基于夜间灯光数据的城市地震韧性分析.docx  # 核心思路文档
├── 论文汇总.docx                   # 3篇核心文献深度解读
├── 开题答辩/                       # 开题报告与答辩材料
├── 中期检查/                       # 中期检查材料
├── 学校文件/                       # 项目申报与管理文件
├── 论文集/                         # 收集的参考论文 PDF
├── arxiv-daily/                   # arXiv 每日追踪文献
└── drive-download-.../            # 原始数据备份
```

---

## 🔬 核心研究模块

### 1. 地震韧性分析（EURm 模型）

基于 Zhang et al.（2024, *Science China*）提出的 **EURm（Evolving Urban Resilience Metric）** 指标，对全球 M6.5+ 地震（城市人口 5 万+）进行韧性量化分析。

| 特性 | 说明 |
|------|------|
| **数据源** | NPP/VIIRS 月度/周度 NTL 数据（GEE 导出） |
| **时间窗口** | 震前 1 年 + 震后 2~3 年 |
| **核心方法** | 时间序列 seasonal_decompose 去季节性 + 韧性趋势拟合 |
| **研究对象** | M6.5+ 地震，18批次周度数据 |
| **输出成果** | 5 个交互式 Folium 地图（`.html`）、韧性趋势曲线、年际 NTL 对比图 |

**核心脚本：**
- `GEE_Uploads/EURm/EURm.py` — 批量地震韧性分析
- `GEE_Uploads/EURm/single.py` — 单城市精细分析

---

### 2. 飓风韧性分析

对全球 Cat3+ 热带气旋（城市人口 50 万+）进行 NTL 时间序列冲击分析。

| 特性 | 说明 |
|------|------|
| **数据规模** | 136 批次城市飓风周度 NTL 数据 |
| **核心算法** | DBSCAN 聚类 — 识别受影响城市范围 |
| **输出成果** | 4 个交互式飓风影响地图（City-Centric / Report / Final / Resilience） |

**核心脚本：**
- `GEE_Uploads/hurricane/hurricane.py`
- `GEE_Uploads/hurricane2/webpage.py`（扩展版，136批次）

---

### 3. 多指标遥感韧性分析

将 NTL 数据与其他遥感指标融合，从多维度刻画城市韧性：

| 指标 | 数据源 | 核心脚本 | 物理含义 |
|------|--------|----------|----------|
| **LST** 地表温度 | MODIS Terra/Aqua | `LST/lst.py` | 城市热环境活动水平 |
| **NO₂** | Sentinel-5P TROPOMI | `NO2/no2.py` | 城市工业/交通经济活动 |

---

### 4. 遥感 → 经济指标预测（ML Pipeline）

利用夜间灯光遥感数据，通过机器学习方法预测城市/国家经济指标。

**rs-fi 项目（NTL → GDP）：**

| 步骤 | 目录 | 说明 |
|------|------|------|
| 数据准备 | `02_Scripts/A_Data_Prep/` | NTL 数据预处理、特征工程 |
| 模型训练 | `02_Scripts/B_Training/` | XGBoost vs. 随机森林对比训练 |
| 模型验证 | `02_Scripts/C_Validation/` | 全球/季度/YOY/COVID/土耳其地震场景验证 |
| 应用推断 | `02_Scripts/D_Application/` | 偏差校正、GDP 估算输出 |
| 最终模型 | `03_Models/` | `best_model.pkl` + `bias_corrector.pkl` |

**rs_IPI 项目（NTL → 工业增加值）：** 同上完整流程，目标变量替换为工业增加值（IPI）。

---

## 📊 数据集概况

| 数据类型 | 来源 | 规模 | 说明 |
|----------|------|------|------|
| NTL 周度（地震） | NPP/VIIRS via GEE | 18 批次 | M6.5+，空间均值 |
| NTL 周度（飓风） | NPP/VIIRS via GEE | 97+39=136 批次 | Cat3+，人口 50 万+ |
| NTL 月度（全球M7+） | NPP/VIIRS via GEE | 多批次 | 月度汇总 |
| NTL 年际（中国地震） | DMSP/OLS + VIIRS | 31 批次，1992-2022 | 年际变化分析 |
| LST 周度 | MODIS via GEE | 18 批次 | 地震城市分析 |
| NO₂ 周度 | TROPOMI via GEE | 10 批次 | M6.5+ 城市 |
| GDP 训练集 | 世界银行 | 全球国家级 | 含人均 GDP、工业增加值 |
| 地震事件 | USGS | M6.5+ | 含城市人口筛选 |

---

## 🛠️ 技术栈

| 工具/库 | 用途 |
|---------|------|
| **Google Earth Engine（GEE）** | 遥感数据提取与导出 |
| **Python** | 核心分析与建模语言 |
| **XGBoost / scikit-learn** | 机器学习（GDP 预测） |
| **statsmodels** | 时间序列分解（seasonal_decompose） |
| **Folium / Plotly** | 交互式地图与图表可视化 |
| **DBSCAN（sklearn）** | 飓风影响区域聚类 |
| **Pandas / NumPy** | 数据处理 |
| **Jupyter Notebook** | 探索性分析 |

---

## 🚀 快速开始

### 环境依赖

```bash
pip install pandas numpy matplotlib seaborn folium plotly
pip install scikit-learn xgboost statsmodels
pip install earthengine-api  # Google Earth Engine Python API
```

### 运行地震韧性分析

```bash
# 批量 EURm 地震韧性分析
python GEE_Uploads/EURm/EURm.py

# 单城市分析
python GEE_Uploads/EURm/single.py
```

### 运行飓风韧性分析

```bash
# 初版飓风分析
python GEE_Uploads/hurricane/hurricane.py

# 扩展版（136批次）
python GEE_Uploads/hurricane2/webpage.py
```

### 运行 GDP 预测流程

```bash
# Step 1: 数据准备
python GEE_Uploads/rs-fi/02_Scripts/A_Data_Prep/...

# Step 2: 模型训练（XGBoost vs RF 对比）
python GEE_Uploads/rs-fi/02_Scripts/B_Training/compare_train.py

# Step 3: 验证
python GEE_Uploads/rs-fi/02_Scripts/C_Validation/...
```

---

## 🧩 核心代码文件详解



### `GEE_Uploads/EURm/EURm.py` — 批量地震韧性分析

**功能概述**  
批量处理多个城市的 M6.5+ 地震 NTL 时间序列数据，为每次地震生成韧性趋势图，并汇总到一张交互式 Folium 全球地图。

**处理流程**

```
读取 M65/ 目录下所有 CSV 文件
  → 合并去重（按 location_name + date）
  → 遍历每条地震记录：
      ① 解析 NTL_YYYY_WW 格式列 → 构建周度时间序列
      ② 线性插值补全缺失值
      ③ EWM 指数平滑（alpha=0.7）去除短期噪声
      ④ seasonal_decompose（加法模型，周期=52周）提取趋势分量
      ⑤ 取震前最后一个趋势值作为"震前基准线"
      ⑥ 绘图：原始平滑线（灰）+ 趋势线（黑）+ 地震线（红）+ 基准线（绿）
      ⑦ 图像编码为 Base64，嵌入 Folium Popup 弹窗
  → 在地图上按震级大小绘制圆形标记（radius = magnitude × 1.5）
  → 保存为 HTML 交互地图
```

**输入**
- `M65/*.csv`：GEE 导出的地震城市 NTL 周度均值表，列格式为 `NTL_YYYY_WW`，必须包含 `location_name`、`date`、`magnitude`、`.geo` (GeoJSON 坐标) 列
- 最低数据要求：每条记录至少有 **104 个有效周**（约 2 年）才会被处理

**输出**
- `urban_earthquake_resilience_final.html`：全球可交互地图，点击标记可查看城市韧性趋势图

**关键配置**

```python
csv_directory_path = r"D:\虚拟c盘\GEE_Uploads\EURm\M65"  # ← 改为你的数据路径！！！！！！！！！
```

**扩展建议**
- 可在趋势线之上额外计算 EURm 指数（冲击幅度 + 恢复速率），目前代码只输出可视化，定量指标尚未存档
- 可将 10% 以内的 NTL 缺失值设为"不可用"，避免插值引入的假趋势

---

### `GEE_Uploads/EURm/single.py` — 单城市精细分析

**功能概述**  
从同一份 CSV 数据库中按名称检索指定地震城市，生成高分辨率（DPI 150）PNG 图像，适合用于论文插图。

**与 `EURm.py` 的区别**

| 对比项 | `EURm.py` | `single.py` |
|--------|-----------|-------------|
| 处理对象 | 全部城市（批量） | 单个指定城市 |
| 输出格式 | Base64 嵌入 HTML | 独立 PNG 文件 |
| 分辨率 | DPI 120 | DPI 150（更适合印刷） |
| 用途 | 展示/交互 | 论文图表 |

**使用方法**

```python
# 修改脚本底部三个变量：
csv_directory_path = r"D:\...\EURm\M65"    # CSV 文件夹路径
target_earthquake_name = "168 km SW of Mawu, China"  # 精确匹配 location_name 字段
output_image_filename = "Mawu_China_earthquake_analysis.png"
```

**注意**：`target_earthquake_name` 必须与 CSV 中 `location_name` 列完全一致（含大小写和空格）

---

### `GEE_Uploads/hurricane/hurricane.py` — 飓风冲击区域聚类分析

**功能概述**  
基于 **DBSCAN 空间聚类**，将地理位置相近的多次飓风事件归为"同一受灾地区"，为每个地区生成包含多次冲击标注的综合时间序列图。

**核心算法设计**

```
步骤1：解析每条飓风记录的 .geo 起始点坐标（经纬度）
步骤2：DBSCAN 聚类
         eps = 5.0°（≈555 km），min_samples = 2
         使用 haversine 球面距离
         → 噪声点（label=-1）被过滤掉（单次孤立飓风不分析）
步骤3：按簇分组，选最强飓风（max_category 最大）作为"代表性事件"
步骤4：以代表性事件的 NTL 时间序列绘制背景趋势线
        + 在图上标注该簇内所有飓风事件的日期（红色=代表性，蓝色=其他）
步骤5：计算簇的地理中心，在地图上放置一个 Marker（红色云图标）
```

**设计意图**  
DBSCAN 聚类的意图是识别"频繁受飓风影响的地区"，而非孤立分析每次飓风，从而能够研究多次冲击对同一城市的累积效应。

**关键参数**

```python
CLUSTER_EPSILON = 5.0       # 聚类半径（度），可调大以合并更远的事件
CLUSTER_MIN_SAMPLES = 2     # 最小样本数，设为 1 则所有事件都被处理
DATA_DIRECTORY = r"..."     # 改为你的 urban_hurricane_*.csv 文件夹路径
```

**输入 CSV 格式**  
文件名需匹配 `urban_hurricane_*.csv`，必须包含字段：`.geo`（起始点坐标 GeoJSON）、`sid`（风暴 ID）、`name`、`start_date`、`max_category`、以及 `NTL_YYYY_WW` 时间序列列

**输出**
- `City_Centric_Hurricane_Impact_Map.html`

---

### `GEE_Uploads/hurricane2/webpage.py` — 扩展版飓风个例分析

**功能概述**  
对单个"飓风-城市"配对事件进行分析，同时展示 NTL 趋势线与飓风实时状态（距离 + 等级）的双轴图。是 `hurricane.py` 的"个例精细版"。

**双轴可视化设计**
- **左 Y 轴（黑色）**：NTL 趋势值，反映城市灯光变化
- **右 Y 轴（蓝色，反转）**：飓风中心距城市距离（km，越小越近）
- **散点颜色**：飓风萨菲尔-辛普森等级（viridis_r 色阶，深色=强）
- **散点大小**：随等级增大（`H_cat × 40 + 20`）

**输入 CSV 格式**  
每行一个飓风-城市事件对，时间序列列格式为 `D_YYYY_WW_NTL`、`D_YYYY_WW_H_dist`、`D_YYYY_WW_H_cat`，元数据包含 `event_id`、`city_name`、`h_name`、`h_impact_time`、`h_max_cat`、`h_min_dist_km`

**运行**

```python
csv_directory_path = r"data2"  # 改为含 CSV 的文件夹（相对或绝对路径）
```

**输出**
- `hurricane_impact_resilience_map.html`（大文件，约 57MB）

---

### `GEE_Uploads/LST/lst.py` — 地表温度韧性分析

**功能概述**  
将 NTL 韧性分析框架复用到 MODIS LST（地表温度）数据上，从"热环境"角度评估地震对城市的影响。

**与 `EURm.py` 的主要差异**

| 对比项 | `EURm.py` (NTL) | `lst.py` (LST) |
|--------|-----------------|----------------|
| 列名格式 | `NTL_YYYY_WW` | `LST_YYYY_WW` |
| 无效值判断 | `value > 0` | `value > -999`（GEE 无效值为 -999） |
| 平滑方法 | `ewm(alpha=0.7)` | `ewm(span=8)`（平滑力度更强） |
| 插值方法 | `linear` | `time`（考虑时间间隔，对 LST 更合理） |
| Y 轴单位 | 夜光辐射值 | °C（地表温度） |
| 标记颜色 | 红/深红 | 紫色（便于在合并地图时区分） |

**扩展建议**  
当前脚本未校正 LST 数据中的云遮挡缺口（GEE 导出时已用 QA 波段过滤，但部分地区仍有系统性数据空缺），后续可引入更长时间的 `fillna` 窗口或 Kriging 插值补全。

---

### `GEE_Uploads/NO2/no2.py` — NO₂ 浓度韧性分析

**功能概述**  
将同一框架应用于 Sentinel-5P TROPOMI NO₂ 数据，NO₂ 浓度可作为城市**工业与交通经济活动**的代理指标，与 NTL、LST 共同构成多指标韧性体系。

**与 `lst.py` 的主要差异**

| 对比项 | `lst.py` | `no2.py` |
|--------|----------|----------|
| 列名格式 | `LST_YYYY_WW` | `NO2_YYYY_WW` |
| 无效值判断 | `value > -999` | `value > 0` |
| Y 轴单位 | °C | mol/m² |
| 标记颜色 | 紫色 | 深蓝/蓝色 |

**当前限制**  
文件名含 `_uncalibrated` 表示尚未对 NO₂ 数据进行大气校正，后续需考虑对流层/平流层 NO₂ 分离及倾斜柱校正，以提高数据的物理可解释性。

---

### `GEE_Uploads/rs-fi/02_Scripts/B_Training/compare_train.py` — 多模型GDP预测对比训练

**功能概述**  
对比 **5 种机器学习算法**（随机森林、XGBoost、LightGBM、SVR、MLP）在 NTL → GDP 预测任务上的性能，自动保存最优模型。

**特征工程**

| 特征 | 来源 | 变换 |
|------|------|------|
| `log_ntl` | NTL_sum 或 NTL_mean | `log1p` 对数化 |
| `NDVI_mean` | Landsat NDVI | 原值（可选 log） |
| `Precip_mean` | 降雨量辅助特征 | 原值 |
| **目标变量** <br>`log_gdp` | 世界银行 GDP | `log1p` 对数化 |

**模型对比方案**

```python
models = {
    "Random Forest":  RandomForestRegressor(n_estimators=200),
    "XGBoost":        XGBRegressor(n_estimators=500, lr=0.05, max_depth=6),
    "LightGBM":       LGBMRegressor(n_estimators=500, lr=0.05),
    "SVR":            Pipeline([StandardScaler(), SVR(kernel='rbf')]),
    "MLP":            Pipeline([StandardScaler(), MLPRegressor(hidden=(100,50))])
}
```
> SVR 和 MLP 对特征尺度敏感，因此通过 `make_pipeline` 自动接入 `StandardScaler`

**评估指标**：R²、RMSE、MAE  
**数据集划分**：80% 训练 / 20% 测试，`random_state=42`

**输出**
- `03_Models/best_model.pkl`：R² 最高的模型（joblib 序列化）
- `model_comparison_results.png`：R² 柱状图 + 各模型真实值 vs 预测值散点图（6 子图）

**扩展建议**
- 当前特征仅 3 列，可扩展加入：城市面积、人口密度、GDP 增速滞后项等
- 可加入 `GridSearchCV` 或 `Optuna` 超参数搜索以进一步提升最优模型性能
- CO₂/NO₂、LST 等指标可作为额外输入特征，与 GDP 关联性值得验证

---

## 📚 核心参考文献

1. **Zhang et al. (2024)** — 利用情境调整 NTL 数据进行全球城市韧性社会经济解释（EURm 指标），*Science China Earth Sciences*
2. **Jean et al. (2016)** — 结合卫星图像与机器学习预测贫困，*Science*
3. **Chen et al. (2022)** — 1992–2019 全球 1km 网格化 GDP 与用电量数据集，*Scientific Data*
4. **Nordhaus & Chen (2011)** — 利用灯光数据作为 GDP 代理指标，*PNAS*
5. **Kyba et al. (2023)** — 全球夜空亮度变化的公民科学观测，*Science*

> 完整文献列表见 `NTL/夜光遥感总结报告与研究方案.md` 与 `论文汇总.docx`

---

*最后更新：2026年3月21日*
