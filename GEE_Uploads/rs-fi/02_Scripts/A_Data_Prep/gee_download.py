import ee
import time

# 初始化
try:
    ee.Initialize(project='awaran-20130924')  # 你的项目ID
except:
    ee.Authenticate()
    ee.Initialize(project='awaran-20130924')

print("🚀 开始 Task 4：提取土耳其地震前后月度数据...")

# 1. 定义区域：土耳其 (Turkey/Turkiye)
country = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
    .filter(ee.Filter.eq('country_na', 'Turkey'))

# 2. 定义时间范围 (涵盖震前和震后)
start_year = 2019
end_year = 2023


# 3. 核心函数：计算单月数据
def get_monthly_data(year, month):
    # 时间窗口
    start_date = ee.Date.fromYMD(year, month, 1)
    end_date = start_date.advance(1, 'month')

    # A. 夜光 (NASA VNP46A2 日度合成月度)
    daily_ntl = ee.ImageCollection("NASA/VIIRS/002/VNP46A2") \
        .filterDate(start_date, end_date)

    def process_ntl(img):
        qf = img.select('QF_Cloud_Mask')
        # 去云
        mask = qf.bitwiseAnd(1 << 0).eq(0).And(qf.bitwiseAnd(1 << 2).eq(0))
        return img.select('DNB_BRDF_Corrected_NTL').updateMask(mask)

    # 计算月度均值 和 总量
    # 注意：模型训练用的是 NTL_sum，所以这里必须算出 Sum
    # 但 image.reduceRegion 算 sum 比较慢，我们先算出 mean，导出后再乘以面积近似 sum
    # 或者直接在 GEE 里算 sum
    ntl_img = daily_ntl.map(process_ntl).mean().rename('NTL_mean')

    # B. 植被 (NDVI)
    ndvi_img = ee.ImageCollection("MODIS/061/MOD13Q1") \
        .filterDate(start_date, end_date) \
        .select('NDVI').mean().multiply(0.0001).rename('NDVI_mean')

    # C. 降水
    precip_img = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterDate(start_date, end_date) \
        .select('precipitation').sum().rename('Precip_mean')  # 用sum代表月降水，命名保持一致方便处理

    # 组合
    combined = ntl_img.addBands(ndvi_img).addBands(precip_img)

    # 统计区域均值 (Mean)
    # 我们导出 Mean，然后在 Python 里用 Mean * Area * 系数 来模拟 Sum
    stats = combined.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=country.geometry(),
        scale=5000,  # 5km 分辨率
        maxPixels=1e13,
        tileScale=16
    )

    # 计算 NTL Sum (近似值：均值 * 像素数 或 均值 * 面积)
    # 这里我们直接导出 Sum 比较保险，为了防止超时，单独算一次 Sum
    ntl_sum_stat = ntl_img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=country.geometry(),
        scale=5000,
        maxPixels=1e13,
        tileScale=16
    )

    return ee.Feature(None, {
        'year': year,
        'month': month,
        'date': start_date.format('YYYY-MM-dd'),
        'NTL_mean': stats.get('NTL_mean'),
        'NTL_sum': ntl_sum_stat.get('NTL_mean'),  # 注意：reduceRegion sum 的结果key还是叫原波段名
        'NDVI_mean': stats.get('NDVI_mean'),
        'Precip_mean': stats.get('Precip_mean')
    })


# 4. 循环生成列表
all_features = []
print("   -> 正在生成计算任务列表...")
for y in range(start_year, end_year + 1):
    for m in range(1, 13):
        all_features.append(get_monthly_data(y, m))

# 5. 导出
# 由于 reduceRegion 是客户端操作，不能直接 map，我们这里用 getInfo (小数据量可行)
# 如果报错超时，则需要改回 batch export。鉴于只有 12*5=60 个点，直接跑应该可以。

print("   -> 正在向 GEE 请求计算 (可能需要1-2分钟)...")
# 将 list 转为 FeatureCollection
fc = ee.FeatureCollection(all_features)

task = ee.batch.Export.table.toDrive(
    collection=fc,
    description='Turkey_Earthquake_Analysis_Monthly',
    folder='GEE_Uploads',  # 你的Drive文件夹
    fileFormat='CSV'
)
task.start()
print("✅ 任务已提交！请等待 'Turkey_Earthquake_Analysis_Monthly.csv' 生成。")