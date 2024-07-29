#########################评估算法####################################

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

import gzip
###########################预测的数据#####################################
df2 = pd.read_csv(r'D:\研究生\生成式AI\wenhui\14.跑实验2_一个月\上海市\上海市station_换电数据_flow_2_del_normalized.csv')
max_value_lu = max(df2['flow'])
min_value_lu = min(df2['flow'])

# 将日期时间列转换为日期时间对象
df2['hour'] = pd.to_datetime(df2['hour'])

# 提取日期、站点和小时信息
df2['date'] = df2['hour'].dt.date
df2['hour_of_day'] = df2['hour'].dt.hour

# 获取唯一日期、站点和小时列表
dates2 = df2['date'].unique()#[:7]
stations2 = df2['name'].unique()
hours2 = np.arange(24)  # 0到23
pred_data = np.zeros((len(dates2), 64, 64))
# 填充数据到三维数组
for i, date in enumerate(dates2):
    for j, station in enumerate(stations2):
        day_data2 = df2[(df2['date'] == date) & (df2['name'] == station)].sort_values(by='hour_of_day')['average_value']
        if j < 64:
            pred_data[i, j, :24] = day_data2*(max_value_lu-min_value_lu)+min_value_lu
pred_data = pred_data[7:14, :49, :24]

###########################真实的数据#####################################

# 读取 CSV 文件
df = pd.read_csv(r'D:\研究生\生成式AI\wenhui\14.跑实验2_一个月\上海市\上海市station_换电数据_flow_2_del_normalized.csv')
max_value_lu_2 = max(df['flow'])
min_value_lu_2 = min(df['flow'])
# 将日期时间列转换为日期时间对象
df['hour'] = pd.to_datetime(df['hour'])

# 提取日期、站点和小时信息
df['date'] = df['hour'].dt.date
df['hour_of_day'] = df['hour'].dt.hour

# 获取唯一日期、站点和小时列表
dates = df['date'].unique()#[:7]
stations = df['name'].unique()
hours = np.arange(24)  # 0到23
reshaped_data = np.zeros((len(dates), 64, 64))
# 填充数据到三维数组
for i, date in enumerate(dates):
    for j, station in enumerate(stations):
        day_data = df[(df['date'] == date) & (df['name'] == station)].sort_values(by='hour_of_day')['Normalized Data']
        if j < 64:
            reshaped_data[i, j, :24] = day_data*(max_value_lu-min_value_lu)+min_value_lu
reshaped_data = reshaped_data[7:14, :49, :24]

##########################reshape和pred的数据在一个范围里########################
def nth_largest(array, n):
    flattened_array = array.flatten()
    sorted_array = np.sort(flattened_array)
    if len(sorted_array) >= n:
        return sorted_array[-n]
    else:
        raise ValueError(f"Array does not have {n} elements")


# 定义一个函数来找到第n小的数
def nth_smallest(array, n):
    flattened_array = array.flatten()
    sorted_array = np.sort(flattened_array)
    if len(sorted_array) >= n:
        return sorted_array[n - 1]
    else:
        raise ValueError(f"Array does not have {n} elements")

# 处理前24个换电站的数据
for station in range(len(reshaped_data[0])):
    for day in range(len(reshaped_data)):
        real = reshaped_data[day, station, :]

        pred_min = np.min(pred_data[day, station, :])
        pred_max = np.max(pred_data[day, station, :])
        real_min = nth_smallest(real, 5)
        real_max = nth_largest(real, 5)
        # 归一化 pred 到 real 的范围
        pred_data[day, station, :] = (pred_data[day, station, :] - pred_min) / (pred_max - pred_min)
        pred_data[day, station, :] = pred_data[day, station, :] * (real_max - real_min) + real_min
#########################mae和rmse##################################


yuanshi_data_flat = reshaped_data.flatten()
pred_data_flat = pred_data.flatten()


mae = mean_absolute_error(yuanshi_data_flat, pred_data_flat)
rmse = np.sqrt(mean_squared_error(yuanshi_data_flat, pred_data_flat))
print("MAE Value:", mae)
print("RMSE Value:", rmse)
# 计算 MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0  # 创建一个掩码，标记所有非零值的位置
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
mape = mean_absolute_percentage_error(yuanshi_data_flat, pred_data_flat)
print("MAPE Value:", mape)

################################mmd##################################
import numpy as np

class MaximumMeanDiscrepancy_numpy(object):
    """calculate MMD"""

    def __init__(self):
        super(MaximumMeanDiscrepancy_numpy, self).__init__()

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = source.shape[0] + target.shape[0]
        total = np.concatenate([source, target], axis=0)  # 合并在一起

        total0 = np.expand_dims(total, axis=0)
        total0 = np.tile(total0, (total.shape[0], 1, 1, 1))

        total1 = np.expand_dims(total, axis=1)
        total1 = np.tile(total1, (1, total.shape[0], 1, 1))


        L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

        # 计算多核中每个核的bandwidth
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = np.sum(L2_distance) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        # 高斯核的公式，exp(-|x-y|/bandwith)
        kernel_val = [np.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

        return sum(kernel_val)  # 将多个核合并在一起

    def __call__(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n = source.shape[0]
        m = target.shape[0]

        kernels = self.guassian_kernel(
            source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:n, :n]
        YY = kernels[n:, n:]
        XY = kernels[:n, n:]
        YX = kernels[n:, :n]

        # K_ss矩阵，Source<->Source
        XX = np.divide(XX, n * n).sum(axis=1).reshape(1, -1)
        # K_st矩阵，Source<->Target
        XY = np.divide(XY, -n * m).sum(axis=1).reshape(1, -1)
        # K_ts矩阵,Target<->Source
        YX = np.divide(YX, -m * n).sum(axis=1).reshape(1, -1)
        # K_tt矩阵,Target<->Target
        YY = np.divide(YY, m * m).sum(axis=1).reshape(1, -1)

        loss = (XX + XY).sum() + (YX + YY).sum()
        return loss

def calculate_mmd(source_matrix, target_matrix, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    mmd_calculator = MaximumMeanDiscrepancy_numpy()
    source = np.array(source_matrix)
    target = np.array(target_matrix)
    return mmd_calculator(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

mmd_value = calculate_mmd(reshaped_data, pred_data)
print("MMD Value:", mmd_value)
##########################FoD#########################################
import numpy as np
from scipy.spatial.distance import jensenshannon

def calculate_jsd(matrix1, matrix2):
    # 将矩阵展平成一维数组
    data1 = matrix1.flatten()
    data2 = matrix2.flatten()

    # 计算两个矩阵的一维概率分布
    min_val = min(data1.min(), data2.min())
    max_val = max(data1.max(), data2.max())
    bins = np.linspace(min_val, max_val, 100)

    hist1, _ = np.histogram(data1, bins=bins, density=True)
    hist2, _ = np.histogram(data2, bins=bins, density=True)

    # 避免零概率，进行平滑处理
    hist1 = np.where(hist1 == 0, 1e-10, hist1)
    hist2 = np.where(hist2 == 0, 1e-10, hist2)

    # 计算JSD
    jsd_value = jensenshannon(hist1, hist2) ** 2

    return jsd_value
# 计算第一阶差分的JSD
jsd_value = calculate_jsd(reshaped_data, pred_data)
print("JSD Value:", jsd_value)

##########################余弦相似度#########################################
# import numpy as np
# from scipy.spatial.distance import cosine
# from scipy.stats import entropy
#
# def compare_matrices(reshaped_data, pred_data):
#     # 将矩阵展平
#     reshaped_vector = reshaped_data.flatten()
#     pred_vector = pred_data.flatten()
#
#     # 计算余弦相似度
#     cos_sim = 1 - cosine(reshaped_vector, pred_vector)
#
#
#     # 避免 log(0) 的情况
#     epsilon = 1e-12
#     reshaped_data = reshaped_data + epsilon
#     pred_data = pred_data + epsilon
#
#     # 计算交叉熵
#     cross_entropy = entropy(reshaped_data.flatten(), pred_data.flatten())
#
#     return cos_sim, cross_entropy
# # 比较矩阵
# cos_sim, cross_entropy = compare_matrices(reshaped_data, pred_data)
#
# # 打印结果
# print('Cosine Similarity（余弦）:', cos_sim)
# print('Cross Entropy(交叉）:', cross_entropy)
