import torch
import torch.nn as nn
import pandas as pd

class TimePointEmbedding(nn.Module):
    def __init__(self, day_of_week_dim, season_dim, is_holiday_dim, station_id_dim, embed_dim):
        super(TimePointEmbedding, self).__init__()
        self.day_of_week_embedding = nn.Linear(day_of_week_dim, embed_dim)
        self.season_embedding = nn.Linear(season_dim, embed_dim)
        self.is_holiday_embedding = nn.Linear(is_holiday_dim, embed_dim)
        self.station_id_embedding = nn.Linear(station_id_dim, embed_dim)
        self.output_dim = embed_dim * 4  # 4个特征的嵌入向量连接起来

    def forward(self, x):
        day_of_week = self.day_of_week_embedding(x[:, 0:1])
        season = self.season_embedding(x[:, 1:2])
        is_holiday = self.is_holiday_embedding(x[:, 2:3])
        station_id = self.station_id_embedding(x[:, 3:4])
        combined_embedding = torch.cat((day_of_week, season, is_holiday, station_id), dim=-1)
        return combined_embedding


# 示例输入特征，形状为 (batch_size, 4)
csv_file = r'E:\luhan2\condition_2024.csv'
data = pd.read_csv(csv_file)
input_data = data.iloc[:, 1:5].values  # 取第二到第五列，索引从0开始
input_features = torch.tensor(input_data, dtype=torch.float32)
# 定义嵌入层
embed_dim = 1  # 嵌入维度
model = TimePointEmbedding(day_of_week_dim=1, season_dim=1, is_holiday_dim=1, station_id_dim=1, embed_dim=embed_dim)

# 前向传播得到嵌入向量
embeddings = model(input_features)
# 保存嵌入向量到TXT文件
output_file = r'E:\luhan2\condition_2024_embeddings.txt'
with open(output_file, 'w') as f:
    for tensor in embeddings:
        f.write(' '.join([str(num.item()) for num in tensor]) + '\n')

print(embeddings.shape)
