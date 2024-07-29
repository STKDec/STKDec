import setproctitle
from torch_geometric.nn import GCNConv
import numpy as np
import json
import os
from tqdm import trange
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from collections import Counter

#########################################GCN网络###################################################
class Net(torch.nn.Module):
    def __init__(self, hidden_gcn=128, hidden_mlp=[128], num_node_features=4,output_dim=4):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_gcn)
        self.conv2 = GCNConv(hidden_gcn, num_node_features)

        self.linears = nn.ModuleList([])
        self.hidden_mlp = [num_node_features] + hidden_mlp
        for ii in range(len(hidden_mlp)):
            self.linears.append(nn.Linear(self.hidden_mlp[ii], self.hidden_mlp[ii + 1]))
            self.linears.append(nn.LayerNorm(self.hidden_mlp[ii + 1]))
        self.linears.append(nn.Linear(self.hidden_mlp[-1], output_dim))
        self.linears.append(nn.Softmax(dim=1))

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)  # convolution 1
        x = F.sigmoid(x)
        y = F.dropout(x, training=self.training)
        y = self.conv2(y, edge_index)  # convolution 2
        return F.relu(y)
        #return torch.cat((x, F.relu(y)), 1)


    def node_classify(self, x):
        for linear in self.linears:
            x = linear(x)
        return x
####################################聚类损失函数############################################################
def criterion_tc_bj_major(node_embeddings, cluster_labels, cluster_centers):
    # 初始化损失值为零
    l = torch.zeros(1, dtype=torch.float, device=node_embeddings.device)
    num_nodes = len(node_embeddings)

    # 遍历每个节点嵌入向量
    for i in range(num_nodes):
        # 获取节点的聚类标签
        node_label = cluster_labels[i]
        # 获取相应聚类中心的嵌入向量
        cluster_center = cluster_centers[node_label]
        # 获取节点的嵌入向量
        node_embedding = node_embeddings[i]

        # 计算节点与聚类中心的余弦相似度
        cos_sim = F.cosine_similarity(node_embedding.unsqueeze(0), cluster_center.unsqueeze(0), dim=1)
        # 累加负对数似然损失
        l = l - torch.log(cos_sim + 1e-9)
    return l / num_nodes

'''def criterion_tc_bj_major(node_embeddings, cluster_labels, cluster_centers, margin=1.0):
    # 初始化损失值为零
    l = torch.zeros(1, dtype=torch.float, device=node_embeddings.device)
    num_nodes = len(node_embeddings)

    # 遍历每个节点嵌入向量
    for i in range(num_nodes):
        # 获取节点的聚类标签
        node_label = cluster_labels[i]
        # 获取相应聚类中心的嵌入向量
        cluster_center = cluster_centers[node_label]
        # 获取节点的嵌入向量
        node_embedding = node_embeddings[i]

        # 计算节点与聚类中心的余弦相似度
        cos_sim = F.cosine_similarity(node_embedding.unsqueeze(0), cluster_center.unsqueeze(0), dim=1)
        # 累加负对数似然损失
        l = l - torch.log(cos_sim + 1e-9)

        # 对不同组的节点，增加一个对比损失
        for j in range(num_nodes):
            if i != j and cluster_labels[i] != cluster_labels[j]:
                other_node_embedding = node_embeddings[j]
                cos_sim_diff = F.cosine_similarity(node_embedding.unsqueeze(0), other_node_embedding.unsqueeze(0),
                                                   dim=1)
                l = l + F.relu(margin - cos_sim_diff)

    return l / num_nodes'''
#######################################图结构数据#############################################################
# 加载站点向量表示
station_features = np.load('data/chengdu/16kg/id2emb.npy')

# 加载站点边索引和权重
with open('data/chengdu/16kg/entity2idx.json', 'r') as f:
    entity2idx = json.load(f)

# 先过滤出前缀为 'station' 的键，然后对它们进行排序
sorted_station_keys = sorted([key for key in entity2idx.keys() if key.startswith("station")], key=lambda x: int(x.replace('station', '')))

# 根据排序后的键生成 station_indices 字典
station_indices = {key: entity2idx[key]['idx'] for key in sorted_station_keys}
#print('station的顺序',station_indices)

# 加载站点的距离矩阵
distances_matrix = np.loadtxt('data/chengdu/road_edge.txt')

# 确保站点数量与距离矩阵一致
num_stations = len(station_indices)
assert distances_matrix.shape == (num_stations, num_stations), "距离矩阵的尺寸与站点数量不一致"

# 创建 PyTorch Geometric 数据对象
station_vectors = []
edge_index = []
edge_attr = []

for i, (station_name, idx) in enumerate(station_indices.items()):
    # 获取站点向量表示
    vector = station_features[idx]
    station_vectors.append(vector)

    for j in range(num_stations):
        if i != j:  # 排除自身环
            # 构建边索引
            edge_index.append([i, j])
            # 获取该站点到其他站点的距离
            edge_attr.append(distances_matrix[i, j])
#print('点的顺序',station_vectors)

# 将列表转换为 numpy 数组
station_vectors = np.array(station_vectors)
edge_index = np.array(edge_index).T  # 转置以符合 PyTorch Geometric 的格式
edge_attr = np.array(edge_attr)

np.savetxt('station_vectors.txt', station_vectors)
# 创建 PyTorch Geometric 数据对象
#station_vectors是点,edge_index是边,edge_attr是权重，是一个有向图。
#您有 54 个站点（节点），它们的特征向量维度是 4，共有 2862 条边及其对应的权重。
data_bj = Data(
    x=torch.tensor(station_vectors, dtype=torch.float),
    edge_index=torch.tensor(edge_index, dtype=torch.long),
    edge_attr=torch.tensor(edge_attr, dtype=torch.float)
)
##################################初始化聚类表示#################################################################
#根据距离将station聚类
#labels_distance=[6, 3, 3, 3, 3, 6, 3, 0, 6, 2, 3, 0, 3, 5, 6, 0, 2, 0, 7, 5, 0, 0, 4, 2, 7, 7, 2, 6, 0, 6, 2, 6, 6, 3, 1, 2, 6, 0, 0, 6, 6, 6, 6, 0, 2, 0, 5, 0, 7, 7, 7, 3, 3, 0]
#center_distance=[53, 34, 23, 33, 22, 19, 40, 24]
#labels_distance=[3, 3, 0, 0, 0, 0, 3, 3, 2, 3, 3, 1, 3, 2, 0, 0, 3, 1, 3, 0, 2, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 0, 0, 0, 0, 2, 2, 3, 1, 0, 2, 0, 3, 3, 3, 3, 2]
#center_distance=[5, 17, 8, 33]
labels_distance=[0, 0, 2, 2, 2, 1, 0, 0, 1, 3, 0, 1, 0, 1, 1, 0, 3, 1, 0, 0, 0, 0, 3, 3, 0, 3, 3, 1, 0, 0, 3, 0, 0, 0, 3, 3, 1, 0, 0, 1, 1, 1, 1, 0, 3, 1, 0, 0, 1, 0, 3, 0, 1, 0]
center_distance=[29, 5, 4, 50]
################################训练过程###############################################################

setproctitle.setproctitle("node_trans_"+"@luhan")#("node_trans_all@hsd")
torch.manual_seed(5)
patience_loss = 250
save_dir = '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
device = torch.device('cpu')
model, data = Net(num_node_features=4).to(
    device), data_bj.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
B_nc, B_lp, B_tc = 1e2, 1, 1  # 1, 1e-6, 1e-3
criterion_nc = nn.KLDivLoss()

loss_best = 1e9
loss_best_it = 0
loss_list = []

loss_tc_list = []
loss_nc_list = []

for epoch in trange(1, 1001):
    model.train()
    optimizer.zero_grad()  # Clear gradients.

    # 初始化一个54x8的零矩阵
    labels_matrix = np.zeros((54, 4), dtype=int)
    # 遍历标签距离列表，并在矩阵相应位置设置为1
    for i, label in enumerate(labels_distance):
        labels_matrix[i, label] = 1

    nc_labels = torch.tensor(labels_matrix, dtype=torch.float).to(device)
    # 使用模型进行节点分类预测
    node_embedding = model.encode(data.x, data.edge_index)
    #print('node_embedding.shape',node_embedding.shape)
    #输入的是节点的向量表示
    nc_results = model.node_classify(node_embedding)
    #print('nc_results.shape',nc_results.shape)
    # 计算节点分类损失
    loss_nc = criterion_nc(torch.log(nc_results + 1e-9), nc_labels)

    cluster_labels = torch.tensor(labels_distance, dtype=torch.long)
    # 获取中心节点的嵌入向量
    center_indices = torch.tensor(center_distance, dtype=torch.long)
    cluster_centers = node_embedding[center_indices]
    #计算loss
    loss_tc = criterion_tc_bj_major(node_embedding, cluster_labels, cluster_centers)
    loss = loss_nc + loss_tc

    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    scheduler.step(loss)
    if epoch % 10 == 0:
        #print(optimizer)
        print(
        f'Epoch: {epoch:05d}, Loss: {loss.item():.10f}, Loss_nc: {loss_nc.item():.10f}, Loss_tc: {loss_tc.item():.10f}')

############################################保存数据######################################################################
    if loss_best > loss:
        loss_best = loss
        loss_best_it = epoch
        node_embedding_np = node_embedding.cpu().detach().numpy()
        #torch.save(model.state_dict(), os.path.join(save_dir, 'Net'+ str('chengdu')))
    elif epoch - loss_best_it > patience_loss:
        break

    loss_list.append(loss.cpu().detach().numpy())
    loss_nc_list.append(loss_nc.cpu().detach().numpy())
    loss_tc_list.append(loss_tc.cpu().detach().numpy())

#您有 54 个站点（节点），它们的特征向量维度是 4

# 将 node_embedding_np 保存到 txt 文件
np.savetxt('node_embedding.txt', node_embedding_np)
# 将 loss_list 保存到 txt 文件
np.savetxt('loss_list.txt', np.array(loss_list))
# 将 loss_tc_list 保存到 txt 文件
np.savetxt('loss_tc_list.txt', np.array(loss_tc_list))
# 将 loss_nc_list 保存到 txt 文件
np.savetxt('loss_nc_list.txt', np.array(loss_nc_list))
# 将 loss_best_it 保存到 txt 文件
with open('epoch_num.txt', 'w') as f:
    f.write(str(loss_best_it))


#np.savez(os.path.join(save_dir,
#                      'node_embedding_KL_log_layerscat_d' +  str('chengdu')),
   #      node_embedding=node_embedding_np,
    #     loss_list = np.array(loss_list),
     #    loss_tc_list = np.array(loss_tc_list),
      #   loss_nc_list = np.array(loss_nc_list),
      #   epoch_num = loss_best_it)

