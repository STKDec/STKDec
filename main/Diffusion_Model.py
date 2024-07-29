import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


import csv

#读取文件中的所有数据
def read_csv(file_path, column_index):
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过第一行（列名）
        for row in reader:
            data.append([float(row[column_index])])
    return data
def read_txt(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append([float(x) for x in line.strip().split()])
    return data
# 文件路径
data_file = r'成都市station_换电数据_flow_2_del_normalized.csv'
kg_file = r'成都知识图谱_扩展.txt'
condition_file = r'时空.txt'
data_lu = read_csv(data_file, 3)
kg_lu = read_txt(kg_file)
condition_lu = read_txt(condition_file)
dataset = torch.Tensor(data_lu).float()
kg = torch.Tensor(kg_lu).float()
conditions = torch.Tensor(condition_lu).float()

# 定义超参数
num_steps = 200
num_attention_units = 128

# 定义每一步的 betas
betas = torch.linspace(-6, 6, num_steps)
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

# 计算 alphas 及相关变量
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == \
       alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape == \
       one_minus_alphas_bar_sqrt.shape  # 确保所有列表长度一致

def q_x(x_0, t):
    """在时间 t 采样 x[t] 给定 x[0]。"""
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return alphas_t * x_0 + alphas_1_m_t * noise  # 给 x[0] 添加噪声

# 定义 MLP Diffusion 模型
class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, num_units=128):
        super(MLPDiffusion, self).__init__()
        self.num_units = num_units

        self.linears = nn.ModuleList(
            [
                nn.Linear(9, num_units),  # 包含条件在输入中
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, num_units),
                nn.ReLU(),
                nn.Linear(num_units, 1),
            ]
        )
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
                nn.Embedding(n_steps, num_units),
            ]
        )

        self.key_layer = nn.Linear(8, num_units)
        self.query_layer = nn.Linear(8, num_units)
        self.value_layer = nn.Linear(8, num_units)
        self.softmax = nn.Softmax(dim=1)

        self.mha = nn.MultiheadAttention(embed_dim=8, num_heads=1)  # 多头注意力机制

    def attention(self, cond1, cond2, kg):
        key = torch.cat((cond1, cond2), dim=-1).unsqueeze(0)  # 拼接并添加维度

        query = kg.unsqueeze(0)  # 添加维度

        value = torch.cat((cond1, cond2), dim=-1).unsqueeze(0)  # 拼接并添加维度

        out, weight = self.mha(query, key, value)

        # 保存权重到 txt 文件，每次覆盖内容
        #weight_2d = weight.mean(dim=1).squeeze(0).detach().cpu().numpy()
       # np.savetxt("attention_weights.txt", weight_2d)


        return out.squeeze(0)  # 移除添加的维度

    def forward(self, x, t, condition1, condition2, kg):

        attention_output = self.attention(condition1, condition2, kg)
        condition_combined = attention_output


        x = torch.cat([x, condition_combined], dim=1)  # 将条件连接到输入中


        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding = embedding_layer(t)
            x = self.linears[2 * idx](x)
            x += t_embedding
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-1](x)
        return x


def diffusion_loss_fn(model, x_0, condition1, condition2, kg, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    t = torch.randint(0, n_steps, size=(x_0.shape[0]//2,))
    t = torch.cat([t, n_steps - 1 - t], dim=0)
    t = t.unsqueeze(-1)
    a = alphas_bar_sqrt[t]
    aml = one_minus_alphas_bar_sqrt[t]
    e = torch.randn_like(x_0)

    x = x_0 * a + e * aml
    output = model(x, t.squeeze(-1), condition1, condition2, kg)
    return (e - output).square().mean()

# 定义采样函数
def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt, condition1, condition2, kg):
    t = torch.tensor([t])
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, t, condition1, condition2, kg)
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return sample

def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt, condition1, condition2, kg):
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt, condition1, condition2, kg)
        x_seq.append(cur_x)
    return x_seq

# 训练过程
print('Training model...')
batch_size = 256
dataloader = torch.utils.data.DataLoader(list(zip(dataset, conditions[:, :4], conditions[:, 4:], kg)), batch_size=batch_size,shuffle=True)
num_epoch = 8000

model = MLPDiffusion(num_steps)  # 输出维度为1，输入是 x、步数和条件
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epoch):

    for batch_x, batch_condition1, batch_condition2, batch_kg in dataloader:


        loss = diffusion_loss_fn(model, batch_x, batch_condition1, batch_condition2, batch_kg, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# 定义保存路径
save_path = 'checkpoint.pth'

# 创建检查点字典
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}

# 保存检查点
torch.save(checkpoint, save_path)
print(f"模型已保存到 {save_path}")
