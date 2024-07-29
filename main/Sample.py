
import torch
import torch.nn as nn
import pandas as pd
# 定义每一步的 betas
num_steps = 150
betas = torch.linspace(-6, 6, num_steps)
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
import torch.nn.functional as F
import csv


def calculate_attention(query, key):
    # 计算内积
    dot_products = torch.matmul(query, key.transpose(-2, -1))

    # 计算缩放后的内积
    scaled_dot_products = dot_products / torch.sqrt(torch.tensor(key.size(-1)).float())

    # 使用softmax计算注意力权重
    attention_weights = F.softmax(scaled_dot_products, dim=-1)

    return attention_weights
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
import openpyxl
import numpy as np
data_file = r'D:\研究生\生成式AI\wenhui\14.跑实验2_一个月\北京市\北京市station_换电数据_flow_2_del_normalized.csv'
kg_file = r'D:\研究生\生成式AI\wenhui\14.跑实验2_一个月\北京市\北京知识图谱_扩展.txt'
condition_file = r'D:\研究生\生成式AI\wenhui\14.跑实验2_一个月\北京市\时空.txt'
data_lu = read_csv(data_file, 3)
kg_lu = read_txt(kg_file)
condition_lu = read_txt(condition_file)
dataset = torch.Tensor(data_lu).float()
kg = torch.Tensor(kg_lu).float()
conditions = torch.Tensor(condition_lu).float()


alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt, condition1, condition2, kg):
    t = torch.tensor([t])
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, t, condition1, condition2, kg)
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))

    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()

    sample = mean + sigma_t * z

    mean_np = mean.detach().cpu().numpy()
    sigma_t_np = sigma_t * z.detach().cpu().numpy()


    ###想写到txt就要只sample一次
    # with open(r'D:\研究生\生成式AI\wenhui\21.步骤图\0次\mean.txt', 'w') as f:
    #
    #     np.savetxt(f, mean_np, delimiter=', ', fmt='%.6f')
    #
    # with open(r'D:\研究生\生成式AI\wenhui\21.步骤图\0次\s.txt', 'w') as f:
    #
    #     np.savetxt(f, sigma_t_np, delimiter=', ', fmt='%.6f')

    return sample
def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt, condition1, condition2, kg):
    cur_x = torch.randn(shape)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt, condition1, condition2, kg)
        x_seq.append(cur_x)
    return x_seq


def save_samples_to_csv(samples, file_path):
    # Create a pandas DataFrame from the samples
    samples_df = pd.DataFrame(samples.T, columns=[f'sample_{i+1}' for i in range(samples.shape[0])])

    # Save the DataFrame to a CSV file
    samples_df.to_csv(file_path, index=False)
    print(f"Samples saved to {file_path}")
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

        self.mha = nn.MultiheadAttention(embed_dim=8, num_heads=8)  # 多头注意力机制


    def attention(self, cond1, cond2, kg):
        key = torch.cat((cond1, cond2), dim=-1).unsqueeze(0)  # 拼接并添加维度

        query = kg.unsqueeze(0)  # 添加维度

        value = torch.cat((cond1, cond2), dim=-1).unsqueeze(0)  # 拼接并添加维度

        out, weight = self.mha(query, key, value)

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

def load_model(checkpoint_path, model_class, *model_args):
    # 创建模型实例
    model = model_class(*model_args)

    # 加载保存的模型状态字典
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 创建优化器实例
    optimizer = torch.optim.Adam(model.parameters())

    # 加载保存的优化器状态字典
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

print("ok1")
# 文件路径
checkpoint_path = r"D:\研究生\生成式AI\wenhui\14.跑实验2_一个月\成都市\checkpoint (14)_step_150.pth"
print("ok1")
# 加载模型和优化器
model, optimizer = load_model(checkpoint_path, MLPDiffusion, num_steps)
print("ok2")
# 设置模型为评估模式
model.eval()
num_samples = 10
generated_samples = torch.empty((num_samples, dataset.shape[0]))
# 使用模型进行推理
for i in range(num_samples):
    print("i-----------",i)
    with torch.no_grad():
        # 输入数据
        x_0 = torch.ones_like(dataset)
        t = torch.tensor([num_steps - 1])  # 起始时间步
        condition1 = conditions[:, :4]
        condition2 = conditions[:, 4:]

        # 执行推理
        x_seq = p_sample_loop(model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt, condition1, condition2, kg)

        generated_data_point = x_seq[-1]
        generated_samples[i] = generated_data_point.flatten()

# 定义保存路径

save_path = r'D:\研究生\生成式AI\wenhui\14.跑实验2_一个月\北京市\generated_samples.csv'
save_samples_to_csv(generated_samples, save_path)
# 计算每行的平均值
average_values = generated_samples.mean(dim=0)

# 读取已保存的样本数据CSV文件
samples_df = pd.read_csv(save_path)

# 将平均值添加到DataFrame中
samples_df['average_value'] = average_values.tolist()

# 保存更新后的DataFrame到CSV文件
samples_df.to_csv(save_path, index=False)

print(f"Average values added to the CSV file: {save_path}")