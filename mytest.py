import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import copy
import matplotlib.pyplot as plt

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)  # 一个全连接层

    def forward(self, x):
        return self.fc(x)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 64 * 6 * 6)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 创建模型实例
# model = SimpleModel()
model = SimpleNet()

global_parameters = model.state_dict() 

# 获取模型的状态字典
# state_dict = model.state_dict()     # 字典的键是层的名称（例如，fc.weight 或 fc.bias），值是对应的参数（tensor)
# print(state_dict.keys())

# for k, v in state_dict.items():
#     # p = k.split('.')
#     # print(p)
#     p = k.split('.')[-1]
#     if p == 'bias':
#         print(p)
#         print(v.size(0))

# a = [k for k in state_dict .keys() if 'weight' in k][-1]
# print(a)

user_idx = [0,1,2,3,4]
model_rate = [0.25,0.25,0.5,0.5,1.0]

idx_i = [None for _ in range(len(user_idx))]            # [None,None...]
idx = [OrderedDict() for _ in range(len(user_idx))]     # [OrderedDict(),OrderedDict()...]
output_weight_name = [k for k in global_parameters.keys() if 'weight' in k][-1]        # glabal_parameters为字典，由model.state_dict()得到
output_bias_name = [k for k in global_parameters.keys() if 'bias' in k][-1]            # 键是层的名称（例如，fc.weight 或 fc.bias），值是对应的参数（tensor)

def split_model(user_idx):
    for k, v in global_parameters.items():
        parameter_type = k.split('.')[-1]                   # 提取出对应的类型：weight/bias
        for m in range(len(user_idx)):
            if 'weight' in parameter_type or 'bias' in parameter_type:
                if parameter_type == 'weight':
                    if v.dim() > 1:
                        input_size = v.size(1)
                        output_size = v.size(0)
                        if idx_i[m] is None:
                            idx_i[m] = torch.arange(input_size, device=v.device)
                        input_idx_i_m = idx_i[m]
                        if k == output_weight_name:
                            output_idx_i_m = torch.arange(output_size, device=v.device)
                        else:
                            scaler_rate = model_rate[user_idx[m]]
                            local_output_size = int(np.ceil(output_size * scaler_rate))
                            output_idx_i_m = torch.arange(output_size, device=v.device)[:local_output_size]
                        idx[m][k] = output_idx_i_m, input_idx_i_m
                        idx_i[m] = output_idx_i_m
                    else:
                        input_idx_i_m = idx_i[m]
                        idx[m][k] = input_idx_i_m
                else:
                    if k == output_bias_name:
                        input_idx_i_m = idx_i[m]
                        idx[m][k] = input_idx_i_m
                    else:
                        input_idx_i_m = idx_i[m]
                        idx[m][k] = input_idx_i_m
            else:
                pass
    return idx

def distribute(user_idx):
    param_idx = split_model(user_idx)
    local_parameters = [OrderedDict() for _ in range(len(user_idx))]
    count = 0
    for k, v in global_parameters.items():
        parameter_type = k.split('.')[-1]
        for m in range(len(user_idx)):
            if 'weight' in parameter_type or 'bias' in parameter_type:
                if 'weight' in parameter_type:
                    if v.dim() > 1:
                        local_parameters[m][k] = copy.deepcopy(v[torch.meshgrid(param_idx[m][k])])
                    else:
                        local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                else:
                    local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                    if(m == 0):
                        print(param_idx[m][k])
                        print(v)
                        print(local_parameters[m][k])
            else:
                local_parameters[m][k] = copy.deepcopy(v)
    return local_parameters, param_idx

def white(messages, byzantinesize):
    # 均值相同，方差较大
    mu = torch.mean(messages[0:-byzantinesize], dim=0)
    messages[-byzantinesize:].copy_(mu)
    noise = torch.randn((byzantinesize, messages.size(1)), dtype=torch.float64)
    messages[-byzantinesize:].add_(30, noise)
def maxValue(messages, byzantinesize):
    mu = torch.mean(messages[0:-byzantinesize], dim=0)
    meliciousMessage = -3*mu
    messages[-byzantinesize:].copy_(meliciousMessage)
def zeroGradient(messages, byzantinesize):
    s = torch.sum(messages[0:-byzantinesize], dim=0)
    messages[-byzantinesize:].copy_(-s / byzantinesize)

byzantinesize = 2
def combine(local_parameters, param_idx, user_idx):
    count = OrderedDict()
    output_weight_name = [k for k in global_parameters.keys() if 'weight' in k][-1]
    output_bias_name = [k for k in global_parameters.keys() if 'bias' in k][-1]
    for k, v in global_parameters.items():
        parameter_type = k.split('.')[-1]
        count[k] = v.new_zeros(v.size(), dtype=torch.float32)
        tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
        for m in range(len(local_parameters)):
            if 'weight' in parameter_type or 'bias' in parameter_type:
                if parameter_type == 'weight':
                    if v.dim() > 1:
                        if k == output_weight_name:
                            label_split = label_split[user_idx[m]]
                            param_idx[m][k] = list(param_idx[m][k])
                            param_idx[m][k][0] = param_idx[m][k][0][label_split]
                            tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k][label_split]
                            count[k][torch.meshgrid(param_idx[m][k])] += 1
                        else:
                            tmp_v[torch.meshgrid(param_idx[m][k])] += local_parameters[m][k]
                            count[k][torch.meshgrid(param_idx[m][k])] += 1
                    else:
                        tmp_v[param_idx[m][k]] += local_parameters[m][k]
                        count[k][param_idx[m][k]] += 1
                else:
                    if k == output_bias_name:
                        label_split = label_split[user_idx[m]]
                        param_idx[m][k] = param_idx[m][k][label_split]
                        tmp_v[param_idx[m][k]] += local_parameters[m][k][label_split]
                        count[k][param_idx[m][k]] += 1
                    else:
                        tmp_v[param_idx[m][k]] += local_parameters[m][k]
                        count[k][param_idx[m][k]] += 1
            else:
                tmp_v += local_parameters[m][k]
                count[k] += 1
        tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
        # Attack
        if tmp_v.shape == v.shape:
            difference = v -tmp_v
            # if(attack != None):
            white(difference,byzantinesize)              # white
            # maxValue(difference,byzantinesize)         # maxvalue
            # zeroGradient(difference,byzantinesize)     # zerogradient
            v[count[k] > 0] = tmp_v[count[k] > 0].to(v.dtype)
        else:
            raise ValueError("The shapes of a and b must be the same.")
        v[count[k] > 0] = difference[count[k] > 0].to(v.dtype)
        
# local_parameters, param_idx = distribute(user_idx)
# combine(local_parameters,param_idx,user_idx)

# python train_classifier_fed.py --data_name MNIST --model_name conv --control_name 1_100_0.1_iid_fix_a2-b8_bn_1_1

# a = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
# b = torch.tensor([[1.1,2.1,3.1],[4.1,5.1,6.1]])
# if a.shape == b.shape:
#     difference = b - a
#     print(difference)

# else:
#     raise ValueError("The shapes of a and b must be the same.")\
    
val_losses = [2.6, 2.0, 1.5, 1.1, 0.9, 0.8]
epochs = range(1, 7)
    
# # 绘制损失曲线
# plt.figure(figsize=(8, 6))
# plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Loss Curve')
# plt.legend()
# plt.grid(True)
# plt.show()

# a = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
# b = torch.mean(a,dim = 1)
# print(b)

# max_iter = 80
# tol = 1e-7

# wList = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0],[1.0,2.0,3.0],[4.0,5.0,6.0],[1.0,2.0,3.0]])
# guess = torch.mean(wList, dim=0)

# for _ in range(max_iter):
#     dist_li = torch.norm(wList-guess, dim=1)
#     for i in range(len(dist_li)):
#         if dist_li[i] == 0:
#             dist_li[i] = 1
#     temp1 = torch.sum(torch.stack([w/d for w, d in zip(wList, dist_li)]), dim=0)
#     temp2 = torch.sum(1/dist_li)
#     guess_next = temp1 / temp2
#     guess_movement = torch.norm(guess - guess_next)
#     guess = guess_next
#     if guess_movement <= tol:
#         break

def Krum_(nodeSize, byzantineSize):
    honestSize = nodeSize - byzantineSize
    dist = torch.zeros(nodeSize, nodeSize, dtype=torch.float32)
    def Krum(wList):
        for i in range(nodeSize):
            for j in range(i, nodeSize):
                distance = wList[i].data - wList[j].data
                distance = (distance*distance).sum()
                distance = -distance # 两处都是取距离的最小值，需要改成负数
                dist[i][j] = distance.data
                dist[j][i] = distance.data
        k = nodeSize - byzantineSize - 2 + 1 # 算上自己和自己的0.00
        topv, _ = dist.topk(k=k, dim=1)
        sumdist = topv.sum(dim=1)
        resindex = sumdist.topk(1)[1].squeeze()
        return wList[resindex]
    return Krum

# from config import cfg
# print(cfg['num_epochs']['global'])

from metrics import Metric

# # 假设 output 和 target 是模型的输出和目标标签
# output = {'score': torch.randn(10, 5)}  # 10个样本，每个样本5个类的预测分数
# target = {'label': torch.randint(0, 5, (10,))}  # 10个样本的真实标签

# # 创建 Metric 类的实例
# metric = Metric()

# # 计算多个评估指标
# metrics_to_evaluate = ['Accuracy', 'Perplexity']
# evaluation = metric.evaluate(metrics_to_evaluate, target, output)

# # 打印评估结果
# print(evaluation)

user_idx = [1,3,5,7,9]

num_active_users  = 5
num_byzantine_users = 2
byzantine_user_idx = torch.randperm(num_active_users)[:num_byzantine_users].tolist()

byzantine_user_idx = [user_idx[i] for i in byzantine_user_idx]
print(byzantine_user_idx)