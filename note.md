### utils.py:

save(input, path, protocol=2, mode='torch'): 将输入数据保存到指定路径；
to_device(input, device): 将输入递归地移动到指定的设备；
process_dataset(dataset): 根据数据集的类型对数据集进行特定的预处理，适配模型训练需求；
collate(input): 将输入数据的每个值堆叠成张量；
ntuple(n): 用于生成具有相同长度的元组；
recur(fn, input, *args): 是一个递归函数，旨在将给定的函数 fn 应用到嵌input的每个元素

process_control(): 初始化并处理训练相关的超参数和控制参数；
make_optimizer(model, lr): 创建对应的优化器实例；
make_scheduler(optimizer): 为优化器创建对应的学习率调度器实例；
resume(model, model_tag, optimizer=None, scheduler=None, load_tag='checkpoint', strict=True, verbose=True): 该函数用于恢复模型训练的状态，包括模型权重、优化器状态、学习率调度器状态以及日志记录。如果指定的检查点不存在，则从头开始初始化训练；

### config.py

global vfg: 检查是否已定义全局变量cfg，如果没有，则从配置文件 config.yml 加载全局配置并赋值给cfg；

### logger.py-Class logger
用于在训练过程中记录、更新和保存模型的训练/验证统计信息。它可以跟踪各种指标的历史，并使用 TensorBoard（通过 SummaryWriter）将其写入日志文件。这对于调试、可视化和性能分析非常有用；

safe(self, write): 开或关闭 TensorBoard 的 SummaryWriter，根据 write 的布尔值决定是否保存到日志文件。如果关闭 TensorBoard，历史数据将保存在 history 字典中；
reset(self): 重置统计信息；
append(self, result, tag, n=1, mean=True): 将新的训练/验证结果添加到日志中。更新平均值并计算统计数据；
write(self, tag, metric_names): 将指定的指标写入日志并打印到控制台。用于显示评估信息；

### metrics.py

Accuracy(output, target, topk=1): 计算准确度；
Perplexity(output, target): 计算困惑度，是基于交叉熵损失函数的指数函数；
class Metric: 将各种评估指标（如准确度、困惑度等）进行组织，并提供一种统一的方式来评估模型性能；

### data.py

fetch_dataset(data_name, subset): 加载和预处理相应的训练集和测试集；
input_collate(batch): 在加载数据时将不同的样本合并成一个batch；
iid(dataset, num_users): 将数据集中的样本分配给多个客户端，并确保每个客户端的数据集是IID的；
non_iid(dataset, num_users, label_split=None): 将数据集中的样本分配给多个客户端，并确保每个客户端的数据集是non-IID的；
split_dataset(dataset, num_users, data_split_mode): 根据给定的数据分割模式（data_split_mode）将数据集划分为每个客户端的数据；
make_data_loader(dataset): 创建并返回一个包含多个数据加载器的字典。每个数据加载器对应于输入数据集的一个部分，例如，训练集、测试集/['train'],['test']；

### fed.py(key): Class Federation

make_model_rate(self): 根据配置文件中定义的 model_split_mode 来设置 model_rate，即每个用户的模型速率；
split_model(self, user_idx): 将全局模型的参数分割；
distribute(self, user_idx): 将分割后的局部模型参数分发给不同的客户端；
combine(self, local_parameters, param_idx, user_idx): 

# 主函数

Class local: 表示一个客户端，它的作用是执行本地训练任务；
make_local(dataset, data_split, label_split, federation): 为多个客户端准备本地训练环境；
test(dataset, data_split, label_split, model, logger, epoch): 用于在测试阶段评估模型的性能，对客户端和全局模型进行测试，并记录相关指标；
train(dataset, data_split, label_split, federation, global_model, optimizer, logger, epoch): 执行训练过程；
stats(dataset, model): 用来返回一个已加载当前全局模型参数的模型副本；