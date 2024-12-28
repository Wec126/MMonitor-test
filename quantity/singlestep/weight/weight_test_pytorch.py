import numpy as np
from MMonitor.quantity.singlestep import *
import torch
from torch import nn as nn
# 设置随机种子
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
from torch import optim
from MMonitor.mmonitor.monitor import Monitor
from MMonitor.visualize import Visualization,LocalVisualization
from MMonitor.quantity.singlestep import *
class Model(nn.Module):
    def __init__(self, w=224, h=224, class_num=5):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 5, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(5)
        self.relu2 = nn.ReLU()
        self.l1 = nn.Linear(w * h * 5, class_num)

    def forward(self, x):
        r1 = self.relu1(self.conv1(x))
        r2 = self.relu2(self.bn(self.conv2(r1)))
        r2 = r2.view(x.size(0), -1)
        l1 = self.l1(r2)
        return l1
def prepare_data(w, h, class_num, length):
    x = np.random.random((length, 3, w, h))  # 或者使用 np.random.randn
    y = np.random.randint(0, class_num, (length,))
    return x, y
def prepare_optimizer(model, lr=1e-2):
    return optim.SGD(model.parameters(), lr=lr)

def show_local(monitor, quantity_name=None):
    project = 'BatchNorm2d'
    localvis = LocalVisualization(project=project)
    localvis.show(monitor, quantity_name=quantity_name, project_name=project)
    print('The result has been saved locally')

def prepare_config():
    config = {'epoch': 100, 'w': 224, 'h': 224, 'class_num': 5, 'len': 100, 'lr': 1e-2}
    config_mmonitor = {
        nn.BatchNorm2d: ['WeightNorm','WeightMean','WeightStd'],
        nn.Conv2d: ['WeightNorm','WeightMean','WeightStd'],
        nn.Linear: ['WeightNorm','WeightMean','WeightStd']
    }
    return config, config_mmonitor

def prepare_loss_func():
    return nn.CrossEntropyLoss()
def if_similar(a,b,model,name):
    print(f'{model}的{name}指标的计算所得值:{a}')
    print(f"{model}的{name}的预期指标：{b}")
    if np.allclose(a, b, rtol=1e-5, atol=1e-8):
        print("True")
    else:
        print("False")
def compute_linear_mean():
    l = nn.Linear(2, 3)
    x_linear = torch.randn((4, 2))
    quantity_l = WeightMean(l)
    i = 0
    y = l(x_linear)
    quantity_l.track(i)
    weight_mean_direct = l.weight.mean().item()
    model = 'pytorch_linear'
    name = 'weight_mean'
    if_similar(quantity_l.get_output()[0].item(), weight_mean_direct,model,name)
def compute_conv_mean():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = torch.randn((4, 1, 3, 3))
    quantity_c = WeightMean(cov)
    i = 0
    y_c = cov(x_conv)
    quantity_c.track(i)
    weight_mean_direct = cov.weight.mean().item()
    model = 'pytorch_conv'
    name = 'weight_mean'
    if_similar(quantity_c.get_output()[0].item(),weight_mean_direct,model,name)
def compute_default_mean():
    # 定义 BatchNorm2d 层
    bn = nn.BatchNorm2d(3)  # 假设有 3 个通道
    # 定义输入
    x_default = torch.randn(2, 3, 4, 4, requires_grad=True)  # BatchSize=2, Channels=3, H=W=4
    # 前向传播
    quantity = WeightMean(bn)
    y = bn(x_default)
    quantity.track(0)
    weight_mean_direct = bn.weight.mean().item()
    model = 'pytorch_bn'
    name = 'weight_mean'
    if_similar(quantity.get_output()[0].item(),weight_mean_direct,model,name)
def compute_linear_norm():
    l = nn.Linear(2, 3)
    x_linear = torch.randn((4, 2))
    quantity_l = WeightNorm(l)
    i = 0
    y = l(x_linear)
    quantity_l.track(i)
    weight_norm_direct = l.weight.norm().item()
    model = 'pytorch_linear'
    name = 'weight_norm'
    if_similar(quantity_l.get_output()[0].item(), weight_norm_direct,model,name)
def compute_conv_norm():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = torch.randn((4, 1, 3, 3))
    quantity_c = WeightNorm(cov)
    i = 0
    y_c = cov(x_conv)
    quantity_c.track(i)
    weight_norm_direct = cov.weight.norm().item()
    model = 'pytorch_conv'
    name = 'weight_norm'
    if_similar(quantity_c.get_output()[0].item(),weight_norm_direct,model,name)
def compute_default_norm():
    # 定义 BatchNorm2d 层
    bn = nn.BatchNorm2d(3)  # 假设有 3 个通道
    # 定义输入
    x_default = torch.randn(2, 3, 4, 4, requires_grad=True)  # BatchSize=2, Channels=3, H=W=4
    # 前向传播
    quantity = WeightNorm(bn)
    y = bn(x_default)
    quantity.track(0)
    weight_norm_direct = bn.weight.norm().item()
    model = 'pytorch_bn'
    name = 'weight_norm'
    if_similar(quantity.get_output()[0].item(),weight_norm_direct,model,name)
def compute_linear_std():
    l = nn.Linear(2, 3)
    x_linear = torch.randn((4, 2))
    quantity_l = WeightStd(l)
    i = 0
    y = l(x_linear)
    quantity_l.track(i)
    weight_std_direct = l.weight.std().item()
    model = 'pytorch_linear'
    name = 'weight_std'
    if_similar(quantity_l.get_output()[0].item(), weight_std_direct,model,name)
def compute_conv_std():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = torch.randn((4, 1, 3, 3))
    quantity_c = WeightStd(cov)
    i = 0
    y_c = cov(x_conv)
    quantity_c.track(i)
    weight_std_direct = cov.weight.std().item()
    model = 'pytorch_conv'
    name = 'weight_std'
    if_similar(quantity_c.get_output()[0].item(),weight_std_direct,model,name)
def compute_default_std():
    # 定义 BatchNorm2d 层
    bn = nn.BatchNorm2d(3)  # 假设有 3 个通道
    # 定义输入
    x_default = torch.randn(2, 3, 4, 4, requires_grad=True)  # BatchSize=2, Channels=3, H=W=4
    # 前向传播
    quantity = WeightStd(bn)
    y = bn(x_default)
    quantity.track(0)
    weight_std_direct = bn.weight.std().item()
    model = 'pytorch_bn'
    name = 'weight_std'
    if_similar(quantity.get_output()[0].item(),weight_std_direct,model,name)
if __name__ == '__main__':
    compute_conv_mean()
    compute_linear_mean()
    compute_default_mean()
    compute_conv_norm()
    compute_linear_norm()
    compute_default_norm()
    compute_conv_std()
    compute_linear_std()
    compute_default_std()
    config, config_mmonitor = prepare_config()
    x, y = prepare_data(config['w'], config['h'], config['class_num'], config['len'])
    x = torch.tensor(x, dtype=torch.float32) 
    y = torch.tensor(y, dtype=torch.long)
    model = Model(config['w'], config['h'], config['class_num'])
    opt = prepare_optimizer(model, config['lr'])
    loss_fun = prepare_loss_func()

    # 监控和可视化设置
    monitor = Monitor(model, config_mmonitor)
    vis = Visualization(monitor, project=config_mmonitor.keys(), name=config_mmonitor.values())

    # 训练循环
    for epoch in range(10):
        # 前向传播
        y_hat = model(x)
        # 计算损失
        loss = loss_fun(y_hat, y)
        # 监控和可视化
        monitor.track(epoch)
        vis.show(epoch)
        print(f"Epoch {epoch} loss: {loss}")

        # 反向传播和优化
        opt.zero_grad()
        loss.backward()
        opt.step()

    # 输出监控结果并显示
    print("Training completed.")
    show_local(monitor)