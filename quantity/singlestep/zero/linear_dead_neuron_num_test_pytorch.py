from regex import F
import torch
from torch import nn as nn

from MMonitor.mmonitor.monitor import Monitor
from torch import optim
import numpy as np
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
        nn.BatchNorm2d: ['LinearDeadNeuronNum'],
        nn.Conv2d: ['LinearDeadNeuronNum'],
        nn.Linear: ['LinearDeadNeuronNum']
    }
    return config, config_mmonitor

def prepare_loss_func():
    return nn.CrossEntropyLoss()
def if_similar(a,b,model,name):
    print(f"{model}的{name}指标的当前计算指标为{a}")
    print(f"{model}的{name}指标的预期指标为{b}")
    if a == b:
        print('True')
    else:
        print('False')
def compute_linear():
    l = nn.Linear(2, 3)
    x_linear = torch.randn((4, 2))
    quantity_l = LinearDeadNeuronNum(l)
    for hook in quantity_l.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    output = y.detach()
    dead_count = 0
    for neuron_idx in range(output.shape[1]):
        if torch.all(torch.abs(output[:, neuron_idx]) < 1e-6).item():  # 使用torch.all()替代np.all()
            dead_count += 1
    quantity_l.track(i)
    model = 'pytorch_linear'
    name = 'linear_dead_neuron_num'
    if_similar(quantity_l.get_output()[0].item(),dead_count,model,name)

def compute_conv():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = torch.randn((4, 1, 3, 3))
    quantity_c = LinearDeadNeuronNum(cov)
    for hook in quantity_c.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y_c = cov(x_conv)
    output = y_c.detach()
    dead_count = 0
    for neuron_idx in range(output.shape[1]):
        if torch.all(torch.abs(output[:, neuron_idx]) < 1e-6).item():
            dead_count += 1
    quantity_c.track(i)
    model = 'pytorch_conv'
    name = 'linear_dead_neuron_num'
    if_similar(quantity_c.get_output()[0].item(),dead_count,model,name)

def compute_default():
    relu = nn.ReLU()
    x_default = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
    # 前向传播
    quantity = LinearDeadNeuronNum(relu)
    for hook in quantity.forward_extensions():
        relu.register_forward_hook(hook)
    y = relu(x_default)
    i = 0
    quantity.track(i)
    output = y.detach()
    dead_count = 0
    for neuron_idx in range(output.shape[1]):
        if torch.all(torch.abs(output[:, neuron_idx]) < 1e-6).item():  # 将 np.all 改为 torch.all
            dead_count += 1
    # 反向传播
    y.sum().backward()
    model = 'pytorch_bn'
    name = 'linear_dead_neuron_num'
    if_similar(quantity.get_output()[0].item(),dead_count,model,name)

if __name__ == '__main__':
    compute_linear()
    compute_conv()
    compute_default()
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
    



