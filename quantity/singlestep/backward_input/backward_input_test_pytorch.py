
# 使用反向验证
import torch
import random
import numpy as np
import torch.nn as nn
from torch import optim
from MMonitor.mmonitor.monitor import Monitor
# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
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
        nn.BatchNorm2d: ['BackwardInputNorm','BackwardInputMean','BackwardInputStd']
    }
    return config, config_mmonitor

def prepare_loss_func():
    return nn.CrossEntropyLoss()
def if_similar(a,b,model,name):
    print(f"{model}的{name}指标的计算所得值为{a}")
    print(f"{model}的{name}指标的理论值为{b}")
    if np.allclose(a, b, rtol=1e-5, atol=1e-8):
        print("True")
    else:
        print("False")
def compute_linear_mean():
    l = nn.Linear(2,3) 
    x = torch.randn(4,2,requires_grad=True)
    quantity_l = BackwardOutputMean(l)
    for hook in quantity_l.backward_extensions():
        l.register_full_backward_hook(hook)
    i = 0
    y = l(x)
    y.retain_grad()
    loss = y.sum()
    loss.backward()
    quantity_l.track(i)
    print(quantity_l.get_output()[0])
    name = 'backward_input_mean'
    model = 'pytorch_linear'
    if_similar(quantity_l.get_output()[0],y.grad.mean(),model,name)
def compute_default_mean():
    relu = nn.ReLU()
    x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
    # 前向传播
    quantity = BackwardOutputMean(relu)
    for hook in quantity.backward_extensions():
        relu.register_full_backward_hook(hook)
    y = relu(x)
    y.retain_grad()
    i = 0
    # 反向传播
    y.sum().backward()
    quantity.track(i)
    print(quantity.get_output()[0])
    name = 'backward_input_mean'
    model = 'pytorch_relu'
    if_similar(quantity.get_output()[0],y.grad.mean(),model,name)

def compute_conv_mean():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_c = torch.randn((4, 1, 3, 3), requires_grad=True)
    quantity = BackwardOutputMean(cov)
    for hook in quantity.backward_extensions():
        cov.register_full_backward_hook(hook)
    i = 0
    y_c = cov(x_c)
    y_c.retain_grad()
    y_c.sum().backward()
    quantity.track(i)
    print(quantity.get_output()[0])
    name = 'backward_input_mean'
    model = 'pytorch_conv'
    if_similar(quantity.get_output()[0],y_c.grad.mean(),model,name)
def compute_linear_std():
    l = nn.Linear(2,3) 
    x = torch.randn(4,2,requires_grad=True)
    quantity_l = BackwardInputStd(l)
    for hook in quantity_l.backward_extensions():
        l.register_full_backward_hook(hook)
    i = 0
    y = l(x)
    y.retain_grad()
    loss = y.sum()
    loss.backward()
    quantity_l.track(i)
    model = 'pytorch_linear'
    name = 'backward_input_std'
    if_similar(quantity_l.get_output()[0],x.grad.std(),model,name)
def compute_default_std():
    relu = nn.ReLU()
    x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
    # 前向传播
    quantity = BackwardInputStd(relu)
    for hook in quantity.backward_extensions():
        relu.register_full_backward_hook(hook)
    y = relu(x)
    y.retain_grad()
    i = 0
    # 反向传播
    y.sum().backward()
    quantity.track(i)
    model = 'pytorch_relu'
    name = 'backward_input_std'
    if_similar(quantity.get_output()[0],x.grad.std(),model,name)

def compute_conv_std():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_c = torch.randn((4, 1, 3, 3), requires_grad=True)
    quantity = BackwardInputStd(cov)
    for hook in quantity.backward_extensions():
        cov.register_full_backward_hook(hook)
    i = 0
    y_c = cov(x_c)
    y_c.retain_grad()
    y_c.sum().backward()
    quantity.track(i)
    model = 'pytorch_conv'
    name = 'backward_input_std'
    if_similar(quantity.get_output()[0],x_c.grad.std(),model,name)
def compute_linear_norm():
    l = nn.Linear(2,3) 
    x = torch.randn(4,2,requires_grad=True)
    quantity_l = BackwardInputNorm(l)
    for hook in quantity_l.backward_extensions():
        l.register_full_backward_hook(hook)
    i = 0
    y = l(x)
    y.retain_grad()
    loss = y.sum()
    loss.backward()
    quantity_l.track(i)
    name = 'backward_input_norm'
    model = 'pytorch_linear'
    if_similar(quantity_l.get_output()[0],x.grad.norm(),name,model)
def compute_default_norm():
    relu = nn.ReLU()
    x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], requires_grad=True)
    # 前向传播
    quantity = BackwardInputNorm(relu)
    for hook in quantity.backward_extensions():
        relu.register_full_backward_hook(hook)
    y = relu(x)
    y.retain_grad()
    i = 0
    # 反向传播
    y.sum().backward()
    quantity.track(i)
    name = 'backward_input_norm'
    model = 'pytorch_relu'
    if_similar(quantity.get_output()[0],x.grad.norm(),model,name)

def compute_conv_norm():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_c = torch.randn((4, 1, 3, 3), requires_grad=True)
    quantity = BackwardInputNorm(cov)
    for hook in quantity.backward_extensions():
        cov.register_full_backward_hook(hook)
    i = 0
    y_c = cov(x_c)
    y_c.retain_grad()
    y_c.sum().backward()
    quantity.track(i)
    name = 'backward_input_norm'
    model = 'pytorch_conv'
    if_similar(quantity.get_output()[0],x_c.grad.norm(),model,name)

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
        # 反向传播和优化
        opt.zero_grad()
        loss.backward()
        monitor.track(epoch)
        vis.show(epoch)
        print(f"Epoch {epoch} loss: {loss}")
        opt.step()


    # 输出监控结果并显示
    print("Training completed.")
    show_local(monitor)




