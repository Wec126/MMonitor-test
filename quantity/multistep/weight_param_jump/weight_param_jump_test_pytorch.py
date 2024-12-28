import torch
import torch.nn as nn
import numpy as np
from MMonitor.quantity.multistep import *   
from MMonitor.mmonitor.monitor import Monitor
from MMonitor.visualize import Visualization,LocalVisualization
from MMonitor.quantity.singlestep import *
from torch import nn,optim

def is_reasonable(value,model,name,threshold=1e-3):
    """判断参数跳跃值是否在合理范围内
    Args:
        value: 计算得到的参数跳跃值
        threshold: 阈值，默认为0.001
    Returns:
        bool: 是否合理
    """
    print(f"{model}的{name}指标的计算值{value}")
    return abs(value) < threshold
# 定义前向钩子来保存输入
def hook(module, input, output):
    module.last_input = input[0]  # input是一个元组，我们取第一个元素
# 创建BatchNorm层
def test_bn():
    l = nn.BatchNorm2d(3)
    x = torch.tensor(np.random.randn(4, 3, 3, 3).astype(np.float32))
    quantity_l = WeightParamJump(l)
    i = 0
    y = l(x)
    # 不需要手动设置forward_input
    quantity_l.track(i)
    model = 'pytorch_bn'
    name = 'weight_param_jump'
    print(is_reasonable(quantity_l.get_output()[0],model,name))

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
    project = 'BatchNorm'
    localvis = LocalVisualization(project=project)
    localvis.show(monitor, quantity_name=quantity_name, project_name=project)
    print('The result has been saved locally')

def prepare_config():
    config = {'epoch': 10, 'w': 224, 'h': 224, 'class_num': 5, 'len': 100, 'lr': 1e-2}
    config_mmonitor = {
        nn.BatchNorm2d: ['WeightParamJump']
    }
    return config, config_mmonitor

def prepare_loss_func():
    return nn.CrossEntropyLoss()

if __name__ == '__main__':
    test_bn()
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


