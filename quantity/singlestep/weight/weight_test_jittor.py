import jittor as jt
from jittor import nn, optim
from MMonitor.quantity.singlestep import *
import numpy as np
jt.set_global_seed(42)
np.random.seed(42)
from MMonitor.mmonitor.monitor import Monitor
from MMonitor.visualize.visualizer import Visualization,LocalVisualization
import numpy as np
class Model(nn.Module):
    def __init__(self, w=224, h=224, class_num=5):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 5, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm(5)
        self.relu2 = nn.ReLU()
        self.l1 = nn.Linear(w * h * 5, class_num)

    def execute(self, x):
        r1 = self.relu1(self.conv1(x))
        r2 = self.relu2(self.bn(self.conv2(r1)))
        r2 = r2.view(x.size(0), -1)
        l1 = self.l1(r2)
        return l1
def prepare_data(w, h, class_num, length):
    x = jt.random((length, 3, w, h))  # Jittor中使用jt.random
    y = jt.randint(0, class_num, (length,))
    return x, y

def prepare_optimizer(model, lr=1e-2):
    return jt.optim.SGD(model.parameters(), lr=lr)

def show_local(monitor, quantity_name=None):
    project = 'BatchNorm2d'
    localvis = LocalVisualization(project=project)
    localvis.show(monitor, quantity_name=quantity_name, project_name=project)
    print('The result has been saved locally')

def prepare_config():
    config = {'epoch': 100, 'w': 224, 'h': 224, 'class_num': 5, 'len': 100, 'lr': 1e-2}
    config_mmonitor = {
        nn.BatchNorm2d: ['WeightMean','WeightNorm','WeightStd'],
        nn.Linear: ['WeightMean','WeightNorm','WeightStd'],
        nn.Conv2d: ['WeightMean','WeightNorm','WeightStd']
    }
    return config, config_mmonitor
def prepare_loss_func():
    return nn.CrossEntropyLoss()
def if_similar(a,b,model,name,tolerance = 0.05):
    if not isinstance(a,(int,float)):
        a = a.item()
    print(f'{model}的{name}指标的当前计算所得值{a}')
    if not isinstance(b,(int,float)):
        b= b.item()
    print(f"{model}的{name}的预期值{b}")
    if abs(a - b) <= tolerance:
        return True
    else:
        return False
def test_linear_mean():
    # 1. 判断初始化方法下的权重均值
    # 使用默认初始化 -> 判断是否在-0.1~0.1之间
    l = nn.Linear(2, 3)
    x_linear = jt.randn((4, 2), requires_grad=True)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = WeightStd(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)
    optimizer.step(loss)
    quantity.track(i)
    # 添加直接计算权重均值的代码
    weight_mean_direct = jt.mean(l.weight).item()
    name = 'weight_mean'
    model = 'jittor_linear'
    print(if_similar(quantity.get_output()[0],weight_mean_direct,model,name))
def test_conv_mean():
    # 使用正态分布初始化
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = jt.randn((4, 1, 3, 3), requires_grad=True)
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = WeightMean(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    weight_mean_direct = jt.mean(cov.weight).item()
    name = 'weight_mean'
    model = 'jittor_conv'
    print(if_similar(quantity.get_output()[0],weight_mean_direct,model,name))
    
def test_default_mean():
    x_default = jt.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x_default.start_grad()
    # 创建 BatchNorm 层
    bn = nn.BatchNorm(2)  # 输入特征数为 2
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = WeightMean(bn)
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)
    i = 0
    y = bn(x_default)
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    weight_mean_direct = jt.mean(bn.weight).item()
    name = 'weight_mean'
    model = 'jittor_vn'
    print(if_similar(quantity.get_output()[0],weight_mean_direct,model,name))
def test_linear_std():
    # 1. 判断初始化方法下的权重均值
    # 使用默认初始化 -> 判断是否在-0.1~0.1之间
    l = nn.Linear(2, 3)
    x_linear = jt.randn((4, 2), requires_grad=True)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = WeightStd(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)
    optimizer.step(loss)
    quantity.track(i)
    # 添加直接计算权重均值的代码
    weight_std_direct = jt.std(l.weight).item()
    name = 'weight_std'
    model = 'jittor_linear'
    print(if_similar(quantity.get_output()[0],weight_std_direct,model,name))
def test_conv_std():
    # 使用正态分布初始化
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = jt.randn((4, 1, 3, 3), requires_grad=True)
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = WeightStd(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    weight_std_direct = jt.std(cov.weight).item()
    name = 'weight_std'
    model = 'jittor_conv'
    print(if_similar(quantity.get_output()[0],weight_std_direct,model,name))
    
def test_default_std():
    x_default = jt.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x_default.start_grad()
    # 创建 BatchNorm 层
    bn = nn.BatchNorm(2)  # 输入特征数为 2
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = WeightStd(bn)
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)
    i = 0
    y = bn(x_default)
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    weight_std_direct = jt.std(bn.weight).item()
    name = 'weight_std'
    model = 'jittor_bn'
    print(if_similar(quantity.get_output()[0],weight_std_direct,model,name))
def test_linear_norm():
    # 1. 判断初始化方法下的权重均值
    # 使用默认初始化 -> 判断是否在-0.1~0.1之间
    l = nn.Linear(2, 3)
    x_linear = jt.randn((4, 2), requires_grad=True)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = WeightNorm(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)
    optimizer.step(loss)
    quantity.track(i)
    # 添加直接计算权重均值的代码
    weight_norm_direct = jt.norm(jt.flatten(l.weight)).item()
    name = 'weight_norm'
    model = 'jittor_linear'
    print(if_similar(quantity.get_output()[0],weight_norm_direct,model,name))
def test_conv_norm():
    # 使用正态分布初始化
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = jt.randn((4, 1, 3, 3), requires_grad=True)
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = WeightNorm(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    weight_norm_direct = jt.norm(jt.flatten(cov.weight)).item()
    name = 'weight_norm'
    model = 'jittor_conv'
    print(if_similar(quantity.get_output()[0],weight_norm_direct,model,name))
    
def test_default_norm():
    x_default = jt.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x_default.start_grad()
    # 创建 BatchNorm 层
    bn = nn.BatchNorm(2)  # 输入特征数为 2
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = WeightNorm(bn)
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)
    i = 0
    y = bn(x_default)
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    weight_norm_direct = jt.norm(jt.flatten(bn.weight)).item()
    name = 'weight_norm'
    model = 'jittor_bn'
    print(if_similar(quantity.get_output()[0],weight_norm_direct,model,name))

if __name__ =='__main__':
    test_linear_mean()
    test_conv_mean()
    test_default_mean()
    test_linear_std()
    test_conv_std()
    test_default_std()
    test_linear_norm()
    test_conv_norm()
    test_default_norm()
        # 主训练循环
    config, config_mmonitor = prepare_config()
    x, y = prepare_data(config['w'], config['h'], config['class_num'], config['len'])
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
        opt.step(loss)

    # 输出监控结果并显示
    print("Training completed.")
    show_local(monitor)
