import numpy as np
import jittor as jt
from jittor import nn, optim
from MMonitor.quantity.singlestep import *
jt.set_global_seed(42)
np.random.seed(42)
from MMonitor.mmonitor.monitor import Monitor
from MMonitor.quantity.singlestep import *
from MMonitor.visualize import Visualization, LocalVisualization
# 设置随机种子
jt.set_global_seed(42)
np.random.seed(42)
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
        nn.BatchNorm2d: ['BackwardOutputNorm','BackwardOutputMean','BackwardOutputStd']
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
    print(f"{model}的{name}指标预期值{b}")
    if abs(a - b) <= tolerance:
        return True
    else:
        return False
def test_linear_mean():
    l = nn.Linear(2, 3)
    x_linear = jt.randn((4, 2), requires_grad=True)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = BackwardOutputMean(l)
    for hook in quantity.backward_extensions():
        l.register_backward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    input_mean = jt.mean(jt.grad(loss,y))
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_linear'
    name = 'backward_output_mean'
    print(if_similar(quantity.get_output()[0],input_mean,model,name))
def test_conv_mean():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = jt.randn((4, 1, 3, 3), requires_grad=True)
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = BackwardOutputMean(cov)
    for hook in quantity.backward_extensions():
        cov.register_backward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    input_mean = jt.mean(jt.grad(loss,y))
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_conv'
    name = 'backward_output_mean'
    print(if_similar(quantity.get_output()[0],input_mean,model,name))

def test_default_mean():
    x_default = jt.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x_default.start_grad()
    # 创建 BatchNorm 层
    bn = nn.BatchNorm(2)  # 输入特征数为 2
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = BackwardOutputMean(bn)
    for hook in quantity.backward_extensions():
        bn.register_backward_hook(hook)
    i = 0
    y = bn(x_default)
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    input_mean = jt.mean(jt.grad(loss,y))
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_bn'
    name = 'backward_output_mean'
    print(if_similar(quantity.get_output()[0],input_mean,model,name))
def test_linear_norm():
    l = nn.Linear(2, 3)
    x_linear = jt.randn((4, 2), requires_grad=True)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = BackwardOutputNorm(l)
    for hook in quantity.backward_extensions():
        l.register_backward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    input_norm = jt.norm(jt.flatten(jt.grad(loss,y)))
    optimizer.step(loss)
    quantity.track(i)   
    model = 'jittor_linear'
    name = 'backward_output_norm' 
    print(if_similar(quantity.get_output()[0],input_norm,model,name))
def test_conv_norm():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = jt.randn((4, 1, 3, 3), requires_grad=True)
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = BackwardOutputNorm(cov)
    for hook in quantity.backward_extensions():
        cov.register_backward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    input_norm = jt.norm(jt.flatten(jt.grad(loss,y)))
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_conv'
    name = 'backward_output_norm' 
    print(if_similar(quantity.get_output()[0],input_norm,model,name))

def test_default_norm():
    x_default = jt.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x_default.start_grad()
    # 创建 BatchNorm 层
    bn = nn.BatchNorm(2)  # 输入特征数为 2
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = BackwardOutputNorm(bn)
    for hook in quantity.backward_extensions():
        bn.register_backward_hook(hook)
    i = 0
    y = bn(x_default)
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    input_norm = jt.norm(jt.flatten(jt.grad(loss,y)))
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_linear'
    name = 'backward_output_norm' 
    print(if_similar(quantity.get_output()[0],input_norm,model,name))
def test_linear_std():
    l = nn.Linear(2, 3)
    x_linear = jt.randn((4, 2), requires_grad=True)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = BackwardOutputStd(l)
    for hook in quantity.backward_extensions():
        l.register_backward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    input_std = jt.std(jt.grad(loss,y))
    optimizer.step(loss)
    quantity.track(i)   
    name = 'backward_output_std'
    model = 'jittor_linear' 
    print(if_similar(quantity.get_output()[0],input_std,model,name))
def test_conv_std():
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x_conv = jt.randn((4, 1, 3, 3), requires_grad=True)
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = BackwardOutputStd(cov)
    for hook in quantity.backward_extensions():
        cov.register_backward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    input_std = jt.std(jt.grad(loss,y))
    optimizer.step(loss)
    quantity.track(i)
    name = 'backward_output_std'
    model = 'jittor_conv' 
    print(if_similar(quantity.get_output()[0],input_std,model,name))

def test_default_std():
    x_default = jt.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x_default.start_grad()
    # 创建 BatchNorm 层
    bn = nn.BatchNorm(2)  # 输入特征数为 2
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = BackwardOutputStd(bn)
    for hook in quantity.backward_extensions():
        bn.register_backward_hook(hook)
    i = 0
    y = bn(x_default)
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    input_std = jt.std(jt.grad(loss,y))
    optimizer.step(loss)
    quantity.track(i)
    name = 'backward_output_std'
    model = 'jittor_linear' 
    print(if_similar(quantity.get_output()[0],input_std,model,name))
if __name__ == '__main__':
    test_linear_mean()
    test_conv_mean()
    test_default_mean()
    test_linear_norm()
    test_conv_norm()
    test_default_norm()
    test_linear_std()
    test_conv_std()
    test_default_std()
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
        model.train()
        # 前向传播
        y_hat = model(x)
        loss = loss_fun(y_hat,y)
        opt.step(loss)
        monitor.track(epoch)
        logs = vis.show(epoch)
        print(f"Epoch {epoch} loss: {loss}")
    # 输出监控结果并显示
    print("Training completed.")
    show_local(monitor)