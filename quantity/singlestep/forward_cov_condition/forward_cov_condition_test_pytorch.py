import torch
import torch.nn as nn
from torch import optim
from MMonitor.mmonitor.monitor import Monitor
from MMonitor.quantity.singlestep import *
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
        nn.BatchNorm2d: ['ForwardInputCovCondition20',
                         'ForwardInputCovCondition50',
                         'ForwardInputCovCondition80',
                         'ForwardInputCovMaxEig',
                         'ForwardInputCovStableRank']
    }
    return config, config_mmonitor

def prepare_loss_func():
    return nn.CrossEntropyLoss()
def is_similar(a, b,model,name,tolerance=0.1):
    print(f"{model}的{name}指标的当前计算值{a}")
    if a < 0:
        return False
    return abs(a - b) < tolerance

def is_similar_for_stable_rank(a, b,model,name,tolerance=0.1):
    print(f"{model}的{name}指标的当前计算值{a}")
    if a < 0:
        return False
    return abs(a - b) >=0
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def test_linear_cov_condition():
    setup_seed(42)
    l = nn.Linear(2, 3)
    
    # 使用PyTorch的初始化方式
    nn.init.normal_(l.weight, std=0.1)
    nn.init.zeros_(l.bias)
    
    batch_size = 1024
    x_linear = torch.randn(batch_size, 2)
    # 标准化输入
    mean = x_linear.mean(dim=0, keepdim=True)
    std = x_linear.std(dim=0, keepdim=True)
    x_linear = (x_linear - mean) / std
    
    quantity = ForwardInputCovCondition(l)
    for extension in quantity.forward_extensions():
        l.register_forward_hook(extension)
    i = 0
    optimizer = torch.optim.SGD(l.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    y = l(x_linear)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    quantity.track(i)
    model = 'pytorch_linear'
    name = 'forward_cov_condition'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_cov_condition():
    setup_seed(42)
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv = nn.Conv2d(1, 2, 3, stride=1, padding=1)
            nn.init.normal_(self.conv.weight, std=0.03)
            nn.init.zeros_(self.conv.bias)
        
        def forward(self, x):
            return self.conv(x)
    
    net = ConvNet()
    batch_size = 1024
    x_conv = torch.randn(batch_size, 1, 3, 3)
    # 标准化输入
    x_conv = x_conv / torch.sqrt((x_conv ** 2).mean(dim=(0,2,3), keepdim=True))
    
    quantity = ForwardInputCovCondition(net.conv)
    for extension in quantity.forward_extensions():
        net.conv.register_forward_hook(extension)
    i = 0
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    y = net(x_conv)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    quantity.track(i)
    model = 'pytorch_conv'
    name = 'forward_cov_condition'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_default_cov_condition():
    setup_seed(42)
    class BNNet(nn.Module):
        def __init__(self):
            super(BNNet, self).__init__()
            self.bn = nn.BatchNorm2d(2)
            self.bn.weight.data.fill_(1.0)
            self.bn.bias.data.zero_()
        
        def forward(self, x):
            return self.bn(x)
    
    net = BNNet()
    batch_size = 1024
    x_bn = torch.randn(batch_size, 2, 4, 4)
    
    # 标准化输入
    mean = x_bn.mean(dim=(0,2,3), keepdim=True)
    std = x_bn.std(dim=(0,2,3), keepdim=True)
    x_bn = (x_bn - mean) / std
    
    quantity = ForwardInputCovCondition(net.bn)
    for extension in quantity.forward_extensions():
        net.bn.register_forward_hook(extension)
    i = 0
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    y = net(x_bn)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    quantity.track(i)
    model = 'pytorch_bn'
    name = 'forward_cov_condition'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_linear_cov_condition_20():
    setup_seed(42)
    l = nn.Linear(2, 3)
    
    # 使用PyTorch的初始化方式
    nn.init.normal_(l.weight, std=0.1)
    nn.init.zeros_(l.bias)
    
    batch_size = 1024
    x_linear = torch.randn(batch_size, 2)
    # 标准化输入
    mean = x_linear.mean(dim=0, keepdim=True)
    std = x_linear.std(dim=0, keepdim=True)
    x_linear = (x_linear - mean) / std
    
    quantity = ForwardInputCovCondition20(l)
    for extension in quantity.forward_extensions():
        l.register_forward_hook(extension)
    i = 0
    optimizer = torch.optim.SGD(l.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    y = l(x_linear)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    quantity.track(i)
    model = 'pytorch_linear'
    name = 'forward_cov_condition_20'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_cov_condition_20():
    setup_seed(42)
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv = nn.Conv2d(1, 2, 3, stride=1, padding=1)
            nn.init.normal_(self.conv.weight, std=0.03)
            nn.init.zeros_(self.conv.bias)
        
        def forward(self, x):
            return self.conv(x)
    
    net = ConvNet()
    batch_size = 1024
    x_conv = torch.randn(batch_size, 1, 3, 3)
    # 标准化输入
    x_conv = x_conv / torch.sqrt((x_conv ** 2).mean(dim=(0,2,3), keepdim=True))
    
    quantity = ForwardInputCovCondition20(net.conv)
    for extension in quantity.forward_extensions():
        net.conv.register_forward_hook(extension)
    i = 0
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    y = net(x_conv)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    quantity.track(i)
    model = 'pytorch_conv'
    name = 'forward_cov_condition_20'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_default_cov_condition_20():
    setup_seed(42)
    class BNNet(nn.Module):
        def __init__(self):
            super(BNNet, self).__init__()
            self.bn = nn.BatchNorm2d(2)
            self.bn.weight.data.fill_(1.0)
            self.bn.bias.data.zero_()
        
        def forward(self, x):
            return self.bn(x)
    
    net = BNNet()
    batch_size = 1024
    x_bn = torch.randn(batch_size, 2, 4, 4)
    
    # 标准化输入
    mean = x_bn.mean(dim=(0,2,3), keepdim=True)
    std = x_bn.std(dim=(0,2,3), keepdim=True)
    x_bn = (x_bn - mean) / std
    
    quantity = ForwardInputCovCondition20(net.bn)
    for extension in quantity.forward_extensions():
        net.bn.register_forward_hook(extension)
    i = 0
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    y = net(x_bn)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    quantity.track(i)
    model = 'pytorch_bn'
    name = 'forward_cov_condition_20'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_cov_condition_50():
    setup_seed(42)
    l = nn.Linear(2, 3)
    
    # 使用PyTorch的初始化方式
    nn.init.normal_(l.weight, std=0.1)
    nn.init.zeros_(l.bias)
    
    batch_size = 1024
    x_linear = torch.randn(batch_size, 2)
    # 标准化输入
    mean = x_linear.mean(dim=0, keepdim=True)
    std = x_linear.std(dim=0, keepdim=True)
    x_linear = (x_linear - mean) / std
    
    quantity = ForwardInputCovCondition50(l)
    for extension in quantity.forward_extensions():
        l.register_forward_hook(extension)
    i = 0
    optimizer = torch.optim.SGD(l.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    y = l(x_linear)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    quantity.track(i)
    model = 'pytorch_linear'
    name = 'forward_cov_condition_50'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_cov_condition_50():
    setup_seed(42)
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv = nn.Conv2d(1, 2, 3, stride=1, padding=1)
            nn.init.normal_(self.conv.weight, std=0.03)
            nn.init.zeros_(self.conv.bias)
        
        def forward(self, x):
            return self.conv(x)
    
    net = ConvNet()
    batch_size = 1024
    x_conv = torch.randn(batch_size, 1, 3, 3)
    # 标准化输入
    x_conv = x_conv / torch.sqrt((x_conv ** 2).mean(dim=(0,2,3), keepdim=True))
    
    quantity = ForwardInputCovCondition50(net.conv)
    for extension in quantity.forward_extensions():
        net.conv.register_forward_hook(extension)
    i = 0
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    y = net(x_conv)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    quantity.track(i)
    model = 'pytorch_conv'
    name = 'forward_cov_condition_50'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_default_cov_condition_50():
    setup_seed(42)
    class BNNet(nn.Module):
        def __init__(self):
            super(BNNet, self).__init__()
            self.bn = nn.BatchNorm2d(2)
            self.bn.weight.data.fill_(1.0)
            self.bn.bias.data.zero_()
        
        def forward(self, x):
            return self.bn(x)
    
    net = BNNet()
    batch_size = 1024
    x_bn = torch.randn(batch_size, 2, 4, 4)
    
    # 标准化输入
    mean = x_bn.mean(dim=(0,2,3), keepdim=True)
    std = x_bn.std(dim=(0,2,3), keepdim=True)
    x_bn = (x_bn - mean) / std
    
    quantity = ForwardInputCovCondition50(net.bn)
    for extension in quantity.forward_extensions():
        net.bn.register_forward_hook(extension)
    i = 0
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    y = net(x_bn)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    quantity.track(i)
    model = 'pytorch_bn'
    name = 'forward_cov_condition_50'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_cov_condition_80():
    setup_seed(42)
    l = nn.Linear(2, 3)
    
    # 使用PyTorch的初始化方式
    nn.init.normal_(l.weight, std=0.1)
    nn.init.zeros_(l.bias)
    
    batch_size = 1024
    x_linear = torch.randn(batch_size, 2)
    # 标准化输入
    mean = x_linear.mean(dim=0, keepdim=True)
    std = x_linear.std(dim=0, keepdim=True)
    x_linear = (x_linear - mean) / std
    
    quantity = ForwardInputCovCondition80(l)
    for extension in quantity.forward_extensions():
        l.register_forward_hook(extension)
    i = 0
    optimizer = torch.optim.SGD(l.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    y = l(x_linear)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    quantity.track(i)
    model = 'pytorch_linear'
    name = 'forward_cov_condition_80'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_cov_condition_80():
    setup_seed(42)
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv = nn.Conv2d(1, 2, 3, stride=1, padding=1)
            nn.init.normal_(self.conv.weight, std=0.03)
            nn.init.zeros_(self.conv.bias)
        
        def forward(self, x):
            return self.conv(x)
    
    net = ConvNet()
    batch_size = 1024
    x_conv = torch.randn(batch_size, 1, 3, 3)
    # 标准化输入
    x_conv = x_conv / torch.sqrt((x_conv ** 2).mean(dim=(0,2,3), keepdim=True))
    
    quantity = ForwardInputCovCondition80(net.conv)
    for extension in quantity.forward_extensions():
        net.conv.register_forward_hook(extension)
    i = 0
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    y = net(x_conv)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    quantity.track(i)
    model = 'pytorch_conv'
    name = 'forward_cov_condition_80'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_default_cov_condition_80():
    setup_seed(42)
    class BNNet(nn.Module):
        def __init__(self):
            super(BNNet, self).__init__()
            self.bn = nn.BatchNorm2d(2)
            self.bn.weight.data.fill_(1.0)
            self.bn.bias.data.zero_()
        
        def forward(self, x):
            return self.bn(x)
    
    net = BNNet()
    batch_size = 1024
    x_bn = torch.randn(batch_size, 2, 4, 4)
    
    # 标准化输入
    mean = x_bn.mean(dim=(0,2,3), keepdim=True)
    std = x_bn.std(dim=(0,2,3), keepdim=True)
    x_bn = (x_bn - mean) / std
    
    quantity = ForwardInputCovCondition80(net.bn)
    for extension in quantity.forward_extensions():
        net.bn.register_forward_hook(extension)
    i = 0
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    y = net(x_bn)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    quantity.track(i)
    model = 'pytorch_bn'
    name = 'forward_cov_condition_80'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_max_eig():
    setup_seed(42)
    l = nn.Linear(2, 3)
    
    # 使用PyTorch的初始化方式
    nn.init.normal_(l.weight, std=0.1)
    nn.init.zeros_(l.bias)
    
    batch_size = 1024
    x_linear = torch.randn(batch_size, 2)
    # 标准化输入
    mean = x_linear.mean(dim=0, keepdim=True)
    std = x_linear.std(dim=0, keepdim=True)
    x_linear = (x_linear - mean) / std
    
    quantity = ForwardInputCovMaxEig(l)
    for extension in quantity.forward_extensions():
        l.register_forward_hook(extension)
    i = 0
    optimizer = torch.optim.SGD(l.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    y = l(x_linear)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    quantity.track(i)
    model = 'pytorch_linear'
    name = 'forward_cov_max_eig'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_max_eig():
    setup_seed(42)
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv = nn.Conv2d(1, 2, 3, stride=1, padding=1)
            nn.init.normal_(self.conv.weight, std=0.03)
            nn.init.zeros_(self.conv.bias)
        
        def forward(self, x):
            return self.conv(x)
    
    net = ConvNet()
    batch_size = 1024
    x_conv = torch.randn(batch_size, 1, 3, 3)
    # 标准化输入
    x_conv = x_conv / torch.sqrt((x_conv ** 2).mean(dim=(0,2,3), keepdim=True))
    
    quantity = ForwardInputCovMaxEig(net.conv)
    for extension in quantity.forward_extensions():
        net.conv.register_forward_hook(extension)
    i = 0
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    y = net(x_conv)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    quantity.track(i)
    model = 'pytorch_conv'
    name = 'forward_cov_max_eig'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_default_max_eig():
    setup_seed(42)
    class BNNet(nn.Module):
        def __init__(self):
            super(BNNet, self).__init__()
            self.bn = nn.BatchNorm2d(2)
            self.bn.weight.data.fill_(1.0)
            self.bn.bias.data.zero_()
        
        def forward(self, x):
            return self.bn(x)
    
    net = BNNet()
    batch_size = 1024
    x_bn = torch.randn(batch_size, 2, 4, 4)
    
    # 标准化输入
    mean = x_bn.mean(dim=(0,2,3), keepdim=True)
    std = x_bn.std(dim=(0,2,3), keepdim=True)
    x_bn = (x_bn - mean) / std
    
    quantity = ForwardInputCovMaxEig(net.bn)
    for extension in quantity.forward_extensions():
        net.bn.register_forward_hook(extension)
    i = 0
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    y = net(x_bn)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    quantity.track(i)
    model = 'pytorch_bn'
    name = 'forward_cov_max_eig'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_stable_rank():
    setup_seed(42)
    l = nn.Linear(2, 3)
    
    # 使用PyTorch的初始化方式
    nn.init.normal_(l.weight, std=0.1)
    nn.init.zeros_(l.bias)
    
    batch_size = 1024
    x_linear = torch.randn(batch_size, 2)
    # 标准化输入
    mean = x_linear.mean(dim=0, keepdim=True)
    std = x_linear.std(dim=0, keepdim=True)
    x_linear = (x_linear - mean) / std
    
    quantity = ForwardInputCovStableRank(l)
    for extension in quantity.forward_extensions():
        l.register_forward_hook(extension)
    i = 0
    optimizer = torch.optim.SGD(l.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    y = l(x_linear)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    quantity.track(i)
    model = 'pytorch_linear'
    name = 'forward_cov_stable_rank'
    print(is_similar_for_stable_rank(quantity.get_output()[0], 1,model,name))

def test_conv_stable_rank():
    setup_seed(42)
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv = nn.Conv2d(1, 2, 3, stride=1, padding=1)
            nn.init.normal_(self.conv.weight, std=0.03)
            nn.init.zeros_(self.conv.bias)
        
        def forward(self, x):
            return self.conv(x)
    
    net = ConvNet()
    batch_size = 1024
    x_conv = torch.randn(batch_size, 1, 3, 3)
    # 标准化输入
    x_conv = x_conv / torch.sqrt((x_conv ** 2).mean(dim=(0,2,3), keepdim=True))
    
    quantity = ForwardInputCovStableRank(net.conv)
    for extension in quantity.forward_extensions():
        net.conv.register_forward_hook(extension)
    i = 0
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    y = net(x_conv)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    quantity.track(i)
    model = 'pytorch_conv'
    name = 'forward_cov_stable_rank'
    print(is_similar_for_stable_rank(quantity.get_output()[0], 1,model,name))

def test_default_stable_rank():
    setup_seed(42)
    class BNNet(nn.Module):
        def __init__(self):
            super(BNNet, self).__init__()
            self.bn = nn.BatchNorm2d(2)
            self.bn.weight.data.fill_(1.0)
            self.bn.bias.data.zero_()
        
        def forward(self, x):
            return self.bn(x)
    
    net = BNNet()
    batch_size = 1024
    x_bn = torch.randn(batch_size, 2, 4, 4)
    
    # 标准化输入
    mean = x_bn.mean(dim=(0,2,3), keepdim=True)
    std = x_bn.std(dim=(0,2,3), keepdim=True)
    x_bn = (x_bn - mean) / std
    
    quantity = ForwardInputCovStableRank(net.bn)
    for extension in quantity.forward_extensions():
        net.bn.register_forward_hook(extension)
    i = 0
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    
    optimizer.zero_grad()
    y = net(x_bn)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    quantity.track(i)
    model = 'pytorch_bn'
    name = 'forward_cov_stable_rank'
    print(is_similar_for_stable_rank(quantity.get_output()[0], 1,model,name))
#运行测试
if __name__ == '__main__':
    test_linear_cov_condition()
    test_conv_cov_condition()
    test_default_cov_condition()
    test_linear_cov_condition_20()
    test_conv_cov_condition_20()
    test_default_cov_condition_20()
    test_linear_cov_condition_50()
    test_conv_cov_condition_50()
    test_default_cov_condition_50()
    test_linear_cov_condition_80()
    test_conv_cov_condition_80()
    test_default_cov_condition_80()
    test_linear_max_eig()
    test_conv_max_eig()
    test_default_max_eig()
    test_linear_stable_rank()
    test_conv_stable_rank()
    test_default_stable_rank()
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

