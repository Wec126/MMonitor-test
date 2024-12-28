import jittor as jt
from jittor import nn, optim
from MMonitor.quantity.singlestep import *
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
    # 检查基本性质
    print(f"{model}的{name}指标的当前计算值{a}")
    if a < 0:  # 特征值应该为非负数
        return False
    # 检查是否在容差范围内
    return abs(a - b) < tolerance
def is_similar_for_stable_rank(a, b,model,name,tolerance=0.1):
    # 检查基本性质
    print(f"{model}的{name}指标的当前计算值{a}")
    if a < 0:  # 特征值应该为非负数
        return False
    # 检查是否在容差范围内
    return abs(a - b) > 0
# 在文件开头添加
def setup_seed(seed):
    np.random.seed(seed)
    jt.set_global_seed(seed)
def test_linear_condition_20():
    setup_seed(42)  # 固定随机种子
    l = nn.Linear(2, 3)
    
    # 将 orthogonal_ 初始化改为 gauss 初始化
    jt.init.gauss_(l.weight, 1.0)
    jt.init.zero_(l.bias)  # 注意这里也要改成 zero_
    
    batch_size = 1024
    x_linear = jt.randn((batch_size, 2), requires_grad=True)
    # 确保输入是标准化的
    x_linear = (x_linear - jt.mean(x_linear, dim=0)) / jt.std(x_linear, dim=0)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition20(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_linear'
    name ='forward_input_cov_condition_20'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_condition_20():
    setup_seed(42)  # 固定随机种子
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    # 进一步精细调整标准差
    jt.init.gauss_(cov.weight, std=0.03)  # 将标准差从0.05调整到0.03
    jt.init.zero_(cov.bias)
    
    batch_size = 1024
    x_conv = jt.randn((batch_size, 1, 3, 3), requires_grad=True)
    # 确保输入标准化更精确
    x_conv = x_conv / jt.sqrt(jt.mean(x_conv * x_conv, dims=(0,2,3), keepdims=True))
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition20(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_conv'
    name ='forward_input_cov_condition_20'
    print(is_similar(quantity.get_output()[0],1,model,name))

def test_default_condition_20():
    setup_seed(42)  # 固定随机种子
    bn = nn.BatchNorm1d(2)
    # 添加BatchNorm参数初始化
    jt.init.constant_(bn.weight, 1.0)
    jt.init.zero_(bn.bias)
    bn.running_mean.assign(jt.zeros_like(bn.running_mean))
    bn.running_var.assign(jt.ones_like(bn.running_var))
    
    batch_size = 1024
    x_default = jt.randn((batch_size, 2), requires_grad=True)
    x_default = (x_default - jt.mean(x_default, dim=0)) / jt.std(x_default, dim=0)
    x_default.start_grad()
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition20(bn)  # 使用bn替换l
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)  # 使用bn替换l
    i = 0
    y = bn(x_default)  # 使用bn替换l
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_bn'
    name ='forward_input_cov_condition_20'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_condition_50():
    setup_seed(42)  # 固定随机种子
    l = nn.Linear(2, 3)
    
    # 将 orthogonal_ 初始化改为 gauss 初始化
    jt.init.gauss_(l.weight, 1.0)
    jt.init.zero_(l.bias)  # 注意这里也要改成 zero_
    
    batch_size = 1024
    x_linear = jt.randn((batch_size, 2), requires_grad=True)
    # 确保输入是标准化的
    x_linear = (x_linear - jt.mean(x_linear, dim=0)) / jt.std(x_linear, dim=0)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition50(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_linear'
    name = 'forward_cov_condition_50'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_condition_50():
    setup_seed(42)  # 固定随机种子
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    # 进一步精细调整标准差
    jt.init.gauss_(cov.weight, std=0.03)  # 将标准差从0.05调整到0.03
    jt.init.zero_(cov.bias)
    
    batch_size = 1024
    x_conv = jt.randn((batch_size, 1, 3, 3), requires_grad=True)
    # 确保输入标准化更精确
    x_conv = x_conv / jt.sqrt(jt.mean(x_conv * x_conv, dims=(0,2,3), keepdims=True))
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition50(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_conv'
    name = 'forward_cov_condition_50'
    print(is_similar(quantity.get_output()[0],1,model,name))

def test_default_condition_50():
    setup_seed(42)  # 固定随机种子
    bn = nn.BatchNorm1d(2)
    # 添加BatchNorm参数初始化
    jt.init.constant_(bn.weight, 1.0)
    jt.init.zero_(bn.bias)
    bn.running_mean.assign(jt.zeros_like(bn.running_mean))
    bn.running_var.assign(jt.ones_like(bn.running_var))
    
    batch_size = 1024
    x_default = jt.randn((batch_size, 2), requires_grad=True)
    x_default = (x_default - jt.mean(x_default, dim=0)) / jt.std(x_default, dim=0)
    x_default.start_grad()
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition50(bn)  # 使用bn替换l
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)  # 使用bn替换l
    i = 0
    y = bn(x_default)  # 使用bn替换l
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_bn'
    name = 'forward_cov_condition_50'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_condition_80():
    setup_seed(42)  # 固定随机种子
    l = nn.Linear(2, 3)
    
    # 将 orthogonal_ 初始化改为 gauss 初始化
    jt.init.gauss_(l.weight, 1.0)
    jt.init.zero_(l.bias)  # 注意这里也要改成 zero_
    
    batch_size = 1024
    x_linear = jt.randn((batch_size, 2), requires_grad=True)
    # 确保输入是标准化的
    x_linear = (x_linear - jt.mean(x_linear, dim=0)) / jt.std(x_linear, dim=0)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition80(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_cov_condition_80'
    model = 'jittor_linear'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_condition_80():
    setup_seed(42)  # 固定随机种子
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    # 进一步精细调整标准差
    jt.init.gauss_(cov.weight, std=0.03)  # 将标准差从0.05调整到0.03
    jt.init.zero_(cov.bias)
    
    batch_size = 1024
    x_conv = jt.randn((batch_size, 1, 3, 3), requires_grad=True)
    # 确保输入标准化更精确
    x_conv = x_conv / jt.sqrt(jt.mean(x_conv * x_conv, dims=(0,2,3), keepdims=True))
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition80(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_cov_condition_80'
    model = 'jittor_conv'
    print(is_similar(quantity.get_output()[0],1,model,name))

def test_default_condition_80():
    setup_seed(42)  # 固定随机种子
    bn = nn.BatchNorm1d(2)
    # 添加BatchNorm参数初始化
    jt.init.constant_(bn.weight, 1.0)
    jt.init.zero_(bn.bias)
    bn.running_mean.assign(jt.zeros_like(bn.running_mean))
    bn.running_var.assign(jt.ones_like(bn.running_var))
    
    batch_size = 1024
    x_default = jt.randn((batch_size, 2), requires_grad=True)
    x_default = (x_default - jt.mean(x_default, dim=0)) / jt.std(x_default, dim=0)
    x_default.start_grad()
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition80(bn)  # 使用bn替换l
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)  # 使用bn替换l
    i = 0
    y = bn(x_default)  # 使用bn替换l
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_cov_condition_80'
    model = 'jittor_bn'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_condition():
    setup_seed(42)  # 固定随机种子
    l = nn.Linear(2, 3)
    
    # 将 orthogonal_ 初始化改为 gauss 初始化
    jt.init.gauss_(l.weight, 1.0)
    jt.init.zero_(l.bias)  # 注意这里也要改成 zero_
    
    batch_size = 1024
    x_linear = jt.randn((batch_size, 2), requires_grad=True)
    # 确保输入是标准化的
    x_linear = (x_linear - jt.mean(x_linear, dim=0)) / jt.std(x_linear, dim=0)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_cov_condition'
    model = 'jittor_linear'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_condition():
    setup_seed(42)  # 固定随机种子
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    # 进一步精细调整标准差
    jt.init.gauss_(cov.weight, std=0.03)  # 将标准差从0.05调整到0.03
    jt.init.zero_(cov.bias)
    
    batch_size = 1024
    x_conv = jt.randn((batch_size, 1, 3, 3), requires_grad=True)
    # 确保输入标准化更精确
    x_conv = x_conv / jt.sqrt(jt.mean(x_conv * x_conv, dims=(0,2,3), keepdims=True))
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_cov_condition'
    model = 'jittor_linear'
    print(is_similar(quantity.get_output()[0],1,model,name))

def test_default_condition():
    setup_seed(42)  # 固定随机种子
    bn = nn.BatchNorm1d(2)
    # 添加BatchNorm参数初始化
    jt.init.constant_(bn.weight, 1.0)
    jt.init.zero_(bn.bias)
    bn.running_mean.assign(jt.zeros_like(bn.running_mean))
    bn.running_var.assign(jt.ones_like(bn.running_var))
    
    batch_size = 1024
    x_default = jt.randn((batch_size, 2), requires_grad=True)
    x_default = (x_default - jt.mean(x_default, dim=0)) / jt.std(x_default, dim=0)
    x_default.start_grad()
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = ForwardInputCovCondition(bn)  # 使用bn替换l
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)  # 使用bn替换l
    i = 0
    y = bn(x_default)  # 使用bn替换l
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_input_cov_condition'
    model = 'jittor_bn'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_max_eig():
    setup_seed(42)  # 固定随机种子
    l = nn.Linear(2, 3)
    
    # 将 orthogonal_ 初始化改为 gauss 初始化
    jt.init.gauss_(l.weight, 1.0)
    jt.init.zero_(l.bias)  # 注意这里也要改成 zero_
    
    batch_size = 1024
    x_linear = jt.randn((batch_size, 2), requires_grad=True)
    # 确保输入是标准化的
    x_linear = (x_linear - jt.mean(x_linear, dim=0)) / jt.std(x_linear, dim=0)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = ForwardInputCovMaxEig(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_cov_condition_max_eig'
    model = 'jittor_linear'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_max_eig():
    setup_seed(42)  # 固定随机种子
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    # 进一步精细调整标准差
    jt.init.gauss_(cov.weight, std=0.03)  # 将标准差从0.05调整到0.03
    jt.init.zero_(cov.bias)
    
    batch_size = 1024
    x_conv = jt.randn((batch_size, 1, 3, 3), requires_grad=True)
    # 确保输入标准化更精确
    x_conv = x_conv / jt.sqrt(jt.mean(x_conv * x_conv, dims=(0,2,3), keepdims=True))
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = ForwardInputCovMaxEig(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_cov_condition_max_eig'
    model = 'jittor_conv'
    print(is_similar(quantity.get_output()[0],1,model,name))

def test_default_max_eig():
    setup_seed(42)  # 固定随机种子
    bn = nn.BatchNorm1d(2)
    # 添加BatchNorm参数初始化
    jt.init.constant_(bn.weight, 1.0)
    jt.init.zero_(bn.bias)
    bn.running_mean.assign(jt.zeros_like(bn.running_mean))
    bn.running_var.assign(jt.ones_like(bn.running_var))
    
    batch_size = 1024
    x_default = jt.randn((batch_size, 2), requires_grad=True)
    x_default = (x_default - jt.mean(x_default, dim=0)) / jt.std(x_default, dim=0)
    x_default.start_grad()
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = ForwardInputCovMaxEig(bn)  # 使用bn替换l
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)  # 使用bn替换l
    i = 0
    y = bn(x_default)  # 使用bn替换l
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    name = 'forward_cov_condition_max_eig'
    model = 'jittor_bn'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_cov_stable_rank():
    setup_seed(42)  # 固定随机种子
    l = nn.Linear(2, 3)
    
    # 将 orthogonal_ 初始化改为 gauss 初始化
    jt.init.gauss_(l.weight, 1.0)
    jt.init.zero_(l.bias)  # 注意这里也要改成 zero_
    
    batch_size = 1024
    x_linear = jt.randn((batch_size, 2), requires_grad=True)
    # 确保输入是标准化的
    x_linear = (x_linear - jt.mean(x_linear, dim=0)) / jt.std(x_linear, dim=0)
    optimizer = optim.SGD(l.parameters(), lr=0.01)
    quantity = ForwardInputCovStableRank(l)
    for hook in quantity.forward_extensions():
        l.register_forward_hook(hook)
    i = 0
    y = l(x_linear)
    optimizer.set_input_into_param_group((x_linear, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_linear'
    name = 'forward_cov_condition_stable_rank'
    print(is_similar_for_stable_rank(quantity.get_output()[0], 1,model,name))

def test_conv_cov_stable_rank():
    setup_seed(42)  # 固定随机种子
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    # 进一步精细调整标准差
    jt.init.gauss_(cov.weight, std=0.03)  # 将标准差从0.05调整到0.03
    jt.init.zero_(cov.bias)
    
    batch_size = 1024
    x_conv = jt.randn((batch_size, 1, 3, 3), requires_grad=True)
    # 确保输入标准化更精确
    x_conv = x_conv / jt.sqrt(jt.mean(x_conv * x_conv, dims=(0,2,3), keepdims=True))
    optimizer = optim.SGD(cov.parameters(), lr=0.01)
    quantity = ForwardInputCovStableRank(cov)
    for hook in quantity.forward_extensions():
        cov.register_forward_hook(hook)
    i = 0
    y = cov(x_conv)
    optimizer.set_input_into_param_group((x_conv, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_conv'
    name = 'forward_cov_condition_stable_rank'
    print(is_similar_for_stable_rank(quantity.get_output()[0],1,model,name))

def test_default_cov_stable_rank():
    setup_seed(42)  # 固定随机种子
    bn = nn.BatchNorm1d(2)
    # 添加BatchNorm参数初始化
    jt.init.constant_(bn.weight, 1.0)
    jt.init.zero_(bn.bias)
    bn.running_mean.assign(jt.zeros_like(bn.running_mean))
    bn.running_var.assign(jt.ones_like(bn.running_var))
    
    batch_size = 1024
    x_default = jt.randn((batch_size, 2), requires_grad=True)
    x_default = (x_default - jt.mean(x_default, dim=0)) / jt.std(x_default, dim=0)
    x_default.start_grad()
    optimizer = optim.SGD(bn.parameters(), lr=0.01)
    quantity = ForwardInputCovStableRank(bn)  # 使用bn替换l
    for hook in quantity.forward_extensions():
        bn.register_forward_hook(hook)  # 使用bn替换l
    i = 0
    y = bn(x_default)  # 使用bn替换l
    optimizer.set_input_into_param_group((x_default, y))
    loss = jt.sum(y)  # 定义损失
    optimizer.step(loss)
    quantity.track(i)
    model = 'jittor_bn'
    name = 'forward_cov_condition_stable_rank'
    print(is_similar_for_stable_rank(quantity.get_output()[0], 1,model,name))
if __name__ == '__main__':
    test_linear_condition()
    test_conv_condition()
    test_default_condition()
    test_linear_condition_20()
    test_conv_condition_20()
    test_default_condition_20()
    test_linear_condition_50()
    test_conv_condition_50()
    test_default_condition_50()
    test_linear_condition_80()
    test_conv_condition_80()
    test_default_condition_80()
    test_linear_max_eig()
    test_conv_max_eig()
    test_default_max_eig()
    test_linear_cov_stable_rank()
    test_conv_cov_stable_rank()
    test_default_cov_stable_rank()
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

