import mindspore as ms
from MMonitor.quantity.singlestep import *
import numpy as np
import mindspore.nn as nn
from mindspore import ops
import mindspore.dataset as ds
from MMonitor.visualize import LocalVisualization,Visualization
from MMonitor.mmonitor.monitor import Monitor
# 添加更完整的随机种子设置
import random
random.seed(42)
np.random.seed(42)
ms.set_seed(42)
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
    l = nn.Dense(2, 3)  # Linear -> Dense
    x_linear = ops.StandardNormal()((4, 2))  # 使用mindspore的随机张量生成
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    quantity = WeightMean(l)
    
    def forward_fn(x):
        return l(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = l(x_linear)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_linear)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_linear'
    name = 'weight_mean'
    weight_mean_direct = ops.reduce_mean(l.weight).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_mean_direct,model,name))
def test_conv_mean():
    # 使用正态分布初始化
    cov = nn.Conv2d(1, 2, 3, pad_mode='pad', padding=1)
    x_conv = ops.StandardNormal()((4, 1, 3, 3))  # 使用mindspore的随机张量生成
    optimizer = nn.SGD(cov.trainable_params(), learning_rate=0.01)
    quantity = WeightMean(cov)
    
    def forward_fn(x):
        return cov(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = cov(x_conv)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_conv)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_conv'
    name = 'weight_mean'
    weight_mean_direct = ops.reduce_mean(cov.weight).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_mean_direct,model,name))
    
def test_default_mean():
    x_default = ms.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    bn = nn.BatchNorm1d(2)  # 输入特征数为 2
    if hasattr(bn, 'weight'):
        data = bn.weight
    elif hasattr(bn, 'gamma'):
        data = bn.gamma
    optimizer = nn.SGD(bn.trainable_params(), learning_rate=0.01)
    quantity = WeightMean(bn)
    
    def forward_fn(x):
        return bn(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = bn(x_default)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_default)
    setattr(bn, 'weight', data)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_bn'
    name = 'weight_mean'
    weight_mean_direct = ops.reduce_mean(bn.gamma).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_mean_direct,model,name))

def test_linear_norm():
    # 1. 判断初始化方法下的权重均值
    # 使用默认初始化 -> 判断是否在-0.1~0.1之间
    l = nn.Dense(2, 3)  # Linear -> Dense
    x_linear = ops.StandardNormal()((4, 2))  # 使用mindspore的随机张量生成
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    quantity = WeightNorm(l)
    
    def forward_fn(x):
        return l(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = l(x_linear)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_linear)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_linear'
    name = 'weight_norm'
    weight_norm_direct = ops.norm(l.weight).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_norm_direct,model,name))
def test_conv_norm():
    # 使用正态分布初始化
    cov = nn.Conv2d(1, 2, 3, pad_mode='pad', padding=1)
    x_conv = ops.StandardNormal()((4, 1, 3, 3))  # 使用mindspore的随机张量生成
    optimizer = nn.SGD(cov.trainable_params(), learning_rate=0.01)
    quantity = WeightNorm(cov)
    
    def forward_fn(x):
        return cov(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = cov(x_conv)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_conv)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_conv'
    name = 'weight_norm'
    weight_norm_direct = ops.norm(cov.weight).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_norm_direct,model,name))
    
def test_default_norm():
    x_default = ms.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    bn = nn.BatchNorm1d(2)  # 输入特征数为 2
    if hasattr(bn, 'weight'):
        data = bn.weight
    elif hasattr(bn, 'gamma'):
        data = bn.gamma
    optimizer = nn.SGD(bn.trainable_params(), learning_rate=0.01)
    quantity = WeightNorm(bn)
    
    def forward_fn(x):
        return bn(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = bn(x_default)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_default)
    setattr(bn, 'weight', data)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_bn'
    name = 'weight_norm'
    
    weight_norm_direct = ops.norm(bn.weight).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_norm_direct,model,name))

def test_linear_std():
    # 1. 判断初始化方法下的权重均值
    # 使用默认初始化 -> 判断是否在-0.1~0.1之间
    l = nn.Dense(2, 3)  # Linear -> Dense
    x_linear = ops.StandardNormal()((4, 2))  # 使用mindspore的随机张量生成
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    quantity = WeightStd(l)
    
    def forward_fn(x):
        return l(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = l(x_linear)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_linear)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_linear'
    name = 'weight_std'
    weight_std_direct = ops.std(l.weight).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_std_direct,model,name))
def test_conv_std():
    # 使用正态分布初始化
    cov = nn.Conv2d(1, 2, 3, pad_mode='pad', padding=1)
    x_conv = ops.StandardNormal()((4, 1, 3, 3))  # 使用mindspore的随机张量生成
    optimizer = nn.SGD(cov.trainable_params(), learning_rate=0.01)
    quantity = WeightStd(cov)
    
    def forward_fn(x):
        return cov(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = cov(x_conv)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_conv)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_conv'
    name = 'weight_std'
    weight_std_direct = ops.std(cov.weight).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_std_direct,model,name))
    
def test_default_std():
    x_default = ms.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    bn = nn.BatchNorm1d(2)  # 输入特征数为 2
    if hasattr(bn, 'weight'):
        data = bn.weight
    elif hasattr(bn, 'gamma'):
        data = bn.gamma
    optimizer = nn.SGD(bn.trainable_params(), learning_rate=0.01)
    quantity = WeightStd(bn)
    
    def forward_fn(x):
        return bn(x)
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters)
    
    i = 0
    y = bn(x_default)
    loss = ops.reduce_sum(y)
    loss_value, grads = grad_fn(x_default)
    setattr(bn, 'weight', data)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_bn'
    name = 'weight_std'
    
    weight_std_direct = ops.std(bn.weight).asnumpy().item()
    print(if_similar(quantity.get_output()[0], weight_std_direct,model,name))

class Model(nn.Cell):
    def __init__(self, w, h, class_num, input_channels=1):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=16, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.fc = nn.Dense(16 * w * h, class_num)

    def construct(self, x):
        r1 = self.relu1(self.bn(self.conv1(x)))  # 卷积 + BatchNorm + ReLU
        return self.fc(r1.view(r1.shape[0], -1))  # Flatten 后输入全连接层


def show_local(monitor, quantity_name=None):
    project = 'BatchNorm2d'
    localvis = LocalVisualization(project=project)
    localvis.show(monitor, quantity_name=quantity_name, project_name=project)
    print('The result has been saved locally')

def prepare_config():
    config = {'epoch': 100, 'w': 224, 'h': 224, 'class_num': 5, 'len': 100, 'lr': 1e-2}
    config_mmonitor = {
        nn.BatchNorm2d: ['ForwardOutputNorm','ForwardOutputMean','ForwardOutputStd']
    }
    return config, config_mmonitor
def prepare_data(width, height, class_num, data_len):
    """
    准备数据集
    """
    # 创建随机数据用于示例
    x_data = np.random.randn(data_len, 3, height, width).astype(np.float32)
    y_data = np.random.randint(0, class_num, (data_len,)).astype(np.int32)

    # 如果你希望将输入数据转为 RGB 形式
    x_data = np.repeat(x_data, 3, axis=1)  # 复制通道维度，变为 [length, 3, 28, 28]

    # 创建 MindSpore 数据集
    dataset_generator = [(x_data[i], y_data[i]) for i in range(data_len)]
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"])
    dataset = dataset.batch(32)
    
    return dataset

def prepare_optimizer(model, learning_rate):
    """
    准备优化器
    """
    return nn.Adam(model.trainable_params(), learning_rate=learning_rate)

def prepare_loss_func():
    """
    准备损失函数
    """
    return nn.CrossEntropyLoss()



def train_model(config, config_mmonitor):
    
    # 准备数据
    dataset = prepare_data(config['w'], config['h'], config['class_num'], config['len'])
    
    # 创建模型、优化器和损失函数
    model = Model(config['w'], config['h'], config['class_num'])
    optimizer = prepare_optimizer(model, config['lr'])
    loss_fn = prepare_loss_func()
    
    # 创建监控和可视化工具
    monitor = Monitor(model, config_mmonitor)
    vis = Visualization(monitor, 
                       project=list(config_mmonitor.keys()), 
                       name=list(config_mmonitor.values()))
    
    # 定义前向传播函数
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits
    
    # 获取梯度函数
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    #model.bn.register_forward_hook(forward_hook_fn)
    # 定义单步训练函数
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        return loss
    
    # 训练循环
    def train_loop():
        model.set_train()
        
        for step, (data, label) in enumerate(dataset.create_tuple_iterator()):
            loss = train_step(data, label)
        return loss
    
    # 执行训练
    t = 0
    for t in range(10):
        #monitor.track(t)
        print(f"Epoch {t+1}\n-------------------------------")
        loss = train_loop()
        print(f"当前loss为{loss}")
        monitor.track(t)
        logs = vis.show(t)
    
    # 输出监控结果
    print("Training completed.")
    show_local(monitor)
if __name__ == "__main__":
    test_linear_mean()
    test_conv_mean()
    test_default_mean()
    test_linear_norm()
    test_conv_norm()
    test_default_norm()
    test_linear_std()
    test_conv_std()
    test_default_std()
    config, config_mmonitor = prepare_config()
    train_model(config, config_mmonitor)