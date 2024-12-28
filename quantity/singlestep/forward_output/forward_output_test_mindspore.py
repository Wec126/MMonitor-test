import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from MMonitor.quantity.singlestep import *
import numpy as np
import mindspore.dataset as ds
from MMonitor.visualize import LocalVisualization,Visualization
from MMonitor.mmonitor.monitor import Monitor
# 添加更完整的随机种子设置
import random
random.seed(42)
np.random.seed(42)
ms.set_seed(42)
def forward_hook_fn(cell,grad_input,grad_output):
    output = grad_output
    setattr(cell, 'output', output) 
def if_similar(a, b,model,name,tolerance=0.05):
    if not isinstance(a,(int,float)):
        a = a.item()
    print(f'{model}的{name}指标当前计算所得值{a}')
    if not isinstance(b,(int,float)):
        b= b.item()
    print(f"{model}的{name}指标预期值{b}")
    if abs(a - b) <= tolerance:
        return True
    else:
        return False

def test_linear_mean():
    l = nn.Dense(2, 3)
    x_linear = ops.StandardNormal()((4, 2))
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    quantity = ForwardOutputMean(l)
    output_y = None
    handle = l.register_forward_hook(forward_hook_fn)
    
    def forward_fn():
        return l(x_linear)
    
    def train_step(inputs):
        nonlocal output_y
        y = forward_fn()
        output_y = y
        loss = ops.ReduceSum()(y)
        return loss
    
    grad_fn = ops.value_and_grad(train_step, None, optimizer.parameters)
    
    i = 0
    loss, grads = grad_fn(x_linear)
    optimizer(grads)
    quantity.track(i)
    name = 'forward_output_mean'
    model = 'mindspore_linear'
    print(if_similar(quantity.get_output()[0], ops.mean(output_y),model,name))
    handle.remove()
def test_conv_mean():
    conv = nn.Conv2d(1, 2, 3, stride=1, padding=1, pad_mode='pad')
    x_conv = ops.StandardNormal()((4, 1, 3, 3))
    optimizer = nn.SGD(conv.trainable_params(), learning_rate=0.01)
    quantity = ForwardOutputMean(conv)
    handle = conv.register_forward_hook(forward_hook_fn)
    output_y = None
    def forward_fn():
        return conv(x_conv)
    
    def train_step(inputs):
        nonlocal output_y
        y = forward_fn()
        output_y = y
        loss = ops.ReduceSum()(y)
        return loss
    
    grad_fn = ops.value_and_grad(train_step, None, optimizer.parameters)
    
    i = 0
    loss, grads = grad_fn(x_conv)
    optimizer(grads)
    quantity.track(i)
    name = 'forward_output_mean'
    model = 'mindspore_conv'
    print(if_similar(quantity.get_output()[0], ops.mean(output_y),model,name))
    handle.remove()
def test_default_mean():
    # 随机初始化 BatchNorm2d 层，通道数为 3
    batch_size = 8
    channels = 3
    height = 32
    width = 3
    bn = nn.BatchNorm2d(channels)
    # 随机初始化输入张量（形状为 [batch_size, channels, height, width]）
    x_default = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    optimizer = nn.SGD(bn.trainable_params(), learning_rate=0.01)
    quantity = ForwardOutputMean(bn)
    target = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)

    output_y = None    
    handle = bn.register_forward_hook(forward_hook_fn)    
    loss_fn = nn.MSELoss()  # 均方误差
    optimizer = nn.Adam(bn.trainable_params(), learning_rate=0.01)
    # 6. 进行前向和反向传播
    model_with_loss = nn.WithLossCell(bn, loss_fn)
    train_step = nn.TrainOneStepCell(model_with_loss, optimizer)

    for epoch in range(1):  # 单轮训练示例
        loss = train_step(x_default, target)
        output_y = bn(x_default)
        quantity.track(epoch)
        if output_y is not None:
            name = 'forward_output_mean'
            model = 'mindspore_bn'
            print(if_similar(quantity.get_output()[0], ops.mean(output_y),model,name))
    handle.remove()
def test_linear_norm():
    l = nn.Dense(2, 3)
    x_linear = ops.StandardNormal()((4, 2))
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    quantity = ForwardOutputNorm(l)
    output_y = None
    handle = l.register_forward_hook(forward_hook_fn)
    
    def forward_fn():
        return l(x_linear)
    
    def train_step(inputs):
        nonlocal output_y
        y = forward_fn()
        output_y = y
        loss = ops.ReduceSum()(y)
        return loss
    
    grad_fn = ops.value_and_grad(train_step, None, optimizer.parameters)
    
    i = 0
    loss, grads = grad_fn(x_linear)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_linear'
    name = 'forward_output_norm'
    print(if_similar(quantity.get_output()[0], ops.norm(output_y),model,name))
    handle.remove()
def test_conv_norm():
    conv = nn.Conv2d(1, 2, 3, stride=1, padding=1, pad_mode='pad')
    x_conv = ops.StandardNormal()((4, 1, 3, 3))
    optimizer = nn.SGD(conv.trainable_params(), learning_rate=0.01)
    quantity = ForwardOutputNorm(conv)
    handle = conv.register_forward_hook(forward_hook_fn)
    output_y = None
    def forward_fn():
        return conv(x_conv)
    
    def train_step(inputs):
        nonlocal output_y
        y = forward_fn()
        output_y = y
        loss = ops.ReduceSum()(y)
        return loss
    
    grad_fn = ops.value_and_grad(train_step, None, optimizer.parameters)
    
    i = 0
    loss, grads = grad_fn(x_conv)
    optimizer(grads)
    quantity.track(i)
    model = 'mindspore_conv'
    name = 'forward_output_norm'
    print(if_similar(quantity.get_output()[0], ops.norm(output_y),model,name))
    handle.remove()
def test_default_norm():
    # 随机初始化 BatchNorm2d 层，通道数为 3
    batch_size = 8
    channels = 3
    height = 32
    width = 3
    bn = nn.BatchNorm2d(channels)
    # 随机初始化输入张量（形状为 [batch_size, channels, height, width]）
    x_default = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    optimizer = nn.SGD(bn.trainable_params(), learning_rate=0.01)
    quantity = ForwardOutputNorm(bn)
    target = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)

    output_y = None    
    handle = bn.register_forward_hook(forward_hook_fn)    
    loss_fn = nn.MSELoss()  # 均方误差
    optimizer = nn.Adam(bn.trainable_params(), learning_rate=0.01)
    # 6. 进行前向和反向传播
    model_with_loss = nn.WithLossCell(bn, loss_fn)
    train_step = nn.TrainOneStepCell(model_with_loss, optimizer)

    for epoch in range(1):  # 单轮训练示例
        loss = train_step(x_default, target)
        output_y = bn(x_default)
        quantity.track(epoch)
        if output_y is not None:
            model = 'mindspore_linear'
            name = 'forward_output_norm'
            print(if_similar(quantity.get_output()[0], ops.norm(output_y),model,name))
    handle.remove()
def test_linear_std():
    l = nn.Dense(2, 3)
    x_linear = ops.StandardNormal()((4, 2))
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    quantity = ForwardOutputStd(l)
    output_y = None
    handle = l.register_forward_hook(forward_hook_fn)
    
    def forward_fn():
        return l(x_linear)
    
    def train_step(inputs):
        nonlocal output_y
        y = forward_fn()
        output_y = y
        loss = ops.ReduceSum()(y)
        return loss
    
    grad_fn = ops.value_and_grad(train_step, None, optimizer.parameters)
    
    i = 0
    loss, grads = grad_fn(x_linear)
    optimizer(grads)
    quantity.track(i)
    name = 'forward_output_std'
    model = 'mindspore_linear'
    print(if_similar(quantity.get_output()[0], ops.std(output_y),model,name))
    handle.remove()
def test_conv_std():
    conv = nn.Conv2d(1, 2, 3, stride=1, padding=1, pad_mode='pad')
    x_conv = ops.StandardNormal()((4, 1, 3, 3))
    optimizer = nn.SGD(conv.trainable_params(), learning_rate=0.01)
    quantity = ForwardOutputStd(conv)
    handle = conv.register_forward_hook(forward_hook_fn)
    output_y = None
    def forward_fn():
        return conv(x_conv)
    
    def train_step(inputs):
        nonlocal output_y
        y = forward_fn()
        output_y = y
        loss = ops.ReduceSum()(y)
        return loss
    
    grad_fn = ops.value_and_grad(train_step, None, optimizer.parameters)
    
    i = 0
    loss, grads = grad_fn(x_conv)
    optimizer(grads)
    quantity.track(i)
    name = 'forward_output_std'
    model = 'mindspore_conv'
    print(if_similar(quantity.get_output()[0], ops.std(output_y),model,name))
    handle.remove()
def test_default_std():
    # 随机初始化 BatchNorm2d 层，通道数为 3
    batch_size = 8
    channels = 3
    height = 32
    width = 3
    bn = nn.BatchNorm2d(channels)
    # 随机初始化输入张量（形状为 [batch_size, channels, height, width]）
    x_default = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    optimizer = nn.SGD(bn.trainable_params(), learning_rate=0.01)
    quantity = ForwardOutputStd(bn)
    target = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)

    output_y = None    
    handle = bn.register_forward_hook(forward_hook_fn)    
    loss_fn = nn.MSELoss()  # 均方误差
    optimizer = nn.Adam(bn.trainable_params(), learning_rate=0.01)
    # 6. 进行前向和反向传播
    model_with_loss = nn.WithLossCell(bn, loss_fn)
    train_step = nn.TrainOneStepCell(model_with_loss, optimizer)

    for epoch in range(1):  # 单轮训练示例
        loss = train_step(x_default, target)
        output_y = bn(x_default)
        quantity.track(epoch)
        if output_y is not None:
            name = 'forward_output_std'
            model = 'mindspore_bn'
            print(if_similar(quantity.get_output()[0], ops.std(output_y),model,name))
    handle.remove()

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
    model.bn.register_forward_hook(forward_hook_fn)
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