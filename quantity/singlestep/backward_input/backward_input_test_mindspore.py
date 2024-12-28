import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from MMonitor.quantity.singlestep import *
from mindspore import Tensor
import numpy as np
import mindspore.dataset as ds
from MMonitor.visualize import LocalVisualization,Visualization
from MMonitor.mmonitor.monitor import Monitor
# 添加更完整的随机种子设置
import random
random.seed(42)
np.random.seed(42)
ms.set_seed(42)
def backward_hook_fn(cell_id,grad_input,grad_output):
    global input_grad
    input_grad = grad_output[0] #（1，10）
def if_similar(a, b, model,name,tolerance=0.05):
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
def test_dense_mean():
    global input_grad  # 添加全局变量声明
    input_grad = None  # 初始化input_grad
    dense = nn.Dense(10,5)
    quantity = BackwardInputMean(dense)
    handle = dense.register_backward_hook(backward_hook_fn)
    x_linear = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], dtype=ms.float32)
    target = ms.Tensor(np.random.rand(1,5).astype(np.float32))
    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(dense.trainable_params(), learning_rate=0.01)
    def forward_fn(inputs):
        logits = dense(inputs)
        loss = loss_fn(logits, target)
        return loss
    grad_fn = ops.value_and_grad(forward_fn, grad_position=0)
    loss, input_grads = grad_fn(x_linear)
    input_grads_mean = ops.reduce_mean(input_grads)
    setattr(dense, 'input_grad', input_grad) 
    quantity.track(0)
    model = 'mindspore_linear'
    name = 'backward_input_mean'
    print(if_similar(quantity.get_output()[0], input_grads_mean,model,name))
    handle.remove()
def test_conv_mean():
    global input_grad # 添加全局变量声明
    input_grad = None  # 初始化oinput_grad

    # 定义卷积层
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1)

    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(conv.trainable_params(), learning_rate=0.01)

    # 定义指标计算类
    quantity = BackwardInputMean(conv)
    # 注册反向传播钩子
    handle = conv.register_backward_hook(backward_hook_fn)

    # 模拟输入和目标
    x_conv = Tensor(np.random.rand(1, 1, 10, 10).astype(np.float32))  # 单张 10x10 图像
    target = ms.Tensor(np.random.rand(1, 1, 10, 10).astype(np.float32))

    # 定义前向函数
    def forward_fn(inputs):
        logits = conv(inputs)
        loss = loss_fn(logits, target)
        return loss

    # 计算前向值和输入梯度
    grad_fn = ops.value_and_grad(forward_fn, grad_position=0)
    loss, input_grads = grad_fn(x_conv)
    # 手动计算输入梯度的均值
    input_grads_mean = ops.reduce_mean(input_grads)

    # 设置钩子捕获的输入梯度到 `conv` 属性中
    setattr(conv, 'input_grad', input_grad)

    # 记录指标
    quantity.track(0)
    model = 'mindspore_conv'
    name = 'backward_input_mean'
    print(if_similar(quantity.get_output()[0], input_grads_mean.asnumpy(),model,name))

    # 移除钩子
    handle.remove()

def test_default_mean():
    global input_grad  # 添加全局变量声明
    input_grad = None  # 初始化input_grad
    batch_size = 8
    channels = 3
    height = 32
    width = 3
    bn = nn.BatchNorm2d(channels)
    quantity = BackwardInputMean(bn)
    loss_fn = nn.MSELoss()
    x_default = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    # 创建与输入形状相同的目标张量
    target = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    optimizer = nn.Adam(bn.trainable_params(), learning_rate=0.01)
    handle = bn.register_backward_hook(backward_hook_fn)
    def forward_fn(inputs):
        logits = bn(inputs)
        loss = loss_fn(logits, target)
        return loss
    grad_fn = ops.value_and_grad(forward_fn, grad_position=0)
    loss, input_grads = grad_fn(x_default)
    input_grads_mean = ops.reduce_mean(input_grads)
    setattr(bn, 'input_grad', input_grad) 
    quantity.track(0)
    model = 'mindspore_bn'
    name = 'backward_input_mean'
    print(if_similar(quantity.get_output()[0], input_grads_mean,model,name))
    handle.remove()
def test_dense_norm():
    global input_grad  # 添加全局变量声明
    input_grad = None  # 初始化input_grad
    dense = nn.Dense(10,5)
    quantity = BackwardInputNorm(dense)
    handle = dense.register_backward_hook(backward_hook_fn)
    x_linear = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], dtype=ms.float32)
    target = ms.Tensor(np.random.rand(1,5).astype(np.float32))
    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(dense.trainable_params(), learning_rate=0.01)
    def forward_fn(inputs):
        logits = dense(inputs)
        loss = loss_fn(logits, target)
        return loss
    grad_fn = ops.value_and_grad(forward_fn, grad_position=0)
    loss, input_grads = grad_fn(x_linear)
    input_grads_norm = ops.norm(ops.flatten(input_grads))
    setattr(dense, 'input_grad', input_grad) 
    quantity.track(0)
    name = 'backward_input_norm'
    model = 'mindspore_linear'
    print(if_similar(quantity.get_output()[0], input_grads_norm,model,name))
    handle.remove()
def test_conv_norm():
    global input_grad # 添加全局变量声明
    input_grad = None  # 初始化oinput_grad

    # 定义卷积层
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1)

    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(conv.trainable_params(), learning_rate=0.01)

    # 定义指标计算类
    quantity = BackwardInputNorm(conv)
    # 注册反向传播钩子
    handle = conv.register_backward_hook(backward_hook_fn)

    # 模拟输入和目标
    x_conv = Tensor(np.random.rand(1, 1, 10, 10).astype(np.float32))  # 单张 10x10 图像
    target = ms.Tensor(np.random.rand(1, 1, 10, 10).astype(np.float32))

    # 定义前向函数
    def forward_fn(inputs):
        logits = conv(inputs)
        loss = loss_fn(logits, target)
        return loss

    # 计算前向值和输入梯度
    grad_fn = ops.value_and_grad(forward_fn, grad_position=0)
    loss, input_grads = grad_fn(x_conv)
    # 手动计算输入梯度的均值
    input_grads_norm = ops.norm(ops.flatten(input_grads))

    # 设置钩子捕获的输入梯度到 `conv` 属性中
    setattr(conv, 'input_grad', input_grad)

    # 记录指标
    quantity.track(0)
    name = 'backward_input_norm'
    model = 'mindspore_conv'
    print(if_similar(quantity.get_output()[0], input_grads_norm.asnumpy(),model,name))

    # 移除钩子
    handle.remove()

def test_default_norm():
    global input_grad  # 添加全局变量声明
    input_grad = None  # 初始化input_grad
    batch_size = 8
    channels = 3
    height = 32
    width = 3
    bn = nn.BatchNorm2d(channels)
    quantity = BackwardInputNorm(bn)
    loss_fn = nn.MSELoss()
    x_default = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    # 创建与输入形状相同的目标张量
    target = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    optimizer = nn.Adam(bn.trainable_params(), learning_rate=0.01)
    handle = bn.register_backward_hook(backward_hook_fn)
    def forward_fn(inputs):
        logits = bn(inputs)
        loss = loss_fn(logits, target)
        return loss
    grad_fn = ops.value_and_grad(forward_fn, grad_position=0)
    loss, input_grads = grad_fn(x_default)
    input_grads_norm = ops.norm(ops.flatten(input_grads))
    setattr(bn, 'input_grad', input_grad) 
    quantity.track(0)
    name = 'backward_input_norm'
    model = 'mindspore_bn'
    print(if_similar(quantity.get_output()[0], input_grads_norm,model,name))
    handle.remove()
def test_dense_std():
    global input_grad  # 添加全局变量声明
    input_grad = None  # 初始化input_grad
    dense = nn.Dense(10,5)
    quantity = BackwardInputStd(dense)
    handle = dense.register_backward_hook(backward_hook_fn)
    x_linear = Tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]], dtype=ms.float32)
    target = ms.Tensor(np.random.rand(1,5).astype(np.float32))
    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(dense.trainable_params(), learning_rate=0.01)
    def forward_fn(inputs):
        logits = dense(inputs)
        loss = loss_fn(logits, target)
        return loss
    grad_fn = ops.value_and_grad(forward_fn, grad_position=0)
    loss, input_grads = grad_fn(x_linear)
    input_grads_std = ops.std(input_grads)
    setattr(dense, 'input_grad', input_grad) 
    quantity.track(0)
    name = 'backward_input_std'
    model = 'mindspore_linear'
    print(if_similar(quantity.get_output()[0], input_grads_std,model,name))
    handle.remove()
def test_conv_std():
    global input_grad # 添加全局变量声明
    input_grad = None  # 初始化oinput_grad

    # 定义卷积层
    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1)

    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(conv.trainable_params(), learning_rate=0.01)

    # 定义指标计算类
    quantity = BackwardInputStd(conv)
    # 注册反向传播钩子
    handle = conv.register_backward_hook(backward_hook_fn)

    # 模拟输入和目标
    x_conv = Tensor(np.random.rand(1, 1, 10, 10).astype(np.float32))  # 单张 10x10 图像
    target = ms.Tensor(np.random.rand(1, 1, 10, 10).astype(np.float32))

    # 定义前向函数
    def forward_fn(inputs):
        logits = conv(inputs)
        loss = loss_fn(logits, target)
        return loss

    # 计算前向值和输入梯度
    grad_fn = ops.value_and_grad(forward_fn, grad_position=0)
    loss, input_grads = grad_fn(x_conv)
    # 手动计算输入梯度的均值
    input_grads_std = ops.std(input_grads)

    # 设置钩子捕获的输入梯度到 `conv` 属性中
    setattr(conv, 'input_grad', input_grad)

    # 记录指标
    quantity.track(0)
    name = 'backward_input_std'
    model = 'mindspore_conv'
    print(if_similar(quantity.get_output()[0], input_grads_std.asnumpy(),model,name))

    # 移除钩子
    handle.remove()

def test_default_std():
    global input_grad  # 添加全局变量声明
    input_grad = None  # 初始化input_grad
    batch_size = 8
    channels = 3
    height = 32
    width = 3
    bn = nn.BatchNorm2d(channels)
    quantity = BackwardInputStd(bn)
    loss_fn = nn.MSELoss()
    x_default = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    # 创建与输入形状相同的目标张量
    target = ms.Tensor(np.random.randn(batch_size, channels, height, width), dtype=ms.float32)
    optimizer = nn.Adam(bn.trainable_params(), learning_rate=0.01)
    handle = bn.register_backward_hook(backward_hook_fn)
    def forward_fn(inputs):
        logits = bn(inputs)
        loss = loss_fn(logits, target)
        return loss
    grad_fn = ops.value_and_grad(forward_fn, grad_position=0)
    loss, input_grads = grad_fn(x_default)
    input_grads_std = ops.std(input_grads)
    setattr(bn, 'input_grad', input_grad) 
    quantity.track(0)
    name = 'backward_input_std'
    model = 'mindspore_bn'
    print(if_similar(quantity.get_output()[0], input_grads_std,model,name))
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
        nn.BatchNorm2d: ['BackwardInputNorm','BackwardInputMean']
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
    model.bn.register_backward_hook(backward_hook_fn)
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
        setattr(model.bn, 'input_grad', input_grad)
        monitor.track(t)
        logs = vis.show(t)
    
    # 输出监控结果
    print("Training completed.")
    show_local(monitor)



if __name__ == "__main__":
    test_dense_mean()
    test_conv_mean()
    test_default_mean()
    test_dense_norm()
    test_conv_norm()
    test_default_norm()
    test_dense_std()
    test_conv_std()
    test_default_std()
    config, config_mmonitor = prepare_config()
    train_model(config, config_mmonitor)
