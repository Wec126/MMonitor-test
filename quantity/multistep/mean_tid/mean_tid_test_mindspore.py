import mindspore.dataset as ds
import mindspore as ms
from mindspore import ops
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np
from MMonitor.quantity.multistep import *
from MMonitor.mmonitor.monitor import Monitor
from MMonitor.visualize import Visualization, LocalVisualization 
from MMonitor.mmonitor.monitor import Monitor
ms.context.set_context(mode=ms.context.PYNATIVE_MODE)
def forward_hook_fn(cell,grad_input,grad_output):
    inputs = grad_input[0]
    setattr(cell, 'input', inputs) 
def is_seasonable(a,model,name):
    print(f"{model}的{name}指标的计算值{a}")
    if a < 0:
        return False
    if a > 1:
        return False
    return True
def test_bn():
    # 创建BatchNorm层
    l = nn.BatchNorm2d(3)
    # 使用numpy生成随机数据，然后转换为mindspore tensor
    x = Tensor(np.random.randn(4, 3, 3, 3).astype(np.float32))

    quantity_l = MeanTID(l)

    i = 0
    y = l(x)
    setattr(l, 'input', x)
    quantity_l.track(i)
    model = 'mindspore_bn'
    name = 'mean_tid'
    print(is_seasonable(quantity_l.get_output()[0],model,name))


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
    project = 'BatchNorm'
    localvis = LocalVisualization(project=project)
    localvis.show(monitor, quantity_name=quantity_name, project_name=project)
    print('The result has been saved locally')
def prepare_config():
    config = {'epoch': 100, 'w': 224, 'h': 224, 'class_num': 5, 'len': 100, 'lr': 1e-2}
    config_mmonitor = {
        nn.BatchNorm2d: ['ForwardInputNorm','ForwardInputMean','ForwardInputStd']
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
    test_bn()
    config, config_mmonitor = prepare_config()
    train_model(config, config_mmonitor)