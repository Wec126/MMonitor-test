import numpy as np
import jittor as jt
from jittor import nn
from MMonitor.quantity.multistep import *
from MMonitor.mmonitor.monitor import Monitor
from MMonitor.visualize import Visualization, LocalVisualization
def is_seasonable(a,model,name):
    print(f"{model}的{name}指标的计算值{a.item()}")
    if a < 0 :
        return False
    if a > 1 :
        return False
    return True
def test_bn():
    l = nn.BatchNorm(3)
    x = jt.randn((4, 3, 3))  


    quantity_l = MeanTID(l)


    for hook in quantity_l.forward_extensions():
        l.register_forward_hook(hook)


    i = 0
    y = l(x)
    quantity_l.track(i)
    model = 'jittor_bn'
    name = 'mean_tid'
    print(is_seasonable(quantity_l.get_output()[0],model,name))
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
    project = 'BatchNorm'
    localvis = LocalVisualization(project=project)
    localvis.show(monitor, quantity_name=quantity_name, project_name=project)
    print('The result has been saved locally')

def prepare_config():
    config = {'epoch': 100, 'w': 224, 'h': 224, 'class_num': 5, 'len': 100, 'lr': 1e-2}
    config_mmonitor = {
        nn.BatchNorm2d: ['MeanTID']
    }
    return config, config_mmonitor

def prepare_loss_func():
    return nn.CrossEntropyLoss()
if __name__ == '__main__':
    # 主训练循环
    test_bn()
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
        opt.step(loss)

        monitor.track(epoch)
        vis.show(epoch)
        print(f"Epoch {epoch} loss: {loss}")
    # 输出监控结果并显示
    print("Training completed.")
    show_local(monitor)