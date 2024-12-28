import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Normal, Zero, Constant, initializer
from MMonitor.quantity.singlestep import *
import numpy as np
import mindspore.dataset as ds
from MMonitor.visualize import LocalVisualization,Visualization
from MMonitor.mmonitor.monitor import Monitor
def is_similar_for_stable_rank(a, b,model,name,tolerance=0.1):
    # 检查基本性质
    print(f"{model}的{name}指标的当前计算值{a}")
    if a < 0:  # 特征值应该为非负数
        return False
    # 检查是否在容差范围内
    return abs(a - b) > 0
def is_similar(a, b, model,name,tolerance=0.1):
    # 检查基本性质
    print(f"{model}的{name}指标的当前计算值{a}")
    if a < 0:  # 特征值应该为非负数
        return False
    # 检查是否在容差范围内
    return abs(a - b) < tolerance
# 在文件开头添加
def setup_seed(seed):
    np.random.seed(seed)
    ms.set_seed(seed)

def test_linear_cov_conition():
    setup_seed(42)  # 固定随机种子
    l = nn.Dense(2, 3)  # MindSpore使用Dense替代Linear
    
    # 使用MindSpore的初始化方式
    l.weight.set_data(initializer(Normal(0.1), l.weight.shape))
    l.bias.set_data(initializer(Zero(), l.bias.shape))
    
    batch_size = 1024
    x_linear = ops.StandardNormal()((batch_size, 2))
    # 确保输入是标准化的
    mean = ops.ReduceMean(keep_dims=True)(x_linear, 0)
    std = ops.std(x_linear, 0)
    x_linear = (x_linear - mean) / std
    
    quantity = ForwardInputCovCondition(l)
    
    i = 0
    def forward_fn(x):
        y = l(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, l.trainable_params())
    
    loss, grads = grad_fn(x_linear)
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(l, "input", x_linear)
    quantity.track(i)
    model = 'mindspore_dense'
    name = 'forward_cov_condition'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_cov_condition():
    setup_seed(42)
    class ConvNet(nn.Cell):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv = nn.Conv2d(1, 2, 3, stride=1, padding=1, pad_mode='pad',
                                weight_init=Normal(0.03), bias_init=Zero())
        
        def construct(self, x):
            return self.conv(x)
    
    net = ConvNet()
    batch_size = 1024
    x_conv = ops.StandardNormal()((batch_size, 1, 3, 3))
    # 标准化输入
    x_conv = x_conv / ops.sqrt(ops.ReduceMean(keep_dims=True)(x_conv * x_conv, (0,2,3)))
    
    quantity = ForwardInputCovCondition(net.conv)
    
    i = 0
    def forward_fn(x):
        y = net(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params())
    
    loss, grads = grad_fn(x_conv)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(net.conv, "input", x_conv)
    quantity.track(i)
    model = 'mindspore_conv'
    name = 'forward_cov_condition'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_default_cov_condition():
    setup_seed(42)
    class BNNet(nn.Cell):
        def __init__(self):
            super(BNNet, self).__init__()
            self.bn = nn.BatchNorm2d(2, gamma_init=Constant(1.0), beta_init=Zero(),
                                   use_batch_statistics=True)
        
        def construct(self, x):
            return self.bn(x)
    
    net = BNNet()
    batch_size = 1024
    # 创建4D输入: (batch_size, channels, height, width)
    x_bn = ops.StandardNormal()((batch_size, 2, 4, 4))
    
    # 标准化输入
    mean = ops.ReduceMean(keep_dims=True)(x_bn, (0, 2, 3))
    std = ops.std(x_bn, (0, 2, 3), keepdims=True)
    x_bn = (x_bn - mean) / std
    
    quantity = ForwardInputCovCondition(net.bn)
    
    i = 0
    def forward_fn(x):
        y = net(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params())
    
    loss, grads = grad_fn(x_bn)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(net.bn, "input", x_bn)
    quantity.track(i)
    model = 'mindspore_bn'
    name = 'forward_cov_condition'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_condition_20():
    setup_seed(42)  # 固定随机种子
    l = nn.Dense(2, 3)  # MindSpore使用Dense替代Linear
    
    # 使用MindSpore的初始化方式
    l.weight.set_data(initializer(Normal(0.1), l.weight.shape))
    l.bias.set_data(initializer(Zero(), l.bias.shape))
    
    batch_size = 1024
    x_linear = ops.StandardNormal()((batch_size, 2))
    # 确保输入是标准化的
    mean = ops.ReduceMean(keep_dims=True)(x_linear, 0)
    std = ops.std(x_linear, 0)
    x_linear = (x_linear - mean) / std
    
    quantity = ForwardInputCovCondition20(l)
    
    i = 0
    def forward_fn(x):
        y = l(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, l.trainable_params())
    
    loss, grads = grad_fn(x_linear)
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(l, "input", x_linear)
    quantity.track(i)
    model = 'mindspore_dese'
    name = 'forward_cov_condition_20'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_condition_20():
    setup_seed(42)
    class ConvNet(nn.Cell):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv = nn.Conv2d(1, 2, 3, stride=1, padding=1, pad_mode='pad',
                                weight_init=Normal(0.03), bias_init=Zero())
        
        def construct(self, x):
            return self.conv(x)
    
    net = ConvNet()
    batch_size = 1024
    x_conv = ops.StandardNormal()((batch_size, 1, 3, 3))
    # 标准化输入
    x_conv = x_conv / ops.sqrt(ops.ReduceMean(keep_dims=True)(x_conv * x_conv, (0,2,3)))
    
    quantity = ForwardInputCovCondition20(net.conv)
    
    i = 0
    def forward_fn(x):
        y = net(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params())
    
    loss, grads = grad_fn(x_conv)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(net.conv, "input", x_conv)
    quantity.track(i)
    model = 'mindspore_conv'
    name = 'forward_cov_condition_20'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_default_condition_20():
    setup_seed(42)
    class BNNet(nn.Cell):
        def __init__(self):
            super(BNNet, self).__init__()
            self.bn = nn.BatchNorm2d(2, gamma_init=Constant(1.0), beta_init=Zero(),
                                   use_batch_statistics=True)
        
        def construct(self, x):
            return self.bn(x)
    
    net = BNNet()
    batch_size = 1024
    # 创建4D输入: (batch_size, channels, height, width)
    x_bn = ops.StandardNormal()((batch_size, 2, 4, 4))
    
    # 标准化输入
    mean = ops.ReduceMean(keep_dims=True)(x_bn, (0, 2, 3))
    std = ops.std(x_bn, (0, 2, 3), keepdims=True)
    x_bn = (x_bn - mean) / std
    
    quantity = ForwardInputCovCondition20(net.bn)
    
    i = 0
    def forward_fn(x):
        y = net(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params())
    
    loss, grads = grad_fn(x_bn)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(net.bn, "input", x_bn)
    quantity.track(i)
    model = 'mindspore_bn'
    name = 'forward_cov_condition_20'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_condition_50():
    setup_seed(42)  # 固定随机种子
    l = nn.Dense(2, 3)  # MindSpore使用Dense替代Linear
    
    # 使用MindSpore的初始化方式
    l.weight.set_data(initializer(Normal(0.1), l.weight.shape))
    l.bias.set_data(initializer(Zero(), l.bias.shape))
    
    batch_size = 1024
    x_linear = ops.StandardNormal()((batch_size, 2))
    # 确保输入是标准化的
    mean = ops.ReduceMean(keep_dims=True)(x_linear, 0)
    std = ops.std(x_linear, 0)
    x_linear = (x_linear - mean) / std
    
    quantity = ForwardInputCovCondition50(l)
    
    i = 0
    def forward_fn(x):
        y = l(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, l.trainable_params())
    
    loss, grads = grad_fn(x_linear)
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(l, "input", x_linear)
    quantity.track(i)
    model = 'mindspore_dese'
    name = 'forward_cov_condition_50'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_condition_50():
    setup_seed(42)
    class ConvNet(nn.Cell):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv = nn.Conv2d(1, 2, 3, stride=1, padding=1, pad_mode='pad',
                                weight_init=Normal(0.03), bias_init=Zero())
        
        def construct(self, x):
            return self.conv(x)
    
    net = ConvNet()
    batch_size = 1024
    x_conv = ops.StandardNormal()((batch_size, 1, 3, 3))
    # 标准化输入
    x_conv = x_conv / ops.sqrt(ops.ReduceMean(keep_dims=True)(x_conv * x_conv, (0,2,3)))
    
    quantity = ForwardInputCovCondition50(net.conv)
    
    i = 0
    def forward_fn(x):
        y = net(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params())
    
    loss, grads = grad_fn(x_conv)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(net.conv, "input", x_conv)
    quantity.track(i)
    model = 'mindspore_conv'
    name = 'forward_cov_condition_50'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_default_condition_50():
    setup_seed(42)
    class BNNet(nn.Cell):
        def __init__(self):
            super(BNNet, self).__init__()
            self.bn = nn.BatchNorm2d(2, gamma_init=Constant(1.0), beta_init=Zero(),
                                   use_batch_statistics=True)
        
        def construct(self, x):
            return self.bn(x)
    
    net = BNNet()
    batch_size = 1024
    # 创建4D输入: (batch_size, channels, height, width)
    x_bn = ops.StandardNormal()((batch_size, 2, 4, 4))
    
    # 标准化输入
    mean = ops.ReduceMean(keep_dims=True)(x_bn, (0, 2, 3))
    std = ops.std(x_bn, (0, 2, 3), keepdims=True)
    x_bn = (x_bn - mean) / std
    
    quantity = ForwardInputCovCondition50(net.bn)
    
    i = 0
    def forward_fn(x):
        y = net(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params())
    
    loss, grads = grad_fn(x_bn)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(net.bn, "input", x_bn)
    quantity.track(i)
    model = 'mindspore_bn'
    name = 'forward_cov_condition_50'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_condition_80():
    setup_seed(42)  # 固定随机种子
    l = nn.Dense(2, 3)  # MindSpore使用Dense替代Linear
    
    # 使用MindSpore的初始化方式
    l.weight.set_data(initializer(Normal(0.1), l.weight.shape))
    l.bias.set_data(initializer(Zero(), l.bias.shape))
    
    batch_size = 1024
    x_linear = ops.StandardNormal()((batch_size, 2))
    # 确保输入是标准化的
    mean = ops.ReduceMean(keep_dims=True)(x_linear, 0)
    std = ops.std(x_linear, 0)
    x_linear = (x_linear - mean) / std
    
    quantity = ForwardInputCovCondition80(l)
    
    i = 0
    def forward_fn(x):
        y = l(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, l.trainable_params())
    
    loss, grads = grad_fn(x_linear)
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(l, "input", x_linear)
    quantity.track(i)
    model = 'mindspore_dese'
    name = 'forward_cov_condition_80'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_condition_80():
    setup_seed(42)
    class ConvNet(nn.Cell):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv = nn.Conv2d(1, 2, 3, stride=1, padding=1, pad_mode='pad',
                                weight_init=Normal(0.03), bias_init=Zero())
        
        def construct(self, x):
            return self.conv(x)
    
    net = ConvNet()
    batch_size = 1024
    x_conv = ops.StandardNormal()((batch_size, 1, 3, 3))
    # 标准化输入
    x_conv = x_conv / ops.sqrt(ops.ReduceMean(keep_dims=True)(x_conv * x_conv, (0,2,3)))
    
    quantity = ForwardInputCovCondition80(net.conv)
    
    i = 0
    def forward_fn(x):
        y = net(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params())
    
    loss, grads = grad_fn(x_conv)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(net.conv, "input", x_conv)
    quantity.track(i)
    model = 'mindspore_conv'
    name = 'forward_cov_condition_80'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_default_condition_80():
    setup_seed(42)
    class BNNet(nn.Cell):
        def __init__(self):
            super(BNNet, self).__init__()
            self.bn = nn.BatchNorm2d(2, gamma_init=Constant(1.0), beta_init=Zero(),
                                   use_batch_statistics=True)
        
        def construct(self, x):
            return self.bn(x)
    
    net = BNNet()
    batch_size = 1024
    # 创建4D输入: (batch_size, channels, height, width)
    x_bn = ops.StandardNormal()((batch_size, 2, 4, 4))
    
    # 标准化输入
    mean = ops.ReduceMean(keep_dims=True)(x_bn, (0, 2, 3))
    std = ops.std(x_bn, (0, 2, 3), keepdims=True)
    x_bn = (x_bn - mean) / std
    
    quantity = ForwardInputCovCondition80(net.bn)
    
    i = 0
    def forward_fn(x):
        y = net(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params())
    
    loss, grads = grad_fn(x_bn)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(net.bn, "input", x_bn)
    quantity.track(i)
    model = 'mindspore_bn'
    name = 'forward_cov_condition_80'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_max_eig():
    setup_seed(42)  # 固定随机种子
    l = nn.Dense(2, 3)  # MindSpore使用Dense替代Linear
    
    # 使用MindSpore的初始化方式
    l.weight.set_data(initializer(Normal(0.1), l.weight.shape))
    l.bias.set_data(initializer(Zero(), l.bias.shape))
    
    batch_size = 1024
    x_linear = ops.StandardNormal()((batch_size, 2))
    # 确保输入是标准化的
    mean = ops.ReduceMean(keep_dims=True)(x_linear, 0)
    std = ops.std(x_linear, 0)
    x_linear = (x_linear - mean) / std
    
    quantity = ForwardInputCovMaxEig(l)
    
    i = 0
    def forward_fn(x):
        y = l(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, l.trainable_params())
    
    loss, grads = grad_fn(x_linear)
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(l, "input", x_linear)
    quantity.track(i)
    model = 'mindspore_dese'
    name = 'forward_cov_condition_max_eig'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_conv_max_eig():
    setup_seed(42)
    class ConvNet(nn.Cell):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv = nn.Conv2d(1, 2, 3, stride=1, padding=1, pad_mode='pad',
                                weight_init=Normal(0.03), bias_init=Zero())
        
        def construct(self, x):
            return self.conv(x)
    
    net = ConvNet()
    batch_size = 1024
    x_conv = ops.StandardNormal()((batch_size, 1, 3, 3))
    # 标准化输入
    x_conv = x_conv / ops.sqrt(ops.ReduceMean(keep_dims=True)(x_conv * x_conv, (0,2,3)))
    
    quantity = ForwardInputCovMaxEig(net.conv)
    
    i = 0
    def forward_fn(x):
        y = net(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params())
    
    loss, grads = grad_fn(x_conv)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(net.conv, "input", x_conv)
    quantity.track(i)
    model = 'mindspore_conv'
    name = 'forward_cov_condition_max_eig'
    print(is_similar(quantity.get_output()[0], 1,model,name))

def test_default_max_eig():
    setup_seed(42)
    class BNNet(nn.Cell):
        def __init__(self):
            super(BNNet, self).__init__()
            self.bn = nn.BatchNorm2d(2, gamma_init=Constant(1.0), beta_init=Zero(),
                                   use_batch_statistics=True)
        
        def construct(self, x):
            return self.bn(x)
    
    net = BNNet()
    batch_size = 1024
    # 创建4D输入: (batch_size, channels, height, width)
    x_bn = ops.StandardNormal()((batch_size, 2, 4, 4))
    
    # 标准化输入
    mean = ops.ReduceMean(keep_dims=True)(x_bn, (0, 2, 3))
    std = ops.std(x_bn, (0, 2, 3), keepdims=True)
    x_bn = (x_bn - mean) / std
    
    quantity = ForwardInputCovMaxEig(net.bn)
    
    i = 0
    def forward_fn(x):
        y = net(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params())
    
    loss, grads = grad_fn(x_bn)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(net.bn, "input", x_bn)
    quantity.track(i)
    model = 'mindspore_bn'
    name = 'forward_cov_condition_max_eig'
    print(is_similar(quantity.get_output()[0], 1,model,name))
def test_linear_stable_rank():
    setup_seed(42)  # 固定随机种子
    l = nn.Dense(2, 3)  # MindSpore使用Dense替代Linear
    
    # 使用MindSpore的初始化方式
    l.weight.set_data(initializer(Normal(0.1), l.weight.shape))
    l.bias.set_data(initializer(Zero(), l.bias.shape))
    
    batch_size = 1024
    x_linear = ops.StandardNormal()((batch_size, 2))
    # 确保输入是标准化的
    mean = ops.ReduceMean(keep_dims=True)(x_linear, 0)
    std = ops.std(x_linear, 0)
    x_linear = (x_linear - mean) / std
    
    quantity = ForwardInputCovStableRank(l)
    
    i = 0
    def forward_fn(x):
        y = l(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, l.trainable_params())
    
    loss, grads = grad_fn(x_linear)
    optimizer = nn.SGD(l.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(l, "input", x_linear)
    quantity.track(i)
    model = 'mindspore_dese'
    name = 'forward_cov_condition_stable_rank'
    print(is_similar_for_stable_rank(quantity.get_output()[0], 1,model,name))

def test_conv_stable_rank():
    setup_seed(42)
    class ConvNet(nn.Cell):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv = nn.Conv2d(1, 2, 3, stride=1, padding=1, pad_mode='pad',
                                weight_init=Normal(0.03), bias_init=Zero())
        
        def construct(self, x):
            return self.conv(x)
    
    net = ConvNet()
    batch_size = 1024
    x_conv = ops.StandardNormal()((batch_size, 1, 3, 3))
    # 标准化输入
    x_conv = x_conv / ops.sqrt(ops.ReduceMean(keep_dims=True)(x_conv * x_conv, (0,2,3)))
    
    quantity = ForwardInputCovMaxEig(net.conv)
    
    i = 0
    def forward_fn(x):
        y = net(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params())
    
    loss, grads = grad_fn(x_conv)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(net.conv, "input", x_conv)
    quantity.track(i)
    model = 'mindspore_conv'
    name = 'forward_cov_condition_stable_rank'
    print(is_similar_for_stable_rank(quantity.get_output()[0], 1,model,name))

def test_default_stable_rank():
    setup_seed(42)
    class BNNet(nn.Cell):
        def __init__(self):
            super(BNNet, self).__init__()
            self.bn = nn.BatchNorm2d(2, gamma_init=Constant(1.0), beta_init=Zero(),
                                   use_batch_statistics=True)
        
        def construct(self, x):
            return self.bn(x)
    
    net = BNNet()
    batch_size = 1024
    # 创建4D输入: (batch_size, channels, height, width)
    x_bn = ops.StandardNormal()((batch_size, 2, 4, 4))
    
    # 标准化输入
    mean = ops.ReduceMean(keep_dims=True)(x_bn, (0, 2, 3))
    std = ops.std(x_bn, (0, 2, 3), keepdims=True)
    x_bn = (x_bn - mean) / std
    
    quantity = ForwardInputCovMaxEig(net.bn)
    
    i = 0
    def forward_fn(x):
        y = net(x)
        return ops.ReduceSum()(y)

    grad_fn = ops.value_and_grad(forward_fn, None, net.trainable_params())
    
    loss, grads = grad_fn(x_bn)
    optimizer = nn.SGD(net.trainable_params(), learning_rate=0.01)
    optimizer(grads)
    
    setattr(net.bn, "input", x_bn)
    quantity.track(i)
    model = 'mindspore_bn'
    name = 'forward_cov_condition_stable_rank'
    print(is_similar_for_stable_rank(quantity.get_output()[0], 1,model,name))
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
        nn.BatchNorm2d: ['ForwardInputCovCondition20','ForwardInputCovMaxEig',
                         'ForwardInputCovStableRank']
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
    # 运行测试
    test_linear_cov_conition()
    test_conv_cov_condition()
    test_default_cov_condition()
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
    test_linear_stable_rank()
    test_conv_stable_rank()
    test_default_stable_rank()
    config, config_mmonitor = prepare_config()
    train_model(config, config_mmonitor)
