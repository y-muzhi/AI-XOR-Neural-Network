import matplotlib.pyplot as plt
import numpy as np
from typing import List

import matplotlib.pyplot as plt
from random import random, seed
from typing import Optional, List, Callable, TypeVar, Tuple
from math import exp

# 固定随机种子便于复现（可删除）
seed(32)

# -----------------------
# 基础函数
# -----------------------
def dot_product(xs: List[float], ys: List[float]) -> float:
    return sum(x * y for x, y in zip(xs, ys))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))

def derivative_sigmoid(x: float) -> float:
    """x 是神经元的线性输出（未经过激活函数的值）"""
    s = sigmoid(x)
    return s * (1 - s)

T = TypeVar("T")

# -----------------------
# Neuron
# -----------------------
class Neuron:
    def __init__(
        self,
        weights: List[float],
        bias: float,
        learning_rate: float,
        activation_function: Callable[[float], float],
        derivative_activation_function: Callable[[float], float],
    ) -> None:
        self.weights: List[float] = weights
        self.bias: float = bias
        self.learning_rate: float = learning_rate
        self.activation_function = activation_function
        self.derivative_activation_function = derivative_activation_function

        # 在前向传播中，这里保存的是线性组合值 z = w·x + b
        self.output_cache: float = 0.0
        # delta 用于反向传播（误差信号）
        self.delta: float = 0.0

    def output(self, inputs: List[float]) -> float:
        """计算 z = w·x + b 并返回激活后的值"""
        self.output_cache = dot_product(inputs, self.weights) + self.bias
        return self.activation_function(self.output_cache)


# -----------------------
# Layer
# -----------------------
class Layer:
    def __init__(
        self,
        previous_layer: Optional["Layer"],
        num_neurons: int,
        learning_rate: float,
        activation_function: Callable[[float], float],
        derivative_activation_function: Callable[[float], float],
    ) -> None:
        self.previous_layer = previous_layer
        self.neurons: List[Neuron] = []

        # 如果有上一层，上一层的神经元数量决定当前神经元每个权重的数量
        prev_count = len(previous_layer.neurons) if previous_layer is not None else 0
        for _ in range(num_neurons):
            if previous_layer is None:
                # 输入层的“神经元”不需要权重/bias（我们只用它保存数量）
                weights: List[float] = []
                bias = 0.0
            else:
                weights = [random() - 0.5 for _ in range(prev_count)]  # 小范围随机初始值
                bias = random() - 0.5
            neuron = Neuron(weights, bias, learning_rate, activation_function, derivative_activation_function)
            self.neurons.append(neuron)

        # 缓存该层的输出（对输入层就是原始输入）
        self.output_cache: List[float] = [0.0 for _ in range(num_neurons)]

    def outputs(self, inputs: List[float]) -> List[float]:
        """计算并返回该层的所有输出"""
        if self.previous_layer is None:
            # 输入层：直接保存并返回输入向量
            self.output_cache = inputs
        else:
            self.output_cache = [n.output(inputs) for n in self.neurons]
        return self.output_cache

    def calculate_deltas_for_output_layer(self, expected: List[float]) -> None:
        """输出层的 delta = derivative(z) * (expected - actual)"""
        for i, neuron in enumerate(self.neurons):
            actual = self.output_cache[i]
            neuron.delta = neuron.derivative_activation_function(neuron.output_cache) * (expected[i] - actual)

    def calculate_deltas_for_hidden_layer(self, next_layer: "Layer") -> None:
        """隐藏层的 delta = derivative(z) * sum(next_weight * next_delta)"""
        for idx, neuron in enumerate(self.neurons):
            # 收集下一层中对应连接到当前神经元的权重
            next_weights = [n.weights[idx] for n in next_layer.neurons]
            next_deltas = [n.delta for n in next_layer.neurons]
            weighted_sum = dot_product(next_weights, next_deltas)
            neuron.delta = neuron.derivative_activation_function(neuron.output_cache) * weighted_sum


# -----------------------
# Network
# -----------------------
class Network:
    def __init__(
        self,
        layer_structure: List[int],
        learning_rate: float,
        activation_function: Callable[[float], float] = sigmoid,
        derivative_activation_function: Callable[[float], float] = derivative_sigmoid,
    ) -> None:
        if len(layer_structure) < 3:
            raise ValueError("需要至少 3 层：输入、隐藏、输出")

        self.layers: List[Layer] = []
        self.learning_rate = learning_rate

        # 创建输入层（previous_layer = None）
        self.layers.append(
            Layer(None, layer_structure[0], learning_rate, activation_function, derivative_activation_function)
        )

        # 创建其他层（隐藏层与输出层）
        for n in layer_structure[1:]:
            self.layers.append(
                Layer(self.layers[-1], n, learning_rate, activation_function, derivative_activation_function)
            )

    def outputs(self, inputs: List[float]) -> List[float]:
        """正向传播，依次让每层计算输出"""
        for layer in self.layers:
            inputs = layer.outputs(inputs)
        return inputs

    def backpropagate(self, expected: List[float]) -> None:
        """计算所有层的 deltas（从输出层向前）"""
        last = len(self.layers) - 1
        # 输出层 deltas
        self.layers[last].calculate_deltas_for_output_layer(expected)
        # 隐藏层 deltas（倒序）
        for l in range(last - 1, 0, -1):
            self.layers[l].calculate_deltas_for_hidden_layer(self.layers[l + 1])

    def update_weights(self) -> None:
        """根据 deltas 更新权重和 bias"""
        for layer in self.layers[1:]:  # 跳过输入层
            prev_outputs = layer.previous_layer.output_cache
            for neuron in layer.neurons:
                # 更新每个权重
                for w_idx in range(len(neuron.weights)):
                    input_val = prev_outputs[w_idx]
                    neuron.weights[w_idx] += neuron.learning_rate * input_val * neuron.delta
                # 更新 bias（bias 当作权重对应输入 1）
                neuron.bias += neuron.learning_rate * 1.0 * neuron.delta

    def train(self, inputs: List[List[float]], expecteds: List[List[float]], epochs: int = 1000, verbose: bool = True) -> None:
        """训练网络。expecteds 中每项应是列表（对输出层每个神经元的目标值）"""
        errors = []
        for epoch in range(epochs):
            total_error = 0.0
            for xs, ys in zip(inputs, expecteds):
                outs = self.outputs(xs)
                # 均方误差累加（方便观察训练进度）
                total_error += sum((y - o) ** 2 for y, o in zip(ys, outs))
                self.backpropagate(ys)
                self.update_weights()
            errors.append(total_error)
            if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}  total_error={total_error:.6f}") 

        if verbose:
            plt.figure()
            plt.plot(range(1, epochs + 1), errors)
            plt.title("Training Loss Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Total Error")
            plt.grid(True)
            # plt.show()

    def predict(self, inputs: List[float], threshold: float = 0.5) -> List[float]:
        """返回原始输出，也可以用 threshold 把输出变成 0/1"""
        outs = self.outputs(inputs)
        return [1 if o >= threshold else 0 for o in outs]

    def validate(self, inputs: List[List[float]], expecteds: List[T], interpret_output: Callable[[List[float]], T]) -> Tuple[int, int, float]:
        correct = 0
        total = len(inputs)
        for x, expected in zip(inputs, expecteds):
            result = interpret_output(self.outputs(x))
            if result == expected:
                correct += 1
        return correct, total, correct / total if total > 0 else 0.0


def visualize_hidden_activations(network: Network, resolution: int = 50) -> None:
    """
    可视化隐藏层每个神经元在输入空间上的激活模式
    
    参数:
        network: 训练好的神经网络
        resolution: 可视化分辨率（网格密度）
    """
    if len(network.layers) < 3:
        print("网络至少需要包含输入层、隐藏层和输出层")
        return
    
    # 获取第一个隐藏层
    hidden_layer = network.layers[1]  # 第一个隐藏层
    num_neurons = len(hidden_layer.neurons)
    
    if num_neurons == 0:
        print("隐藏层没有神经元")
        return
    
    # 创建输入空间网格
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # 为每个隐藏层神经元创建激活图
    fig, axes = plt.subplots(1, num_neurons, figsize=(5*num_neurons, 5))
    if num_neurons == 1:
        axes = [axes]
    
    for neuron_idx, neuron in enumerate(hidden_layer.neurons):
        # 计算网格上每个点的激活值
        activations = np.zeros_like(xx)
        for i in range(resolution):
            for j in range(resolution):
                inputs = [xx[j, i], yy[j, i]]
                # 直接计算线性组合值（不经过激活函数）
                linear_output = dot_product(inputs, neuron.weights) + neuron.bias
                # 使用激活函数获取最终激活值
                activations[j, i] = neuron.activation_function(linear_output)
        
        # 绘制热力图
        im = axes[neuron_idx].imshow(activations, extent=[x_min, x_max, y_min, y_max],
                                     origin='lower', cmap='viridis')
        axes[neuron_idx].set_title(f'Neuron {neuron_idx+1} Activation')
        axes[neuron_idx].set_xlabel('Input 1')
        axes[neuron_idx].set_ylabel('Input 2')
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[neuron_idx])
        
        # 绘制训练数据点作为参考
        training_points = [
            ([0, 0], 'red'),   # XOR: 0
            ([0, 1], 'blue'),  # XOR: 1
            ([1, 0], 'blue'),  # XOR: 1
            ([1, 1], 'red')    # XOR: 0
        ]
        
        for point, color in training_points:
            axes[neuron_idx].scatter(point[0], point[1], c=color, s=100, edgecolor='white')
    
    plt.tight_layout()
    plt.show()

def visualize_decision_boundary(network: Network, resolution: int = 100) -> None:
    """
    可视化网络的整体决策边界
    
    参数:
        network: 训练好的神经网络
        resolution: 决策边界的分辨率
    """
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    
    # 计算整个网格的网络输出
    zz = np.zeros_like(xx)
    for i in range(resolution):
        for j in range(resolution):
            inputs = [xx[j, i], yy[j, i]]
            output = network.outputs(inputs)
            zz[j, i] = output[0]  # 假设单输出
    
    # 绘制决策边界
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(xx, yy, zz, levels=50, cmap='RdYlBu', alpha=0.7)
    plt.colorbar(contour)
    
    # 绘制等高线
    contours = plt.contour(xx, yy, zz, levels=[0.3, 0.5, 0.7], colors='black', linestyles='--', linewidths=1)
    plt.clabel(contours, inline=True, fontsize=8)
    
    # 绘制训练数据点
    training_data = [
        ([0, 0], 0, 'red'),
        ([0, 1], 1, 'blue'),
        ([1, 0], 1, 'blue'),
        ([1, 1], 0, 'red')
    ]
    
    for point, target, color in training_data:
        marker = 'o' if target == 0 else '^'
        plt.scatter(point[0], point[1], c=color, s=100, marker=marker, 
                   edgecolor='black', linewidth=1, label=f'({point[0]}, {point[1]}) -> {target}')
    
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('Network Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# 修改主程序部分以使用可视化功能
if __name__ == "__main__":
    # 训练XOR网络
    nn = Network([2, 4, 1], learning_rate=0.5)

    xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    ys = [[0], [1], [1], [0]]

    nn.train(xs, ys, epochs=2500, verbose=True)

    print("\nXOR 预测结果：")
    for x in xs:
        print(f"{x} -> {nn.outputs(x)} -> {nn.predict(x)}")
    
    # 可视化隐藏层神经元激活区域
    visualize_hidden_activations(nn, resolution=50)
    
    # 可视化整体决策边界
    visualize_decision_boundary(nn, resolution=100)



cd /Users/muzhi/Desktop/C++/测试程序/AI4.py

git init

git config user.name "y-muzhi"
git config user.email "2317223363@qq.com"


git add .


git commit -m "Initial commit: Add XOR neural network training code"


git remote add origin https://github.com/y-muzhi/AI-XOR-Neural-Network.git

git branch -M main
git push -u origin main