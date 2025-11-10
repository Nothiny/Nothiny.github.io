---
title:      "Python 中的广播机制 (Broadcasting)"
date:       2025-10-27 12:00:00
tags:
    - hello world
---

# Python 中的广播机制 (Broadcasting)

## 什么是广播

**广播 (Broadcasting)** 是 NumPy 和 PyTorch 等科学计算库中的一种机制，允许不同形状的数组进行算术运算，而无需显式复制数据。

### 核心思想

>  广播通过**虚拟扩展**较小数组的形状来匹配较大数组，在计算时**重复使用**数据，而不实际占用额外内存。

**优势：**

-  **内存高效**：不创建数据副本
- **代码简洁**：避免手写循环
-  **计算快速**：利用向量化操作

## 广播的三大规则

### 规则 1：维度对齐

如果两个数组维度数不同，在形状较小的数组**前面**补 1，直到维度数相同。

```python
import numpy as np

a = np.array([1, 2, 3, 4])      # shape: (4,)
b = np.array([[10], [20], [30]]) # shape: (3, 1)

# a 自动变为 (1, 4)
# b 保持为 (3, 1)
```

### 规则 2：维度兼容性检查

从右向左逐个比较每个维度，满足以下条件之一即为兼容：

- 两个维度相等
- 其中一个维度为 1

```python
#  兼容示例
(3, 1) 和 (1, 4)  -> 可以广播
(5, 3, 4) 和 (3, 4) -> 可以广播
(8, 1, 6, 1) 和 (7, 1, 5) -> 可以广播

#  不兼容示例
(3, 4) 和 (3, 5)  -> 无法广播 (最后一维 4≠5)
```

### 规则 3：形状扩展

将维度为 1 的维度"拉伸"到匹配另一个数组的对应维度。

```python
(3, 1) -> (3, 4)  # 第2维从1扩展到4
(1, 4) -> (3, 4)  # 第1维从1扩展到3
```

## 基础示例

### 示例 1：向量与标量

```python
import numpy as np

# 标量广播到向量
a = np.array([1, 2, 3, 4])
b = 10

result = a + b
print(result)  # [11 12 13 14]

# 等价于：
# b 被广播为 [10, 10, 10, 10]
```

### 示例 2：一维数组与二维数组

```python
# 向量广播到矩阵
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

vector = np.array([10, 20, 30])

result = matrix + vector
print(result)
# [[11 22 33]
#  [14 25 36]
#  [17 28 39]]

# vector 被广播为：
# [[10, 20, 30],
#  [10, 20, 30],
#  [10, 20, 30]]
```

### 示例 3：外积运算

```python
# 列向量 + 行向量 = 矩阵
x = np.array([1, 2, 3, 4])      # shape: (4,)
y = np.array([10, 20, 30])      # shape: (3,)

# 增加维度
x_col = x[:, np.newaxis]  # shape: (4, 1)
y_row = y[np.newaxis, :]  # shape: (1, 3)

result = x_col + y_row
print(result)
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]
#  [14 24 34]]
```

### 示例 4：复杂形状

```python
# 三维张量广播
a = np.ones((3, 4, 5))      # shape: (3, 4, 5)
b = np.ones((4, 1))         # shape: (4, 1)

# b 自动变为 (1, 4, 1)，然后广播到 (3, 4, 5)
result = a + b
print(result.shape)  # (3, 4, 5)
```

## 常见应用场景

### 1. 数据归一化

```python
# 按列归一化
data = np.random.randn(100, 5)  # 100个样本，5个特征

# 计算每列的均值和标准差
mean = data.mean(axis=0)  # shape: (5,)
std = data.std(axis=0)    # shape: (5,)

# 广播标准化
normalized = (data - mean) / std  # mean 和 std 自动广播到 (100, 5)
```

### 2. 图像处理

```python
# RGB 图像每个通道减去均值
image = np.random.randint(0, 255, (224, 224, 3))  # H×W×C
mean_rgb = np.array([123.675, 116.28, 103.53])    # shape: (3,)

# 广播减法
centered_image = image - mean_rgb  # mean_rgb 广播到 (224, 224, 3)
```

### 3. 距离矩阵计算

```python
# 计算所有点对之间的欧式距离
points = np.random.randn(100, 2)  # 100个2D点

# 利用广播计算距离矩阵
diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  
# shape: (100, 100, 2)

distances = np.sqrt((diff ** 2).sum(axis=2))  # shape: (100, 100)
```

## 深度学习中的应用

### 0. Triton Kernel 中的向量外积

```python
import numpy as np

def add_vec_kernel_numpy(x, y):
    z = y[:, None] + x[None, :]
    return z

# 示例
x = np.array([1, 2, 3, 4])       # B0 = 4
y = np.array([10, 20, 30])       # B1 = 3

z = add_vec_kernel_numpy(x, y)
print(z)
# [[11 12 13 14]
#  [21 22 23 24]
#  [31 32 33 34]]

print(f"原始形状: x{x.shape}, y{y.shape}")
y_col = y[:, None]
x_row = x[None, :]
print(f"增加维度: y[:,None]{y_col.shape}, x[None,:]{x_row.shape}")
'''
步骤1 - 原始形状: x(4,), y(3,)
步骤2 - 增加维度: y[:,None](3, 1), x[None,:](1, 4)
步骤3 - 广播扩展:
  y[:,None] 从 (3,1) 广播到 (3,4)
  x[None,:] 从 (1,4) 广播到 (3,4)
步骤4 - 结果形状: (3, 4)
'''
```

### 1. Batch Normalization

```python
# BatchNorm 中的广播操作
def batch_norm(x, gamma, beta, eps=1e-5):
    """
    x: (N, C, H, W) - 批量图像
    gamma, beta: (C,) - 可学习参数
    """
    # 计算每个通道的均值和方差
    mean = x.mean(dim=(0, 2, 3), keepdim=True)  # shape: (1, C, 1, 1)
    var = x.var(dim=(0, 2, 3), keepdim=True)    # shape: (1, C, 1, 1)
    
    # 标准化
    x_norm = (x - mean) / torch.sqrt(var + eps)
    
    # gamma 和 beta 广播
    gamma = gamma.view(1, -1, 1, 1)  # (1, C, 1, 1)
    beta = beta.view(1, -1, 1, 1)    # (1, C, 1, 1)
    
    return gamma * x_norm + beta

# 使用示例
x = torch.randn(32, 64, 28, 28)  # (N, C, H, W)
gamma = torch.ones(64)
beta = torch.zeros(64)

output = batch_norm(x, gamma, beta)
print(output.shape)  # torch.Size([32, 64, 28, 28])
```

## 性能对比

### 广播 vs 显式循环

```python
import time
import numpy as np
# 准备数据
matrix = np.random.randn(1000, 1000)
vector = np.random.randn(1000)
# 方法1：广播（推荐）
start = time.time()
result1 = matrix + vector
time_broadcast = time.time() - start
# 方法2：显式循环
start = time.time()
result2 = np.zeros_like(matrix)
for i in range(matrix.shape[0]):
    result2[i] = matrix[i] + vector
    
time_loop = time.time() - start

print(f"广播耗时: {time_broadcast:.6f}s")
print(f"循环耗时: {time_loop:.6f}s")
print(f"加速比: {time_loop / time_broadcast:.1f}x")

# 典型输出：
# 广播耗时: 0.000631s
# 循环耗时: 0.001083s
# 加速比: 1.7x
```

## 常见陷阱与调试技巧

### 陷阱 1：意外的广播

```python
# 错误示例
a = np.array([[1, 2, 3]])     # shape: (1, 3)
b = np.array([[1], [2], [3]]) # shape: (3, 1)

c = a + b  # 意外得到 (3, 3) 的结果！
print(c.shape)  # (3, 3)

assert a.shape == b.shape, "形状不匹配"
```

###  调试技巧

```python
# 1. 使用 .shape 检查
print(f"a.shape = {a.shape}, b.shape = {b.shape}")

# 2. 使用 np.broadcast_shapes 预测结果
from numpy import broadcast_shapes
result_shape = broadcast_shapes(a.shape, b.shape)
print(f"广播后形状: {result_shape}")

# 3. 使用 keepdim 保持维度
mean = a.mean(axis=1, keepdims=True)  # 保持维度便于广播
```

