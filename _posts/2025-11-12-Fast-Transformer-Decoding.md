---
title:      "Fast Transformer Decoding: One Write-Head is All You Need 论文阅读"
date:       2025-11-12 12:00:00
header-img: img/wallhaven-lm6jm2.jpg
tags:
    - llm 推理
    - kv cache
---

# Fast Transformer Decoding: One Write-Head is All You Need 论文阅读

> 本篇文章在MHA的基础上做了修改，不使用多头注意力机制，而是使用多query机制，所有的query共享一个k，v头。所以这带来性能提升时必然的，因为减少了计算。所以在不降低原有的准确率似乎是重点。
>
> 刚接触这个领域，如有错误，希望能得到您的指正。

## introduction

首先作者提出了Transformer 依靠注意力层在序列之间和序列之间传递信息。 Transformer 的一大挑战是增量推理（incremental inference）的速度。现代计算硬件上增量 Transformer 推理的速度受到重新加载attention layers状态的大“key”和“value”张量所需的内存带宽的限制。作者提出一种架构变体（多查询注意力），该变体极大地提高了推理速度，而质量仅略有下降。

## background

这里的实例中省略了除以维度$\sqrt{d_k}$  ，因为这是一个标量计算对整个计算过程的时间复杂度不会产生影响。

#### 点乘注意力机制

```python
def DotProductAttention (q , K, V) :
	'''Dot−Product Attention on one query.
	Args :
		q : a vector with shape [k]
		K: a matrix with shape [m, k]
		V: a matrix with shape [m, v]
	Returns :
	    y : a vector with shape [v]
	'''
    logits = tf.einsum("k , mk−>m" , q , K)
    weights = tf.softmax(logits)
    return tf.einsum("m, mv−>v", weights , V)
```

#### batched多头注意力

transformer模型使用h个不同的注意力头，他们在计算时互相独立，只有在最终合并输出时才会产生关联。同时，将多个query合并在一起使用批处理的方式会高效得多。这里使用mask来防止后向信息流。

```python
def MultiheadAttention(
	X, M, mask, P_q, P_k, P_v, P_o):
    '''Multi-head Attention on one query
    Args:
    	X : a vector with shape [b, n, d]
    	M : a matrix with shape [b, m, d]
    	mask: a tensor with shape [b, h, n, m]
    	P_q: a tensor with shape [h, d, k]
    	P_k: a tensot with shape [h, d, k]
    	P_v: a tensor with shape [h, d, v]
    	P_o: a tensor with shape [h, d, v]
    Returns:
    	y: a vector with shape [b, n, d]
    '''
    Q = tf.einsum("bnd,hdk -> bhnk", X, P_q)
    K = tf.einsum("bmd,hdk -> bhmk", M, P_k)
    V = tf.einsum("bmd,hdv -> bhmv", M, P_v)
    logits = tf.einsum("bhnk,bhmk -> bhnm", Q, K)
    weights = tf.softmax(logits + mask)
    o = tf.einsum("bhnm,bhmv -> bhnv", weight, V)
    y = tf.einsum("bhnv,hdv -> bnd", o, P_o)
    return y
```

维度符号说明

| 符号  | 含义              | 说明                                    |
| ----- | ----------------- | --------------------------------------- |
| **b** | batch size        | 批次大小，一次处理多少个独立序列        |
| **n** | query序列长度     | 当前序列中有多少个位置需要计算attention |
| **m** | key/value序列长度 | 用于计算attention的上下文长度           |
| **d** | 模型维度          | 输入/输出的embedding维度                |
| **h** | 注意力头数        | 多少个并行的attention头                 |
| **k** | key/query维度     | 每个头的key和query维度（通常 k = d/h）  |
| **v** | value维度         | 每个头的value维度（通常 v = d/h）       |

这里介绍了典型值的说明，同时做出了简化

* $m=n$
* $k=v=d/h$ 
* $n <= d$  

运算的总的复杂度是$O(bnd^2)$ ，内存访问的复杂度是$O(bnd + bhn^2 + d^2)$，第一项由 X、M、Q、K、V、O 和 Y 决定，第二项由 Logits 和weight决定，第三项由投影张量 P_q、P_k、P_v 和 P_o 决定。

将二者做除法，内存访问与算术运算的比率为$O(1/k + 1/(bn))$ ，这和现代gpu/tpu想匹配，因为计算能力可能比内存访问速度高两个数量级。

#### 增量多头注意力机制

某些情况下，由于数据依赖性使得无法并行处理来自多个位置的query。在训练过程中，由于所有位置的token已知，可以使用上面的并行方式进行处理，但是在训练好模型进行推理之后，一个位置的输出会影响后续位置的输出，这导致无法并行计算。

```python
def MultiheadSelfAttentionIncremental(x, prev_K, prev_V, P_q, P_k, P_v, P_o):
    """
    Multi-head Self-Attention
    
    Args:
        x: a tensor with shape [b, d]          
        prev_K: tensor with shape [b, h, m, k] 
        prev_V: tensor with shape [b, h, m, v] 
        P_q: a tensor with shape [h, d, k]     
        P_k: a tensor with shape [h, d, k]     
        P_v: a tensor with shape [h, d, v]     
        P_o: a tensor with shape [h, d, v]    
    
    Returns:
        y: a tensor with shape [b, d]        
        new_K: tensor with shape [b, h, m+1, k]
        new_V: tensor with shape [b, h, m+1, v] 
    """
   
    q = tf.einsum("bd,hdk->bhk", x, P_q)
    new_K = tf.concat([
        prev_K, 
        tf.expand_dims(
            tf.einsum("bd,hdk->bhk", x, P_k),  
            axis=2  
        )
    ], axis=2)  
    
    new_V = tf.concat([
        prev_V,  
        tf.expand_dims(
            tf.einsum("bd,hdv->bhv", x, P_v),  
            axis=2  
        )
    ], axis=2) 
    
    logits = tf.einsum("bhk,bhmk->bhm", q, new_K)
    weights = tf.softmax(logits)
    o = tf.einsum("bhm,bhmv->bhv", weights, new_V)
    y = tf.einsum("bhv,hdv->bd", o, P_o)
    
    return y, new_K, new_V
```

维度解释

| 变量      | 形状             | 含义                                           |
| --------- | ---------------- | ---------------------------------------------- |
| `x`       | `[b, d]`         | **当前步**的输入（batch中每个序列的当前token） |
| `prev_K`  | `[b, h, m, k]`   | **历史缓存**：已生成m个token的Key              |
| `prev_V`  | `[b, h, m, v]`   | **历史缓存**：已生成m个token的Value            |
| `q`       | `[b, h, k]`      | 当前token的Query（h个头）                      |
| `new_K`   | `[b, h, m+1, k]` | 更新后的Key缓存（增加了当前token）             |
| `new_V`   | `[b, h, m+1, v]` | 更新后的Value缓存（增加了当前token）           |
| `logits`  | `[b, h, m+1]`    | 当前Query对所有(m+1)个Key的分数                |
| `weights` | `[b, h, m+1]`    | Attention权重分布                              |
| `o`       | `[b, h, v]`      | h个头的输出                                    |
| `y`       | `[b, d]`         | 当前位置的最终输出                             |

性能分析，使用相同的上述假设，算数运算的复杂度仍然是$O(bnd^2)$，内存访问总量是$O(bn^2d + nd^2)$，第一项由 K 和 V 决定，第二项由 P_q、P_k、P_v 和 P_o 决定。此时比率为$O(n/d + 1/b)$，当$n\approx d$ 或者$b\approx 1$ 时，该比率接近为1，此时内存带宽成为现代硬件的瓶颈。为了是这一项尽可能减小，可以增加批量大小b，这是很显然的，$n/d$这项如何减小是比较困难的。这一项与重复加载k和v张量有关。一种方式通过限制序列长度n，另一种方式是通过减少token关注的邻居位置，或者使用其他方式压缩位置信息来减少被关注的位置数量（就是不计算某token与之前的全部token的attention分数）。而在本文中，是使用一种正交的方式，即删除 heads 维度，但是保存query的 heads 维度。

## Multi-Query Attention

多头注意力由多个注意力层（头）组成，并与查询、键、值和输出上的不同线性变换并行。多查询注意力是相同的，只是不同的头共享一组键和值。 （增量）多查询（自）注意力的代码与上面列出的多头注意力的代码相同，只是我们从 tf.einsum 方程中删除了字母“h”，其中它代表 K、V、Pk 或 Pv 的“头”维度。

```python
def MultiqueryAttentionBatched(X, M, mask, P_q, P_k, P_v, P_o):
    """
    Multi-Query Attention
    
    Args:
        X: [b, n, d] - batch中n个query位置的输入
        M: [b, m, d] - batch中m个key/value位置的输入
        mask: [b, h, n, m] - 注意力掩码
        P_q: [h, d, k] - h个query投影矩阵
        P_k: [d, k] - 单个key投影矩阵（所有头共享）
        P_v: [d, v] - 单个value投影矩阵（所有头共享）
        P_o: [h, d, v] - h个output投影矩阵
    
    Returns:
        Y: [b, n, d] - 输出
    """
    Q = tf.einsum("bnd,hdk->bhnk", X, P_q)
    K = tf.einsum("bmd,dk->bmk", M, P_k)
    V = tf.einsum("bmd,dv->bmv", M, P_v)
    logits = tf.einsum("bhnk,bmk->bhnm", Q, K)
    weights = tf.softmax(logits + mask)
    O = tf.einsum("bhnm,bmv->bhnv", weights, V)
    Y = tf.einsum("bhnv,hdv->bnd", O, P_o)
    return Y


def MultiquerySelfAttentionIncremental(x, prev_K, prev_V, P_q, P_k, P_v, P_o):
    """
    Multi-Query Self-Attention (增量版本)
    
    Args:
        x: [b, d] - 当前token的输入
        prev_K: [b, m, k] - 之前缓存的Keys（所有头共享）
        prev_V: [b, m, v] - 之前缓存的Values（所有头共享）
        P_q: [h, d, k] - h个query投影矩阵
        P_k: [d, k] - 单个key投影矩阵（所有头共享）
        P_v: [d, v] - 单个value投影矩阵（所有头共享）
        P_o: [h, d, v] - h个output投影矩阵
    
    Returns:
        y: [b, d] - 当前位置的输出
        new_K: [b, m+1, k] - 更新后的Key缓存
        new_V: [b, m+1, v] - 更新后的Value缓存
    """
    q = tf.einsum("bd,hdk->bhk", x, P_q)
    new_K = tf.concat([
        prev_K,
        tf.expand_dims(tf.einsum("bd,dk->bk", x, P_k), axis=1)
    ], axis=1)
    new_V = tf.concat([
        prev_V,
        tf.expand_dims(tf.einsum("bd,dv->bv", x, P_v), axis=1)
    ], axis=1)
    logits = tf.einsum("bhk,bmk->bhm", q, new_K)
    weights = tf.softmax(logits)
    o = tf.einsum("bhm,bmv->bhv", weights, new_V)
    y = tf.einsum("bhv,hdv->bd", o, P_o)
    return y, new_K, new_V
```

**符号说明:**

- **b**: batch size (批次大小)
- **n**: query序列长度
- **m**: key/value序列长度
- **d**: 模型维度
- **h**: attention头数
- **k**: key/query维度
- **v**: value维度

**Multi-Query Attention的关键区别:**

- P_k和P_v没有h维度 → 所有头共享同一个Key/Value投影
- prev_K和prev_V也没有h维度 → KV缓存大小减少h倍

使用前述假设计算复杂度，算术运算的复杂度仍然是$O(bnd^2)$。内存访问的复杂度是$O(bnd + bn^2d +nd^2)$，第一项由 x、q、o 和 y 产生，第二项由 K 和 V 产生，第三项由 P_q、P_k、P_v、P_o 产生。将内存除以计算，发现内存访问与算术运算的比率为$O(1/d + n/(dh) + 1/b)$，将计算量 n d 减少了 h 倍。理论上，给定大批量大小 b，这应该会显着提高增量生成的性能。

## 实验和结果

使用attention is all you need的相同任务，并将其作为baseline进行对比。

| 实验设置项              | 值                                       |
| ----------------------- | ---------------------------------------- |
| **数据集**              | WMT 2014 English-German                  |
| **模型架构**            | Encoder-Decoder Transformer              |
| **层数**                | 6 layers                                 |
| **模型维度 (d_model)**  | 1024                                     |
| **前馈网络维度 (d_ff)** | 4096                                     |
| **注意力头数 (h)**      | 8                                        |
| **Key维度 (d_k)**       | 128                                      |
| **Value维度 (d_v)**     | 128                                      |
| **位置编码**            | Learned positional embeddings            |
| **权重共享**            | Token embedding 和 output layer 共享     |
| **总参数量**            | 211 million                              |
| **训练步数**            | 100,000 steps (~20 epochs)               |
| **Batch size**          | 128 examples                             |
| **序列长度**            | 256 tokens (input) + 256 tokens (target) |
| **序列构造方式**        | 多个句子拼接至256 tokens                 |
| **训练硬件**            | 32-core TPUv3 cluster                    |
| **训练时长**            | ~2 hours per model                       |

在“多query”模型中，将模型中的所有注意力层替换为多查询注意力，这包括编码器自注意力层、解码器自注意力层和编码器-解码器注意力层。将前馈隐藏层从 4096 扩大到 5440，以使总参数计数等于baseline。

为了证明局部注意力和多查询注意力是正交的，还训练了baseline和多查询模型的“局部”版本，其中解码器自注意力层（而不是其他注意力层）将注意力限制在当前位置和之前的 31 个位置。

减小 K 和 V 大小的一种更简单的替代方法是减少头 h 的数量和/或减小键和值的维度 k 和 v。该论文训练了几个这样的模型进行比较，同时再次扩大前馈隐藏层以使总参数计数等于基线。对于baseline，使用 6 层模型，$d_{model} = 1024， d_{ff} = 8192，h = 8，d_k = d_v = 128$。baseline和所有变体的总参数计数为 1.92 亿。

### 模型质量

下表显示了机器翻译实验的结果。使用贪婪最大似然解码对开发集进行解码，并使用 sacrebleu `sacrebleu -t wmt13 -l en-de -tok intl`计算 BLEU 分数。还列出了开发集上每个sub token的困惑度。根据这两个指标，多查询注意力模型似乎比baseline稍差，但更接近减少 h、dk 和 dv 的任何替代方案。

| Attention Type    |    h | d_k, d_v | d_ff | ln(PPL) (dev) | BLEU (dev) | BLEU (test) beam 1 / 4 |
| :---------------- | ---: | :------: | :--: | :-----------: | :--------: | :--------------------: |
| multi-head        |    8 |   128    | 4096 |   **1.424**   |  **26.7**  |    27.7 / **28.4**     |
| multi-query       |    8 |   128    | 5440 |     1.439     |    26.5    |    27.5 / **28.5**     |
| multi-head local  |    8 |   128    | 4096 |     1.427     |    26.6    |      27.5 / 28.3       |
| multi-query local |    8 |   128    | 5440 |     1.437     |    26.5    |    **27.6** / 28.2     |
| multi-head        |    1 |   128    | 6784 |     1.518     |    25.8    |           -            |
| multi-head        |    2 |    64    | 6784 |     1.480     |    26.2    |      26.8 / 27.9       |
| multi-head        |    4 |    32    | 6784 |     1.488     |    26.1    |           -            |
| multi-head        |    8 |    16    | 6784 |     1.513     |    25.8    |           -            |

通过使用贪婪解码和beam search（beam 4，α = 0.6）对测试集进行解码来验证结果，并使用 sacrebleu `sacrebleu -t wmt14 -l en-de -tok intl`进行评估。同样，多查询模型的表现与baseline类似，并且实际上在使用 Beam-4 解码时具有最高的 BLEU 分数 (28.5)。

下表显示了十亿字语言建模基准的结果。模型是通过开发集上的每个字（而不是每个sub token）的困惑度来评估的。结果与翻译结果相似。多查询注意力模型比基线稍差，但明显优于涉及减少 h、dk 和 dv 的任何替代方案

| Attention Type |    h | d_k, d_v | d_ff | dev-PPL  |
| :------------- | ---: | :------: | :--: | :------: |
| multi-head     |    8 |   128    | 8192 | **29.9** |
| multi-query    |    8 |   128    | 9088 |   30.2   |
| multi-head     |    1 |   128    | 9984 |   31.2   |
| multi-head     |    2 |    64    | 9984 |   31.1   |
| multi-head     |    4 |    32    | 9984 |   31.0   |
| multi-head     |    8 |    16    | 9984 |   30.9   |

### 速度

下表显示了各种模型的训练和推理时间。训练和推理速度均在一个 TPUv2（8 核）上进行评估。基础模型的训练步骤（由 32,768 个输入token和 32,768 个目标token组成）花费了 433 毫秒，多查询模型花费了 425 毫秒。除以 32,768，发现每个（输入token + 目标token）的训练时间为 13.2μs。

使用 128 个token的源序列长度和 128 个目标序列长度对一批 1024 个序列（每个核心 128 个）运行增量贪婪推理。对于baseline模型，模型的编码器部分花费了 222 毫秒，解码器的每个增量步骤花费了 47 毫秒。除以相应的token数量，发现编码器的平均推理时间为每个token 1.7μs，解码器的平均推理时间为每个token 46μs。对于多查询模型，编码器每步花费 195ms，解码器每步花费 3.9ms，摊销的每个token成本分别为 1.5μs 和 3.8μs。

| Attention Type    | Training | Inference enc. + dec. | Beam-4 Search enc. + dec. |
| :---------------- | :------: | :-------------------: | :-----------------------: |
| multi-head        |   13.2   |       1.7 + 46        |         2.0 + 203         |
| multi-query       | **13.0** |   **1.5** + **3.8**   |     **1.6** + **32**      |
| multi-head local  |   13.2   |       1.7 + 23        |         1.9 + 47          |
| multi-query local | **13.0** |   **1.5** + **3.3**   |     **1.6** + **16**      |

列出的值以每个输出token的 TPUv2 微秒为单位。

> 本文提出了训练可以并行计算，并且算术强度比内存访问多的多，和gpu/tpu的特性相符合，但是推理时无法进行并行计算，因此内存会成为推理时的瓶颈。
