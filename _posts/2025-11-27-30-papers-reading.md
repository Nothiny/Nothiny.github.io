---
title:      "记录最近阅读的30篇论文"
date:       2025-11-27 12:00:00
header-img: /img/back.jpg
tags:
    - paper-reading
---

# 记录最近阅读的30篇论文

## Attention Is All You Need

大模型的奠基之作，介绍了Scaled Dot-Product Attention，attention以及transformer架构，

<img src="https://arxiv.org/html/1706.03762v7/Figures/ModalNet-21.png" alt="img" style="zoom: 25%;" />

在 **Attention is All You Need**这篇论文中，作者首次提出了基于 Attention 的 Transformer 架构，并在 **机器翻译（Machine Translation）** 任务上进行了实验，显著提升了性能。该工作被认为是 **大语言模型（LLM）时代的起点**，因为其提出的自注意力机制（Self-Attention）有效解决了 RNN/CNN 在长序列建模中的诸多不足。

在计算注意力分数时，自注意力机制需要对序列中 **每一个 token 与序列中所有其他 token** 之间的关联进行计算，即计算 Query 和 Key 的点积：$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V$ 

其中矩阵 $QK^T$ 的形状为 $N \times N$，意味着：

- 对于长度为 $N$ 的序列，需要计算 $N^2$ 次 token 对之间的相似度
- 因此 Attention 的计算复杂度为 **$O(N^2)$**，随序列长度增长呈 **平方级增长**
- 无论两个 token 在序列上相距多远，它们之间都需要计算 attention 分数

这种全连接的注意力结构带来了强大的全局上下文建模能力，但代价是：

- **计算量与存储量巨大**（需要存储 $N^2$ attention matrix）
- 当序列长度从 2k → 16k 时，计算和显存开销急剧增长
- 限制了长上下文场景（如长文档理解、多轮对话、代码建模）的发展

因此，Transformer 的 attention 机制虽然拥有强大的全局建模能力，但其计算与显存开销随序列长度呈平方级增长，是长上下文场景中的主要瓶颈。这一限制直接推动了 FlashAttention 等高效注意力优化算法的诞生，也为 LLM 时代的系统优化研究奠定了基础。

## Fast Transformer Decoding: One Write-Head is All  You Need

传统的多头注意力（MHA）为每个注意力头分别维护一组 Key 和 Value，因此在推理阶段需要缓存所有头的 KV，这会带来巨大的显存开销，尤其是长序列场景下 KV Cache 会随着序列长度线性增长。MQA 的核心思想是所有的 Query 依然保持多头，但所有 Query 共享同一组 Key 和 Value。这样的设计显著减少了推理阶段的 KV 缓存大小，从而提高推理速度并减少显存消耗。

因为减少 Key 和 Value 的数量会降低注意力机制的表达能力，因此能否在不显著损失精度的前提下加速推理是研究的重要点。论文作者采用 uptraining 的方式，将已经训练好的 Transformer 模型用 MQA 结构继续训练，使其适应改变后的注意力结构。实验结果表明，模型在推理速度上有显著提升，而模型精度仅有非常轻微的下降，甚至在部分任务上几乎无损。

这些结果说明，多头注意力中存在大量冗余结构，尤其在 Key 和 Value 的维度上存在重复信息。在保证模型性能的前提下，可以减少这些冗余，从而提升推理效率。因此，在实际运行时并不需要完整保留所有 KV heads，这为后续高效注意力结构提供了基础。

## GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

![Refer to caption](https://arxiv.org/html/2305.13245v3/extracted/5314337/images/gmq_architecture.png)

这篇论文提出了 GQA（Grouped Query Attention），它是介于传统的多头注意力（MHA）和多查询注意力（MQA）之间的一种折衷设计。在 GQA 中，将多个 Query 头划分成若干组，每一组内的 Query 共享同一份 Key 和 Value，而不是像 MHA 那样每个 head 独立，也不是像 MQA 那样所有 head 全部共享同一个 KV。对于组内的处理，可通过对原来各个 head 的 Key 和 Value 进行平均或线性合成来生成该组共享的键和值。

因此，GQA 的两个极端情况分别对应：当组数等于 head 数时，退化为 MHA；当组数为 1 时，退化为 MQA。因此 GQA 可以被视为 MHA 和 MQA 之间的连续可调结构压缩方案。

GQA 的提出主要解决了 MQA 在表达能力上的潜在损失问题。虽然 MQA 可以显著减少推理阶段 KV Cache 的显存占用，从而加速模型推理，但过度共享 KV 会导致多头注意力之间的表达差异被抹平，从而可能带来精度下降。GQA 通过限制共享的粒度，使模型在降低 KV Cache 开销的同时，仍保留一定程度的 head 多样性，从而在推理效率和模型精度之间取得更好的平衡。

实验表明，GQA 在精度损失上通常比 MQA 更小，同时仍能获得显著的推理加速，特别适合大模型推理场景。在真实部署中，GQA 能够减少 KV Cache 大小，提高 GPU 利用率，使得更大的 batch、更长的上下文、更高的并发部署成为可能。因此，GQA 是现代大语言模型（例如 LLaMA、Qwen 等）普遍采用的注意力结构，对高性能推理至关重要。

总结而言，GQA 证明了注意力头之间存在大量冗余，并展示了减少冗余维度仍可以保持模型表达能力的可能性



## DHA: Learning Decoupled-Head Attention from Transformer Checkpoints via Adaptive Heads Fusion

![Refer to caption](https://arxiv.org/html/2406.06567v2/x1.png)

这篇论文提出了DHA(Decoupled-Head Attention,解耦头注意力)机制,主要解决大语言模型推理时的计算和内存开销问题。

论文设计了一种解耦头注意力机制,能够在不同层自适应地为键头和值头配置组共享,在性能和效率之间实现更好的平衡。

论文的创新点在于自适应头融合(Adaptive Heads Fusion):

- 通过三个阶段将传统多头注意力(MHA)快速转换为DHA:搜索、融合和持续预训练
- 首先对相似功能的注意力头进行聚类分组
- 然后通过线性融合相似头的参数,保留原始模型的知识
- 不同层根据其冗余程度分配不同数量的键头和值头

实验显示DHA只需要原始模型0.25%的预训练预算就能达到97.6%的性能,同时节省75%的KV缓存 。与Group-Query Attention(GQA)相比,DHA实现了5倍的训练加速和更好的性能。

简单来说,这篇论文提供了一种高效的方法,能够将已有的Transformer模型转换成更节省计算和内存的版本,且只需极少的额外训练成本。
