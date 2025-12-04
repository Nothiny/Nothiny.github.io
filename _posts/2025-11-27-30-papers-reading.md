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

## FlashAttention: Fast and Memory-Efficient Exact Attention  with IO-Awareness

## FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning 

![img](https://pica.zhimg.com/v2-025a32657c6aef99a0edb1ca95118d18_r.jpg)

Flash Attention系列论文提出了一种创新的注意力机制优化方法,通过分块(tiling)计算策略显著加速了Transformer模型中attention的计算过程。这一方法的核心思想是充分利用GPU的内存层次结构,通过精心设计的分块策略来减少高带宽内存(HBM)的访问次数。

**Flash Attention的基本原理:**

传统的attention计算需要将完整的Q、K、V矩阵加载到GPU内存中,计算QK^T后再进行softmax归一化,最后与V相乘得到输出。这个过程涉及大量的内存读写操作,成为计算瓶颈。Flash Attention通过将Q、K、V矩阵分割成更小的块,每次只将一小块数据加载到GPU的片上内存(SRAM)中进行计算,从而大幅减少了对慢速HBM的访问。

**Flash Attention 2的优化策略:**

在Flash Attention 2中，计算流程采用了更优化的循环嵌套结构:外层循环遍历Q矩阵的分块,内层循环遍历K和V矩阵的分块。这种组织方式相比第一代进行了调整,能够更好地利用GPU的并行计算能力，减少同步开销,并提高计算单元的利用率。具体来说,对于每个Q块,依次处理所有的K、V块,通过累积的方式逐步计算出最终的attention输出。

**Online Softmax算法:**

传统的softmax计算通常需要三次遍历(3-pass):第一次计算最大值用于数值稳定性,第二次计算指数和,第三次进行归一化。这种多次遍历会产生额外的内存访问开销。Flash Attention采用了online softmax(也称为safe softmax)算法,将这个过程优化为单次遍历(1-pass)。

Online softmax的关键在于维护一个运行时的最大值和累积和,在处理每个新的分块时,动态更新这些统计量,并对之前已计算的结果进行相应的重新缩放。具体来说,当处理新的K、V块时,会计算当前块的注意力分数,更新全局最大值,然后使用新的最大值对之前的累积结果和当前结果进行修正,最后累加到输出中。这种增量式的计算方式避免了多次完整遍历,显著降低了内存带宽需求。

**性能提升:**

通过结合分块计算和online softmax算法,Flash Attention 2实现了以下优势:

- **降低内存访问**: 将大部分中间计算保持在快速的SRAM中,减少了HBM访问次数
- **提高计算效率**: 优化的循环结构提升了GPU计算单元的利用率
- **保持数值精度**: 尽管采用了增量计算,仍能保证与标准attention完全相同的数值结果
- **支持长序列**: 内存效率的提升使得处理更长的序列成为可能

这些优化使得Flash Attention 2相比标准实现获得了数倍的加速,并且这种加速在长序列场景下更加明显,为大规模语言模型的训练和推理提供了重要的技术支撑。

## H_2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models

<img src="https://arxiv.org/html/2306.14048v3/x3.png" style="zoom:50%;" />

这篇论文主要介绍稀疏注意力计算，虽然和我现在目前的学习重合度不高，但是我还是简单看了一下，这篇论文是发现注意力计算中的部分token起到了比较大的作用，这有点类似二八定律，因此在推理中只保留这部分比较重要的token，同时驱逐不重要的token，能够保持性能的同时大幅度提高推理速度。

这项工作对长文本生成特别重要。通过只保留20-30%的KV Cache,H2O可以在几乎不损失性能的情况下,将推理速度提升数倍,并支持更长的上下文长度。这对于实际部署大语言模型非常有价值。

H2O提出了一种动态的KV Cache管理策略:

1. Heavy-Hitter识别: 通过追踪每个token的累积注意力分数,识别出那些持续获得高注意力权重的"重量级"token

2. 动态驱逐策略: 当KV Cache达到预设大小时,选择性地驱逐那些注意力分数较低的token,只保留:

   Heavy-hitter tokens(注意力权重大的关键token)  Recent tokens(最近生成的token)

3. 保留策略: 最近的token通常也很重要(局部性原理),所以即使它们当前的累积注意力分数不高,也会被保留

## EFFICIENTLY SCALING TRANSFORMER INFERENCE

这篇论文介绍了如何推理一个100b到500b的模型，感觉对我目前的帮助不是很大，因此我现在主要还是学习单卡推理阶段，对于张量并行，流水并行相关的研究还不是很多。

这篇论文讲解了如何在一个多卡的集群中并行attention计算以及并行FFN计算，然后还提出了相关的符号定义比如：计算FLOPs的分析内，存访问模式，通信开销的量化，并行策略的形式化描述，这对后续的研究起到了一个基础的导向作用，因此这篇论文对于理解推理流程还是十分重要的。

同时这篇论文指出了计算和内存的瓶颈计算方法，全面介绍了推理的流程和开销的来源。

## DistServe: Disaggregating Prefill and Decoding for Goodput-optimized  Large Language Model Serving

<img src="https://arxiv.org/html/2401.09670v3/x6.png" style="zoom:50%;" />

这篇文章使用分离prefill阶段和decode阶段来完成推理过程，但是我认为这个技术在几年之前都已经是业内共识（经常听别人讨论），本文的主要贡献可能是使用模拟器系统分析了pd分离对比vLLM以及 DeepSpeed-MII。本文做的pd分离在多卡并行的策略上是使用了prefill和decode分离到不同的gpu上运行，做了一个类似流水线似的工具。对比不同策略他们的性能差异发现目前pd分离还是有优势存在的，本文通过实验证明了在特定的场景下，pd分离仍然具有优势存在。

vllm目前还没有使用pd分离，而是集成了chunked-prefill，这个实现更加的简单，不需要跨集群通信，在需要低延迟的场景下更加的有优势。

## Efficient Memory Management for Large Language  Model Serving with PagedAttention

![](https://cdn.prod.website-files.com/618399cd49d125734c8dec95/661fcd418406a9fff4c1d563_xcYIBMkizsjzWrCADxMF6lRBFcyGPY2Lt6lNPqOjahqJDqKGqi1T_3xytwUn5SxXZeBWyVmO-xN3iBbURFlTYZ8zRa-HYnv6QQm-jhdoYMHCme3eEbvBqUpThd-ooS47Szv1IEdAGJ2RzVSFbcPZOF4.png)

这篇论文借鉴操作系统的分页内存的方式，来管理gpu中的显存，这样可以减少显存利用中的碎片问题（包括内部碎片，外部碎片，这个操作系统的相关概念类似）。

使用分页的方式将kvcache分成固定大小的块，每个kvcache由不同的非连续页组成，改变了以往的连续页的管理方式，此时碎片只有最后一个块中的部分，与操作系统中的概念相同。显存页可以按需申请，因此在推理时降低了对显存的需求。同时对不同页内的显存进行计数，使得不同的sequence可以共享同一个显存块（这里使用copy on write的方式共享），从而减少显存的利用。

当显存不足或者是有抢占式的sequence到来时，使用两种策略来对驱逐出去的kvcache进行重新生成，包括重计算或者保存到内存之中，论文也做了实验来比较这两种策略的差异。

## Triton: An Intermediate Language and Compiler for  Tiled Neural Network Computations

<img src="https://pica.zhimg.com/v2-0d18896899a4350455a37772f9a1528c_r.jpg" style="zoom: 67%;" />

论文中提出了Triton-C，一种类C的领域特定语言。现在的Triton改用Python作为前端语言，因为Python更容易被机器学习社区接受，学习成本更低。但是论文中提出的分块计算和广播等机制和python中的概念不谋而合。

Triton的核心特性比如分块计算、程序ID、张量广播等,是Triton编译器自己实现的，而不是直接用Python或NumPy的功能。Python只是作为语法的载体，让写kernel代码更简单。编译器后端是用C++/MLIR实现的，最终生成的是GPU机器码。

从Triton-C到Python-Triton的演进,主要是为了降低使用门槛,让更多研究者和工程师能够方便地写高性能GPU代码,而不需要深入了解CUDA编程的细节。
