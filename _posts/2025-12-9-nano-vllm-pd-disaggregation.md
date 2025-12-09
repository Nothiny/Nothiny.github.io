---
title:      "深入理解 LLM 推理中的 Prefill-Decode 分离调度"
date:       2025-02-09 12:00:00
header-img: img/bg-little-universe.jpg
tags:
    - hello world
---

> 本文基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 项目的实现，深入剖析 PD（Prefill-Decode）分离调度的设计思想与实现细节。代码位于https://github.com/nothiny/nano-vllm

## 一、背景：为什么需要 PD 分离？

### 1.1 LLM 推理的两个阶段

大语言模型（LLM）的推理过程可以分为两个截然不同的阶段：

| 阶段 | 特点 | 计算特性 |
|------|------|----------|
| **Prefill（预填充）** | 并行处理整个 prompt | 计算密集型（Compute-bound） |
| **Decode（解码）** | 逐 token 自回归生成 | 内存带宽密集型（Memory-bound） |

用文字归纳两阶段的差异：Prefill 需一次性看完整个 prompt，适合大批量并行，计算密度高；Decode 逐 token 自回归，批量小、对内存带宽敏感，容易受 Prefill 阻塞。

### 1.2 传统调度的问题

在传统的调度方式中，Prefill 和 Decode 混合执行，这会导致：

1. **资源竞争**：Prefill 的高计算需求会阻塞 Decode 请求，导致首 token 延迟（TTFT）增加
2. **批次大小受限**：为了维持低延迟，难以最大化 GPU 利用率
3. **调度灵活性差**：无法针对不同阶段的特点进行优化

### 1.3 PD 分离的优势

PD 分离调度将两个阶段完全解耦：

- **更低的延迟**：Decode 批次不会被长 Prefill 阻塞
- **更高的吞吐**：可以分别优化两个阶段的批次大小
- **更灵活的调度**：支持不同的调度策略（Decode 优先 / Prefill 优先）

## 二、核心架构设计

### 2.1 整体架构

nano-vllm 的 PD 分离调度由三个核心组件构成（去掉文本画图，用文字拆解）：
- **PDScheduler**：维护 `waiting / prefilling / decoding` 三个队列，决定本轮执行 Prefill 还是 Decode，并产生 `prefill_info`（chunk 边界）。
- **BlockManager**：管理 KVCache 块的分配、复用与抢占，配合前缀缓存，确保 Prefill/Decode 都有可用的 KV 空间。
- **ModelRunner**：根据调度结果执行前向；Prefill 时按 chunk 构造输入并填充 KV，Decode 时仅处理最后一个 token；采样器按阶段选择性采样。

### 2.2 序列状态机

每个序列（Sequence）的状态流转（用步骤描述而非文本图）：
1. **WAITING**：新请求进入等待队列，尚未分配 KV 块。
2. **PREFILLING**：被调度进行（分块）Prefill；若被抢占，会重置到此状态以重建 KV。
3. **DECODING**：Prefill 完成后进入解码队列；如追加 KV 失败被抢占，同样回到 PREFILLING 重新写入上下文。
4. **FINISHED**：遇到 EOS 或达到 `max_tokens`，释放 KV，生命周期结束。

**关键代码实现：**

```python
class SequenceStatus(Enum):
    WAITING = auto()      # 等待prefill
    PREFILLING = auto()   # 正在进行prefill（用于chunked prefill）
    DECODING = auto()     # 正在decode
    RUNNING = auto()      # 兼容旧调度器
    FINISHED = auto()     # 已完成
```

### 2.3 三队列设计

与传统调度器的两队列（waiting + running）不同，PD 调度器采用三队列设计：

| 队列 | 作用 | 入队条件 | 出队条件 |
|------|------|----------|----------|
| `waiting` | 等待 Prefill | 新请求到达 | 开始 Prefill |
| `prefilling` | Chunked Prefill | Prefill 未完成 | Prefill 完成 |
| `decoding` | 等待/正在 Decode | Prefill 完成 | 生成结束 |

```python
class PDScheduler:
    def __init__(self, config: Config):
        # 分离的队列
        self.waiting: deque[Sequence] = deque()      # 等待prefill的序列
        self.prefilling: deque[Sequence] = deque()   # 正在进行chunked prefill的序列
        self.decoding: deque[Sequence] = deque()     # 等待/正在decode的序列
```

### 2.4 调度循环如何落地

- `LLMEngine.step()` 调用 `scheduler.schedule()`，决定执行 Prefill 还是 Decode，并返回 `prefill_info`（chunk 边界）。
- `ModelRunner.run()` 根据 `is_prefill` 分支构造输入；Prefill 写入 KV，Decode 仅处理最后一个 token。
- `scheduler.postprocess()` Prefill 时只对完成序列采样并转入 `decoding`；Decode 时追加 token 并判断结束。
- 每轮循环独立选择阶段，避免 Prefill/Decode 混杂带来的资源互相争用

## 三、Chunked Prefill：长 Prompt 的分块处理

### 3.1 为什么需要 Chunked Prefill？

当 prompt 很长时（如 4K+ tokens），一次性 Prefill 会：
- 占用大量 GPU 显存
- 阻塞其他请求较长时间
- 导致首 token 延迟飙升

Chunked Prefill 将长 prompt 分块处理，每次只处理 `prefill_chunk_size` 个 token。

### 3.2 实现细节

```python
def _schedule_prefill(self) -> tuple[list[Sequence], dict]:
    """
    调度Prefill批次，支持Chunked Prefill
    """
    scheduled_seqs = []
    prefill_info = {
        "chunk_starts": [],  # 每个序列的起始位置
        "chunk_ends": [],    # 每个序列的结束位置
    }
    num_batched_tokens = 0
    
    # 首先处理正在进行chunked prefill的序列
    while self.prefilling:
        seq = self.prefilling.popleft()
        
        # 计算本次chunk的大小
        remaining = seq.remaining_prefill_tokens
        chunk_size = min(remaining, self.prefill_chunk_size)
        
        if num_batched_tokens + chunk_size > self.max_prefill_batch_tokens:
            remaining_prefilling.append(seq)
            continue
        
        # ... 分配blocks并调度
        
        chunk_start = seq.num_prefilled_tokens
        chunk_end = seq.num_prefilled_tokens + chunk_size
        
        prefill_info["chunk_starts"].append(chunk_start)
        prefill_info["chunk_ends"].append(chunk_end)
        
        num_batched_tokens += chunk_size
        scheduled_seqs.append(seq)
    
    return scheduled_seqs, prefill_info
```

### 3.3 Chunked Prefill 执行流程

文字化的示例（假设 prompt=10000 tokens，`prefill_chunk_size=4096`）：
- **第 1 块**：处理 [0,4096)，状态 `WAITING → PREFILLING`，未完成，放回 `prefilling` 队列。
- **第 2 块**：处理 [4096,8192)，状态仍为 `PREFILLING`，未完成，再次放回。
- **第 3 块**：处理 [8192,10000)，Prefill 完成，状态 `PREFILLING → DECODING`，进入解码队列并对完成序列采样首个 token。

### 3.4 Prefill 采样策略

- Prefill 阶段并非对所有序列采样，只对“本次 chunk 后恰好 Prefill 完成”的序列采样首个 token。
- 未完成的序列不采样，直接回到 `prefilling` 队列继续下一块，减少无效计算与采样开销。
- 这一逻辑体现在 `ModelRunner.run()`：Prefill 分支会根据 `prefill_info["chunk_ends"]` 筛选完成的序列取样。

## 四、调度策略

### 4.1 两种策略

nano-vllm 支持两种调度策略：

#### Decode First（默认）

优先保证正在生成的请求，减少生成延迟：

```python
if self.schedule_policy == "decode_first":
    # 只有当decode队列足够大，或者没有prefill工作时才做decode
    has_enough_decode = len(self.decoding) >= self.min_decode_batch_size
    has_prefill = self._has_prefill_work()
    
    if self._has_decode_work() and (has_enough_decode or not has_prefill):
        seqs = self._schedule_decode()
        if seqs:
            return seqs, False, None
    
    if has_prefill:
        seqs, prefill_info = self._schedule_prefill()
        if seqs:
            return seqs, True, prefill_info
```

#### Prefill First

优先处理新请求，提高首 token 响应速度：

```python
else:  # prefill_first
    if self._has_prefill_work():
        seqs, prefill_info = self._schedule_prefill()
        if seqs:
            return seqs, True, prefill_info
    
    if self._has_decode_work():
        seqs = self._schedule_decode()
        if seqs:
            return seqs, False, None
```

### 4.2 策略对比

| 指标 | Decode First | Prefill First |
|------|--------------|---------------|
| TTFT（首token延迟） | 较高 | 较低 |
| TPOT（生成延迟） | 较低 | 较高 |
| 适用场景 | 长文本生成 | 交互式对话 |

## 五、内存管理与抢占机制

### 5.1 Block 分配

PD 调度器与 BlockManager 紧密配合，实现 KV Cache 的动态管理：

```python
def _try_allocate_blocks(self, seq: Sequence, num_blocks: int) -> bool:
    """尝试为序列分配更多blocks"""
    if len(self.block_manager.free_block_ids) < num_blocks:
        return False
    
    for _ in range(num_blocks):
        block_id = self.block_manager.free_block_ids[0]
        self.block_manager._allocate_block(block_id)
        seq.block_table.append(block_id)
    return True
```

### 5.2 抢占机制

当 GPU 显存不足时，PD 调度器会抢占低优先级序列：

```python
def _preempt(self, seq: Sequence):
    """抢占序列"""
    if seq.is_prefill_complete:
        # 如果prefill已完成，回退到prefilling队列重新开始
        seq.status = SequenceStatus.PREFILLING
        seq.num_prefilled_tokens = 0
        self.block_manager.deallocate(seq)
        self.prefilling.appendleft(seq)
    else:
        # 否则回退到waiting队列
        seq.status = SequenceStatus.WAITING
        seq.num_prefilled_tokens = 0
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
```

抢占后的序列会被放回队列头部，保证公平性。

### 5.3 延迟与吞吐的取舍

- **TTFT（首 token 延迟）**：受 Prefill 阶段影响最大，Chunked Prefill 与 `prefill_first` 策略可以显著降低 TTFT。
- **TPOT/TPOT（生成速度）**：Decode 阶段更看重批大小；`decode_first` 搭配较大的 `max_decode_batch_size` 可提升吞吐。
- **抢占副作用**：被抢占的 Decode 序列需重新 Prefill（重建 KV），可能拉长尾部延迟；需要在 `max_prefill_batch_tokens`、`max_decode_batch_size`、`prefill_chunk_size` 上做场景化权衡。

## 六、与普通调度器的对比

### 6.1 代码结构对比

**普通调度器（scheduler.py）：**

```python
class Scheduler:
    def __init__(self, config: Config):
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()  # 只有两个队列
    
    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill 和 decode 交替执行
        # prefill
        while self.waiting and num_seqs < self.max_num_seqs:
            # ... 处理 waiting 队列
        if scheduled_seqs:
            return scheduled_seqs, True
        
        # decode
        while self.running and num_seqs < self.max_num_seqs:
            # ... 处理 running 队列
        return scheduled_seqs, False
```

**PD 分离调度器（pd_scheduler.py）：**

```python
class PDScheduler:
    def __init__(self, config: Config):
        self.waiting: deque[Sequence] = deque()
        self.prefilling: deque[Sequence] = deque()  # 三个队列
        self.decoding: deque[Sequence] = deque()
    
    def schedule(self) -> tuple[list[Sequence], bool, dict | None]:
        # 根据策略选择 prefill 或 decode
        if self.schedule_policy == "decode_first":
            # ... decode 优先逻辑
        else:
            # ... prefill 优先逻辑
```

### 6.2 核心差异

| 维度 | 普通调度器 | PD 分离调度器 |
|------|------------|---------------|
| 队列数量 | 2（waiting + running） | 3（waiting + prefilling + decoding） |
| Chunked Prefill | 不支持 | 支持 |
| 调度策略 | 固定（prefill 优先） | 可配置 |
| 状态管理 | 简单 | 精细 |
| 返回值 | (seqs, is_prefill) | (seqs, is_prefill, prefill_info) |

## 七、配置参数详解

nano-vllm 提供了丰富的配置参数来调优 PD 分离调度：

```python
@dataclass
class Config:
    # PD分离相关配置
    enable_pd_disaggregation: bool = False  # 是否启用PD分离
    prefill_chunk_size: int = 4096          # chunked prefill的块大小
    max_prefill_batch_tokens: int = 8192    # prefill阶段最大batch token数
    max_decode_batch_size: int = 256        # decode阶段最大batch大小
    min_decode_batch_size: int = 1          # decode阶段最小batch大小
    pd_schedule_policy: str = "decode_first"  # 调度策略
```

| 参数 | 作用 | 调优建议 |
|------|------|----------|
| `prefill_chunk_size` | 控制每次 Prefill 的 token 数 | 较小值降低延迟，较大值提高吞吐 |
| `max_prefill_batch_tokens` | Prefill 批次的最大 token 数 | 根据 GPU 显存调整 |
| `max_decode_batch_size` | Decode 批次的最大序列数 | 增大可提高吞吐 |
| `min_decode_batch_size` | Decode 批次的最小序列数 | 用于 decode_first 策略的触发阈值 |

### 7.1 实战调优建议

- 交互式对话：选择 `prefill_first`，减小 `prefill_chunk_size`（如 1024/2048），保证首 token 快速返回。
- 长文本生成：选择 `decode_first`，放大 `max_decode_batch_size` 与 `max_prefill_batch_tokens`，确保 decode 批次饱满。
- 显存吃紧：适度减小 `max_prefill_batch_tokens`，必要时调低 `max_decode_batch_size`，降低抢占频率。
- 观察指标：重点看 TTFT、decode tokens/s、抢占次数（可在日志或监控中补充埋点）。

## 八、性能实测（Qwen3-0.6B，单 GPU）

测试环境：单卡（12GB 5070），输入 128 序列（长度 100–512），输出 50–256，`max_model_len=4096`，`enforce_eager=False`。

### 8.1 吞吐对比（bench_pd_compare.py）

| 模式 | 总时长(s) | 吞吐(tok/s) | 提升 |
|------|-----------|-------------|------|
| Normal | 3.93 | 5030.89 | 1.00x |
| PD-decode_first | 2.67 | 7406.39 | **1.47x** |
| PD-prefill_first | 3.77 | 5237.34 | 1.04x |

观察：
- `decode_first` 提升最明显（+47% 吞吐），Decode 批次数更多（416 vs prefill_first 的 272），更好填满生成阶段。
- `prefill_first` 吞吐接近 Normal，但仍小幅提升；适合优先响应新请求。

### 8.2 TTFT 对比（bench_ttft.py，64 序列，输入 256，输出 32）

| 模式 | 平均 TTFT(ms) | P50(ms) | P90(ms) |
|------|---------------|---------|---------|
| Normal | 356.59 | 356.59 | 356.65 |
| PD-prefill_first | **322.31** | 401.27 | 401.32 |
| PD-decode_first | 344.17 | 523.10 | 523.14 |

观察：
- 平均 TTFT：`prefill_first` 最优（≈-9.6%），`decode_first` 略优于 Normal（≈-3.5%）。
- 分布差异：`decode_first` 由于解码批次增大，P50/P90 略高；`prefill_first` 更偏向前期响应。

实用结论：
- 追求吞吐：选 `decode_first`，适当放大 `max_decode_batch_size`。
- 追求首 token 体验或交互式：选 `prefill_first`，结合较小的 `prefill_chunk_size`。

## 九、使用示例

### 8.1 启用 PD 分离调度

```python
from nanovllm import LLM, SamplingParams

# 启用 PD 分离调度
llm = LLM(
    "/path/to/model",
    enable_pd_disaggregation=True,
    prefill_chunk_size=2048,
    max_prefill_batch_tokens=4096,
    max_decode_batch_size=128,
    pd_schedule_policy="decode_first"
)

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, how are you?", "What is the meaning of life?"]
outputs = llm.generate(prompts, sampling_params)
```

### 8.2 对比测试

```python
# 普通调度
llm_normal = LLM("/path/to/model", enable_pd_disaggregation=False)

# PD 分离调度
llm_pd = LLM("/path/to/model", enable_pd_disaggregation=True)

# 对比长 prompt 场景下的性能差异
long_prompts = ["..." * 4000 for _ in range(10)]  # 10个长 prompt
```

## 十、总结

nano-vllm 的 PD 分离调度实现展示了一种优雅的 LLM 推理优化方案：

1. **清晰的架构设计**：三队列 + 状态机，代码简洁易懂
2. **灵活的调度策略**：支持 decode_first 和 prefill_first
3. **高效的内存管理**：Chunked Prefill + 抢占机制
4. **良好的扩展性**：可配置参数丰富，易于调优

这种设计思想在 vLLM、SGLang 等主流推理框架中都有体现，是理解现代 LLM 推理系统的重要基础。

## 十一、参考资料

- [nano-vllm GitHub](https://github.com/GeeeekExplorer/nano-vllm)
- [vLLM: Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/abs/2311.18677)

