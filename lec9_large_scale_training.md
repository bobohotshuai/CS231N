# 课程笔记：大规模 GPU 训练（Lecture 9）

## 一、GPU 集群概述

![GPU Cluster 概览](/lec9-1.jpg)

**目标**：在一个巨大的 GPU 集群上训练一个神经网络

**集群规模统计**：
- 24,576 个 GPU
- 1.875 PB GPU 内存
- 415M FP32 核心
- 13M Tensor 核心
- 24.3 EFLOP/sec = 24.3 × 10¹⁸

集群由多个 GPU Pod 组成，每个 Pod 包含大量 GPU 芯片。

## 二、数据并行（Data Parallelism, DP）

![数据并行原理](/lec9-2.jpg)

**核心思想**：损失函数通常在 N 个样本的 minibatch 上求平均，因此可以将 MN 个样本的 minibatch 分配到 M 个 GPU 上。

**工作流程**：
1. 每个 GPU 拥有自己的模型副本和优化器
2. 每个 GPU 加载自己的数据批次
3. 每个 GPU 前向计算损失
4. 每个 GPU 反向计算梯度
5. **跨所有 GPU 平均梯度**
6. 每个 GPU 更新自己的权重

**关键优化**：步骤 (4) 和 (5) 可以并行执行！

由于梯度是线性的，每个 GPU 计算自己的梯度，然后对所有 GPU 的梯度求平均。

## 三、混合分片数据并行（Hybrid Shared Data Parallel, HSDP）

![HSDP 架构](/lec9-3.jpg)

将 N = M×K 个 GPU 分成 M 组，每组 K 个 GPU。

**核心策略**：
- 每组 K 个 GPU 执行 FSDP（完全分片数据并行），将模型权重分片到 K 个 GPU 上，K 可达 O(100) 个 GPU
- 在 M 组之间执行 DP（数据并行）

**多维并行**：在同一时间使用不同的并行策略！将 GPU 组织成 2D 网格。

**通信优化**：
- 组内 K 个 GPU 之间有 3× 通信：前向传 W，反向传 W + dL/dW，保持在同一 node/pod 内
- 跨 M 组之间仅 1× 通信：反向传 dL/dW，可使用较慢的通信

**示例**：M=2 组，每组 K=4 个 GPU

## 四、完全分片数据并行（Fully Shared Data Parallelism, FSDP）

![FSDP 工作流程](/lec9-4.jpg)

**核心思想**：将模型权重分片到多个 GPU 上。

每个权重 W_i 由一个 GPU 拥有，该 GPU 同时持有其梯度和优化器状态。

**执行流程**：
1. 层 i 前向前，拥有 W_i 的 GPU 将其广播到所有 GPU
2. 所有 GPU 执行层 i 的前向，然后删除本地的 W_i 副本
3. 层 i 反向前，所有者广播 W_i 到所有 GPU
4. 所有 GPU 执行层 i 的反向计算 dL/dW_i 并删除 W_i
5. 反向后，所有 GPU 将本地的 dL/dW_i 发送给所有者并删除
6. W_i 的所有者执行梯度更新

**关键优化**：在计算当前层时，提前获取下一层的权重。

**同时执行**：
- 发送梯度并更新 W_3
- 用 W_2 执行反向
- 获取 W_1

**优化技巧**：前向结束时不删除最后一层权重，避免立即重新发送。

*参考*：Rajbhandari et al, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", arXiv 2019

## 五、激活检查点（Activation Checkpointing）

![激活检查点原理 - 基础](/lec9-5.jpg)

**问题**：N² 的计算量很糟糕！

**思路**：不重新计算所有内容；每 C 层保存一个检查点。

网络中的每一层实际上是两个函数：
- **前向**：计算下一层激活 $A_{i+1} = F_i^{\rightarrow}(A_i)$
- **反向**：计算上一层梯度 $G_i = F_i^{\leftarrow}(A_i, G_{i+1})$

**复杂度分析**：
- 前向+反向：O(N) 计算，O(N) 内存
- 完全重计算：O(N²) 计算，O(1) 内存

![激活检查点优化](/lec9-6.jpg)

**优化策略**：
- C 个检查点：O(N²/C) 计算，O(C) 内存
- √N 个检查点：O(N√N) 计算，O(√N) 内存

在反向传播过程中，从最近的检查点重新计算需要的激活。

## 六、大规模训练实践指南

![训练策略概览](/lec9-7.jpg)

**HSDP + 激活检查点可以走得很远！**

**扩展配方**：
1. 对于 ~1B 参数模型，使用**数据并行**最多 ~128 GPU
2. 始终将每个 GPU 的批次大小设置为**最大以充分利用 GPU 内存**
3. 如果模型 >1B 参数，考虑 **FSDP**
4. 添加**激活检查点**以适应每个 GPU 更大的批次
5. 如果有 >256 GPU，考虑 **HSDP**
6. 如果有 >1K GPU，模型 >50B 参数，或序列长度 >16K，则使用更高级的策略（CP, PP, TP）

**问题**：有很多超参数需要调整！应该如何设置？

**解决方案**：最大化**模型 FLOPs 利用率（MFU）**

## 七、模型 FLOPs 利用率（MFU）

![MFU 定义](/lec9-8.jpg)

**理念**：GPU 理论峰值 FLOPs 中有多少比例被用于"有用"的模型计算？

**计算步骤**：
1. 计算 $\text{FLOP}_{\text{theoretical}}$ = 前向+反向中矩阵乘法的总 FLOPs 数量（可近似反向 = 2× 前向）（忽略非线性、归一化、逐元素操作如残差，它们在 FP32 核心上运行）
2. 查找 $\text{FLOP/sec}_{\text{theoretical}}$ = 设备的理论最大吞吐量（H100: 989 TFLOP/sec）
3. 计算 $t_{\text{theoretical}} = \text{FLOP}_{\text{theoretical}} / \text{FLOP/sec}_{\text{theoretical}}$
4. 测量 $t_{\text{actual}}$ = 完整迭代的实际时间（数据加载、前向、反向、优化器步骤）
5. $\text{MFU} = t_{\text{theoretical}} / t_{\text{actual}}$

## 八、上下文并行（Context Parallelism, CP）

![CP 原理 - Ring Attention](/lec9-9.jpg)

**（通常用于 Transformer）**

**核心思想**：Transformer 在长度为 S 的序列上操作。使用多个 GPU 处理一个长序列。

**QKV 投影**：与 MLP 相同，在序列上并行化，像 DP 一样同步梯度。

**注意力算子**：最难并行化

**（选项 1）Ring Attention**：分块并分布到 GPU 上。键/值内循环，查询外循环。实现复杂但可扩展到非常长的序列。

*参考*：Liu et al, "Ring Attention with Blockwise Transformers for Near-Infinite Context", arXiv 2023

![CP 原理 - Ulysses](/lec9-10.jpg)

**（选项 2）Ulysses**：不尝试分布注意力矩阵，而是在多头注意力中跨头并行化。更简单，但最大并行度 = 头数。

*参考*：Jacobs et al, "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models", arXiv 2023

## 九、流水线并行（Pipeline Parallelism, PP）

![流水线并行](/lec9-11.jpg)

**核心思想**：将模型的层跨 GPU 分割。在 GPU 边界的层之间复制激活。

**问题**：
- 顺序依赖；GPU 大部分时间处于空闲
- N 路 PP 的最大 MFU 为 1/N

**解决方案**：同时运行多个 **microbatch**，通过 GPU 流水线传输

*参考*：Huang et al, "Gpipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism", arXiv 2018

## 十、张量并行（Tensor Parallelism, TP）

![张量并行 - 两层示例](/lec9-12.jpg)

**核心思想**：分割模型层跨 GPU。

**4 路 TP 示例**：
- 层 1：$XW = Y$，其中 $W = [W_1, W_2, W_3, W_4]$，$Y = [Y_1, Y_2, Y_3, Y_4]$
- 层 2：$YU = Z$，其中 $U = [U_1; U_2; U_3; U_4]$（垂直堆叠）

**计算**：
$$Z = Y_1U_1 + Y_2U_2 + Y_3U_3 + Y_4U_4$$

**块形状**：
- X: [N×D]
- W: [D×D], 分成 [1×4] 块
- Y: [N×D], 分成 [1×4] 块  
- U: [D×D], 分成 [4×1] 块
- Z: [D×D], 单块 [1×1]

## 十一、N 维并行（ND Parallelism）

![N 维并行](/lec9-13.jpg)

**终极方案**：同时使用 TP、CP、PP 和 DP！

将 GPU 组织成 **4D 网格**。

GPU 在网格中的索引给出其在每个并行维度上的秩。

优化设置以**最大化 MFU**。

**示例：LLama3-405B**

| GPUs | TP | CP | PP | DP | Seq. Len. | Batch size/DP | Tokens/Batch | TFLOPs/GPU | BF16 MFU |
|------|----|----|----|----|-----------|---------------|--------------|------------|----------|
| 8,192 | 8 | 1 | 16 | 64 | 8,192 | 32 | 16M | 430 | 43% |
| 16,384 | 8 | 1 | 16 | 128 | 8,192 | 16 | 16M | 400 | 41% |
| 16,384 | 8 | 16 | 16 | 8 | 131,072 | 16 | 16M | 380 | 38% |

*参考*：Llama Team, "The Llama3 Herd of Models", arXiv 2024

---

**关键要点**：
- 数据并行是最简单的策略，适用于中等规模训练
- FSDP 和 HSDP 通过分片权重节省内存
- 激活检查点以计算换内存
- MFU 是衡量训练效率的关键指标
- 超大规模训练需要组合多种并行策略（TP、CP、PP、DP）
- 所有策略的目标都是最大化 GPU 利用率

*图片来源*：Stanford CS231n 课程材料
