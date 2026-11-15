# 联合成本预算控制机制


DSpark 一轮并行生成整个 draft block。本实现固定 gamma=5，在线选择验证预算 V，并将完整 drafting 成本计入选择；它没有在线切换 gamma，也不是 EAGLE 自回归 drafter 的逐层提前停止。

预算目标是：

```text
选择 V，最大化：
    (batch_size + 所选 draft 前缀的预计接受 token 数)
    / 实测的整轮时间(draft_block=5, verify_budget=V)
```

[dspark_live_cost.py](../../python/sglang/srt/speculative/dspark_components/dspark_live_cost.py) 管理成本，[dspark_worker_v2.py](../../python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py) 接入运行时，原始 [dspark_planner.py](../../python/sglang/srt/speculative/dspark_components/dspark_planner.py) 负责根据 confidence 分配合法前缀。

### 用简单公式理解预算选择

这里的“预算”是**当前 batch 总共验证多少个 draft token**，不是预先规定一个毫秒上限。控制器选择预计吞吐最高的预算；联合成本指把 drafting、verification 和 commit 的成本一起考虑，而不是分别调节两个模型的生成长度。

**第一步：定义能调节的量。** 设当前实际运行的请求数为 $B$，每个请求固定生成 $\gamma=5$ 个 draft token；请求 $i$ 选择验证前 $l_i$ 个，满足 $0\le l_i\le5$。则总 draft 验证预算为：

$$
V=\sum_{i=1}^{B}l_i,\qquad 0\le V\le5B.
$$

每个请求还需要一个 root/基础 token 的验证位置，所以送入验证的逻辑 token 数为 $B+V$，实际 CUDA graph 可能使用补齐后的档位。$V=0$ 表示只验证 root，**这时仍然已经支付了完整 draft block 的成本**。$B$ 是这一轮的实际 batch size，不一定等于客户端设置的并发数。

**第二步：估计多验证一些 token 能得到多少收益。** 令 $p_{i,j}$ 表示 confidence head 对请求 $i$ 第 $j$ 个 draft 位置的接受信心。代码用累乘得到“前 $j$ 个都能接受”的估计概率：

$$
q_{i,j}=\prod_{k=1}^{j}p_{i,k}.
$$

例如信心值为 $[0.9,0.8,0.5]$，则前缀存活概率为 $[0.9,0.72,0.36]$。验证这三个位置的预计接受 draft 数是 $0.9+0.72+0.36=1.98$，不是简单地计为 3。

因此，给定每个请求的验证长度，整轮预计有效输出为：

$$
\widehat N(l_1,\ldots,l_B)
=B+\sum_{i=1}^{B}\sum_{j=1}^{l_i}q_{i,j}.
$$

其中 $B$ 是每请求一个基础输出 token 的收益模型，求和项是额外接受的 draft token 数。它是调度用的预测，不保证实际接受数相同；EOS、输出长度上限也可能使实际输出减少。

对于同一个总预算 $V$，planner 把所有请求的 $q_{i,j}$ 从高到低排序，累加前 $V$ 个，得到最佳预计收益 $\widehat N^*(V)$。因为同一请求的前缀概率随位置不增，可以优先把预算分给更可信的前缀；运行时的前缀分配器保证不能跳过前面的 token 单独验证后面的 token。实现还会过滤低于 `survival_eps` 的候选。

**第三步：测量整轮成本，而不是只算 target 验证时间。** 记当前平均上下文长度的分桶为 $c=\lfloor\overline L/1024\rfloor$，成本曲线为：

$$
\widehat T(B,c,V)
=\text{该 batch、上下文档位和预算下的实测整轮 GPU 时间}.
$$

它可以概念性地理解为 $T_{\rm draft}+T_{\rm verify}+T_{\rm commit}$，但实现通过 CUDA events **直接测量完整区间**，没有独立计时三项后再相加。draft 固定生成 5 个位置，其成本始终包含在分母中；缩短验证长度并不会省掉这部分成本。

每个 $(B,c)$ 先探索 $V\in\{0,B,2B,3B,4B,5B\}$，每个档位至少取得 3 次有效测量，才启用成本曲线。每个成本单元前 3 次取最小值，后续以 0.05 的权重平滑更新，并将单次更新样本上限限制为当前估计的两倍。曲线会做单调化；未直接测量的预算沿用最近一个更大预算的成本。例如只测过 $V=64$ 和 $128$ 时，$V=100$ 按 $128$ 的成本估计，不假设 GPU 时间按 token 数线性减少。

**第四步：选择预计吞吐最高的预算。** 最终决策为：

$$
V^*=\underset{V}{\arg\max}\;
\frac{\widehat N^*(V)}{\widehat T(B,c,V)}.
$$

实际只在 planner 的有效候选预算范围内搜索。下面是一个 **$B=64$ 的假设示例，不是实验实测数据**，仅列几个候选点：

| 总 draft 验证预算 $V$ | 预计有效输出 $\widehat N^*(V)$ | 整轮成本 | 预计输出速度 |
| ---: | ---: | ---: | ---: |
| 0 | 64 | 16 ms | 4000 token/s |
| 64 | 121.6 | 18 ms | 6756 token/s |
| 128 | 166.4 | 22 ms | 7564 token/s |
| 320（全部验证） | 204.8 | 36 ms | 5689 token/s |

在这几个候选点中会选择 $V=128$：相比全部验证，预计每轮输出少了一些，但整轮耗时减少得更多，最终每秒输出反而更高。$128/64=2$ 只是平均每请求验证 2 个 draft token，实际可以给高信心请求更多、低信心请求更少。

也可以用一个简单条件理解“是否值得多验证”。若当前收益和成本是 $N,T$，增加预算带来的增量是 $\Delta N,\Delta T$，则只有满足下式，预计吞吐才会上升：

$$
\frac{N+\Delta N}{T+\Delta T}>\frac NT
\quad\Longleftrightarrow\quad
T\Delta N>N\Delta T.
$$

代码直接比较各预算的完整比值，没有按这个增量公式逐 token 决策。这个公式只是说明：新增接受收益必须足以补偿新增成本；若处于同一个 CUDA graph 成本平台，增加验证位置可能几乎不增加耗时。

**与代码的对应关系：**

- `dspark_planner.py::_two_steps_prior_survival()`：用 `cumprod` 计算前缀存活概率。overlap 模式使用流水线延迟对齐的历史 confidence；不匹配的请求代次按全 1 回退，因此它不是提前知道本轮真实接受结果。
- `dspark_planner.py::compute_verify_token_budget()`：排序、前缀求和得到 `tau_star`，即上述 $\widehat N^*(V)$；计算 `theta = tau_star * sps` 并取 `argmax`。这里 `sps` 是整轮时间的倒数 $1/\widehat T$，不是输出 token/s。
- `dspark_live_cost.py::BlockCostHistory.curve()`：生成按 $(B,c,V)$ 区分的实测成本曲线，并转换为 planner 使用的 `sps`。
- `dspark_live_cost.py::LiveBlockCost.prepare()`：在完整验证、探索校准和按成本优化之间切换。本次配置中 $B<32$ 直接完整验证，不进行上述最优预算搜索。

因此，“根据硬件负载动态调整”在这里具体是：**通过实际 batch size、上下文档位和持续更新的 GPU 耗时反映运行成本，再重新选择验证预算**；没有读取 GPU 利用率百分比，也没有直接用客户端并发数查一个固定长度表。

### 运行时接入流程

```python
# 概念伪代码；实际接入点见上述源文件。
prepare_batch(batch):
    costs = history.lookup(batch_size, context_bin)
    if batch_size < 32:
        force_full_verify()                  # 不在短暂尾部反复探索
    elif costs_not_ready:
        probe_one_budget_from_zero_to_full() # root-only 仍支付 drafting 成本
    else:
        planner.use_whole_round_costs(costs)
    original_planner.prepare_budget()        # 保留 confidence 的延迟对齐

run_decode(batch):
    start_cuda_event()
    draft = original_dspark_draft()
    verify = original_target_verify(selected_prefixes(draft))
    original_commit(verify)
    record_cuda_duration_and_actual_budget()
```

- `BlockCostHistory` 按 batch size、1024-token 上下文区间、总验证预算存储实测成本。每个均匀预算至少测三次；root-only 单独测量，不能误当成免费操作。
- 未测预算采用相邻较大预算的成本，保留 CUDA graph 的阶梯效应；不会假设任意少验一个 token 都线性省时。
- GPU events 包围 draft、verify、commit 的工作流。额外的 begin-to-begin 墙钟诊断单独保存，不混入 GPU 成本曲线。
- 完整预算使用原始 uniform layout；小 batch（本次阈值 32）回退为原始完整验证。
- overlap 模式在 scheduler 的预算准备回调前安装成本曲线，保留原始 confidence relay/carry 延迟。worker 消费已准备的分桶，避免把上一批成本策略错用于下一批。
- DSV4 GPU-only attention 不一定提供 CPU 长度镜像，此时用已提交请求长度作为粗粒度上下文提示，可能落后少量在途 tokens；不会为成本控制添加 D2H 同步。
- 缓存身份包含模型、TP、运行时、实现代码、相关配置和调度模式。当前限制 DP1 / PP1；实际验证为 TP8。在线探索有冷启动开销，本次正收益是在请求预热之后测得。

