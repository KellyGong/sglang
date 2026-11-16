# DSpark 联合成本预算控制：运行、机制与结果

本分支为 DeepSeek-V4-Flash-0731 的 DSpark 增加在线整轮成本校准，在 SGLang 中动态选择每批请求的总验证预算，再由已有 confidence planner 分配各请求的验证前缀。draft 始终生成 5 个 token；不改模型权重，不需要重新训练。默认不开启此功能。

本分支基于实验的固定基础提交 `c053fc71d3bdb4334335260fa6f685356890d9eb`，不是基于当前 main 的移植。完整数学说明见 [MECHANISM.md](MECHANISM.md)。

## 已测结果

8 × H20-3e、TP8、DP1/PP1、FP8 KV、Marlin、overlap；每轮 512 请求，每请求固定输出 256 token，temperature=0、ignore_eos=true。每组运行三轮，吞吐取中位数；token 数按三轮正式请求的验证轮次加权，不包含预热。

| 并发 | 原生 tok/s | DSpark tok/s | 联合预算 tok/s | 相对 DSpark | DSpark → 联合预算平均验证 draft 数 |
|---:|---:|---:|---:|---:|---:|
| 16 | 1321.4 | 1835.1 | 1829.8 | -0.29% | 5 → 5.000 |
| 32 | 1944.9 | 2521.9 | 2534.0 | +0.48% | 5 → 4.828 |
| 64 | 2986.2 | 3215.5 | 3437.3 | **+6.90%** | **5 → 2.795** |
| 128 | 3894.5 | 3633.3 | 4170.3 | **+14.78%** | **5 → 2.646** |

平均验证 draft 数为每请求、每轮、不含 root；加入 root 后并发 64/128 分别为 3.795/3.646。各请求实际可取 0～5，不是固定验证长度。完整结果、长度分布和计算口径见 [results/REPORT.md](results/REPORT.md)。

并发 16 无加速证据，并发 32 的差异在波动范围内。并发 64 的 P95 延迟增加约 2.3%。并发 64 来自较早的独立会话；未随机交错 A/B，并发 128 每轮仅约四批请求。上述是历史硬件实验结果，不是本分支整理时重新运行的测量，不能保证其他硬件或工作负载有同等收益。

## 环境准备

需要 8 张能容纳此模型的 GPU、已下载的模型，以及与该基础提交兼容的 SGLang Python/CUDA/kernel 环境。模型 snapshot 为 `7872f01b1d1fe23eabc4c98b48bffcef5a386062`。历史包版本见 [packages.json](packages.json)，该文件是环境清单，不是可直接安装的锁文件；模型、CUDA、编译缓存和 Python 环境不随 Git 提供。请按仓库原有安装文档准备兼容环境。

```bash
git clone --branch dspark-joint-cost-budget https://github.com/KellyGong/sglang.git
cd sglang
# 先激活兼容该提交的 Python 环境，再指定实际路径。
export DSPARK_PY=/path/to/compatible/environment/bin/python
export DSPARK_MODEL_PATH=/path/to/DeepSeek-V4-Flash-0731/snapshot
source benchmarks/dspark_budget/env.sh
```

`env.sh` 将本仓库的 `python/` 加入 PYTHONPATH，无需依赖 EAGLE-4 或 SpecForge 目录。模型必须已在本地，环境默认启用离线模式。服务使用本机 31600/31601 端口，测试需独占相应 GPU 和端口；冷启动的内核编译可能需要较长时间，runner 最多等待服务就绪一小时。

## 运行一次完整对照

在仓库根目录执行；所有输出目录必须尚不存在。下面依次执行并发 64 的原生、DSpark、联合预算三组，各组包含三轮正式测量：

```bash
for variant in native dspark joint; do
  "$DSPARK_PY" benchmarks/dspark_budget/run_trial.py \
    --variant "$variant" --schedule overlap --concurrency 64 \
    --requests 512 --tokens 256 --repeats 3 \
    --min-free-slots 8 --live-min-batch 32 \
    --output "benchmarks/dspark_budget/runs/c64-$variant"
done
```

运行并发 16/32/128 的完整三组对照：

```bash
"$DSPARK_PY" benchmarks/dspark_budget/run_concurrency_matrix.py \
  --output benchmarks/dspark_budget/runs/concurrency-16-32-128
```

首轮预热 8×并发个请求，后续每轮预热 2×并发个请求，均生成 256 token，预热不计入正式吞吐。runner 保存启动参数、源码哈希、环境、逐请求响应、每轮 summary 和完成标记，并在结束或失败时停止自己启动的服务进程组。不要并行运行以上命令。

每轮结果为 `repeat*/summary.json`；矩阵运行完成后生成 `comparison.json`。`COMPLETE` 表示对应运行完成；如发生错误，应先查看 `server/server.log`。本交付脚本聚焦固定长度吞吐，不包含之前单独进行的自然 EOS 质量评测。

## 直接接入自己的服务

对于已有可运行的 DSpark 服务，在启动前设置：

```bash
export SGLANG_DSPARK_LIVE_COST_PATH=/persistent/path/live-cost.json
export SGLANG_DSPARK_LIVE_COST_OVERLAP=1
export SGLANG_DSPARK_LIVE_COST_MIN_BATCH_SIZE=32
export SGLANG_RAGGED_VERIFY_MODE=compact
```

模型必须带有训练好的 confidence head，使用 `--speculative-algorithm DSPARK --speculative-dspark-block-size 5`，且 DP1/PP1。缓存路径需可写；每个独立服务使用独立路径。关闭 overlap 时将 `SGLANG_DSPARK_LIVE_COST_OVERLAP` 设为 0，并加 `--disable-overlap-schedule`。不设置 `SGLANG_DSPARK_LIVE_COST_PATH` 即不启用成本控制。

控制器按实际 batch size 和平均上下文的 1024-token 桶记录成本，而不是读取 GPU 利用率。每个新档位先探索预算 0、B、2B、…、5B，成本就绪后选择“预计有效输出量 / 实测整轮耗时”最大的预算。实际 batch<32 时完整验证。校准有冷启动开销，历史收益在预热之后测得。

## 改动代码

- [dspark_live_cost.py](../../python/sglang/srt/speculative/dspark_components/dspark_live_cost.py)：新增 CUDA event 整轮计时、成本缓存、校准探索、阶梯成本曲线和小 batch 回退。
- [dspark_worker_v2.py](../../python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py)：在预算准备及 draft/verify/commit 流程接入控制器。
- [dspark_planner.py](../../python/sglang/srt/speculative/dspark_components/dspark_planner.py)：复用既有 confidence 前缀分配与吞吐优化逻辑，本分支没有修改该文件。
- [deepseek_v4.py](../../python/sglang/srt/models/deepseek_v4.py)：额外保留历史实验的可选加载线程数控制 `DSPARK_EXPERIMENT_LOADER_WORKERS`，仅影响启动加载；三组均设为 1，未设置时保持默认行为。
- [成本模型测试](../../test/registered/spec/dspark/test_dspark_live_cost.py)及[真实 planner 接入测试](../../test/registered/spec/dspark/test_dspark_cost_integration.py)：检查预算选择、root-only 成本、上下文分桶、overlap 准备与回退。

运行 CPU 测试：

```bash
"$DSPARK_PY" -m pytest -q \
  test/registered/spec/dspark/test_dspark_live_cost.py \
  test/registered/spec/dspark/test_dspark_cost_integration.py
```

原实验中未启用的候选词表和后续树形实验不在此分支内。历史 36 轮 summary/settings 位于 [results/measurements](results/measurements)，大体积原始响应和模型文件没有上传。重新运行会生成完整响应，以便重新审计。

本次交付校验：15 项 CPU 测试通过；三种启动配置、源码快照生成、模拟响应下的验证长度统计、脚本语法和文档链接检查通过。本次未重新启动 GPU 服务测量性能。

## 拟合成本模型与低负载结论（2026-11-16）

在联合预算之上的三处改动，以及“低负载时加大 draft 预算”的实测结论。和联合预算在计算上的区别，简明版见 [FITTED_COST.md](FITTED_COST.md)。8 × H20-3e、TP8、overlap，每请求 256 token、temperature=0、ignore_eos；每组 3 轮取中位数。

### 新增开关（默认全部关闭）

```bash
export SGLANG_DSPARK_LIVE_COST_PATH=/persistent/path/live-cost.json
export SGLANG_DSPARK_LIVE_COST_OVERLAP=1
export SGLANG_DSPARK_LIVE_COST_MODEL=fit          # 默认 cells，即原来的逐格测量
export SGLANG_DSPARK_LIVE_COST_MIN_BATCH_SIZE=12  # 拟合模型下可从 32 降到 12
export SGLANG_RAGGED_VERIFY_MODE=compact
export SGLANG_RAGGED_VERIFY_EXTRA_TIERS=2,3,...   # 额外的 compact verify 图档位（token 数）
```

推荐配置是 block size 5、`fit`、最小定价 batch 12。额外档位的上限取 `最大并发 × (gamma + 1)`，每个档位多捕获一张 CUDA graph。

- `fit`：一步的 GPU 时间拟合为 `fixed + per_request * bs + per_token * graph_tokens`。跑过的 (batch, 图档位) 用自己的实测值，没跑过的用拟合线，因此新 batch size 不需要再做一轮 0…5B 的探索。代码在 [dspark_step_cost.py](../../python/sglang/srt/speculative/dspark_components/dspark_step_cost.py) 和 `dspark_live_cost.py::FittedCostHistory`。
- 额外档位：默认只有 `captured_bs × (gamma + 1)` 这些档位，预算落在两档之间时多出来的 token 白算。本次用 `2..24` 全部、`26..48` 步长 2、`52..96` 步长 4、`104..` 步长 16。
- rank 间预算一致：每个 TP rank 各自对置信度取 argmax，档位变细后两个 rank 会落在不同图档位，NCCL 互相等待而挂死（并发 64 约 1400 步后复现）。现在由 rank 0 广播预算，对 `cells` 模型同样生效。

### 结果

| 并发 | DSpark tok/s | 联合预算（cells，≥32）| 拟合模型（≥12）| 拟合 / DSpark | 拟合 / 联合预算 |
|---:|---:|---:|---:|---:|---:|
| 1 | 331.8 | — | 329.6 | -0.7% | — |
| 2 | 477.8 | — | 479.8 | +0.4% | — |
| 4 | 689.1 | — | 671.1 | -2.6% | — |
| 8 | 924.0 | — | 898.3 | -2.8% | — |
| 16 | 1537.4 | — | 1711.6 | **+11.3%** | — |
| 32 | 2510.6 | 2531.4 | 2746.1 | **+9.4%** | +8.5% |
| 64 | 3181.7 | 3409.3 | 3567.9 | **+12.1%** | +4.7% |
| 128 | 3629.6 | 4141.4 | 4315.3 | **+18.9%** | +4.2% |

并发 1–16 用 `lowload_sweep.py`（一台 max-running=16 的服务，不设 min-free-slots）；并发 32–128 沿用上文协议（max-running=并发、min-free-slots=8、512 请求）。并发 1–8 批量低于 12，走完整验证，差异在两次基线之间的波动范围内（同一配置两台机器的基线相差 ±2%）。并发 16 的 +11.3% 在三个代码快照上各测一次，分别为 1711.6 / 1711.4 / 1710.7。表中并发 64、128 的拟合模型从 batch 0 起定价；设为 12 后重测为 3544.3（另一台机器）和 4323.9。

### 低负载加大 draft 预算：测过，没有收益

一步解码约为 7.3 ms 固定 verify + 每个验证 token 0.14–0.19 ms + 1.0 ms draft。并发 1 时一步 9.3 ms，多验证一个 token 至少要多拿回 0.06 个 token 才不亏。

- 加长：head 按 5 训练，第 6–9 位条件接受率 0.68 / 0.59 / 0.50 / 0.44，但 gamma=9 全验证的每步接受长度只从 3.66 到 3.86（短 block 剩下的易中 token 下一步本来也能拿到）。gamma=7 静态在并发 1 为 -1%（单次测量），gamma=9 加拟合定价在并发 1–8 为 -2%～-5%，在并发 64 只有 +3.4%（gamma=5 为 +12.1%）。
- 加路径：不实现树形 verify，先统计“首个被拒位置上 target 的 token 在 draft 分布里排第几、换成它之后的 Markov 续写能中几个”。价值最高的兄弟节点为每步 0.063 token，与验证它的成本持平，其余都更低。
- 链式第二段 draft（同一张宽度 5 的 draft 图以第 5 个 draft 为锚再跑一遍）：接受长度 4.04（单段 gamma=9 为 3.86），但 draft 多 0.7 ms，并发 1 每步 GPU 时间 10.7 ms 对 9.9 ms，净 -3%。
- 用当前步置信度同步决定验证长度：离线回放约 +8%（回放对短预算偏悲观、对基线也偏低，实际更小），但 draft 之后的主机同步让 verify 的启动准备不再被 draft 遮住，GPU 每步空转 1.75 ms，并发 1 每步 11.7 ms 对 9.9 ms。
- overlap 管线里两步之前的置信度对 1–8 个请求的批量基本是噪声，这是低于 12 不定价的原因。

链式第二段 draft 和同步门控的实验代码，以及本节的逐轮 summary，没有随分支提交。

### 运行

本节代码在分支 `dspark-fitted-cost`；环境准备同上，clone 时把分支名换成 `dspark-fitted-cost`。以下命令在仓库根目录执行。每个 `--variant` 是 `<名字>=<server.py 参数>`，独占 8 张 GPU 启动一台服务，测完后停止；输出目录必须尚不存在。

并发 1–16（一台 max-running=16 的服务依次测各并发）：

```bash
source benchmarks/dspark_budget/env.sh
TIERS=$(python3 -c "print(','.join(map(str, list(range(2,25))+list(range(26,49,2))+list(range(52,97,4)))))")
"$DSPARK_PY" benchmarks/dspark_budget/lowload_sweep.py --output benchmarks/dspark_budget/runs/fit --repeats 3 \
  --variant "base5=--block-size 5 --overlap" \
  --variant "fit5m12=--block-size 5 --overlap --live-cost --live-min-batch 12 --env SGLANG_DSPARK_LIVE_COST_MODEL=fit --env SGLANG_RAGGED_VERIFY_EXTRA_TIERS=$TIERS"
"$DSPARK_PY" benchmarks/dspark_budget/lowload_report.py benchmarks/dspark_budget/runs/fit --baseline base5
```

并发 32–128（沿用上文协议：max-running=并发、min-free-slots=8、512 请求；额外档位补到 `并发 × 6`）：

```bash
for C in 32 64 128; do
  HIGH=$(python3 -c "print(','.join(map(str, range(104, $C * 6 + 1, 16))))")
  "$DSPARK_PY" benchmarks/dspark_budget/lowload_sweep.py --output benchmarks/dspark_budget/runs/fit-c$C \
    --max-running $C --concurrency $C --requests-per-concurrency 1 --min-requests 512 \
    --first-warmup-multiplier 8 --repeats 3 \
    --variant "dspark5=--block-size 5 --overlap --min-free-slots 8" \
    --variant "joint5=--block-size 5 --overlap --min-free-slots 8 --live-cost --live-min-batch 32" \
    --variant "fit5m12=--block-size 5 --overlap --min-free-slots 8 --live-cost --live-min-batch 12 --env SGLANG_DSPARK_LIVE_COST_MODEL=fit --env SGLANG_RAGGED_VERIFY_EXTRA_TIERS=$TIERS,$HIGH"
  "$DSPARK_PY" benchmarks/dspark_budget/lowload_report.py benchmarks/dspark_budget/runs/fit-c$C --baseline dspark5
done
```

每个 variant 目录下有 `server.log`、`command.json`、`environment.json` 和各轮的 `c<并发>-r<轮次>/summary.json`。拟合出的系数在该目录 `live-cost-final.json` 的 `described` 字段（`fixed / per_request / per_token`，单位 ms），换硬件时先看这三个数。

分析脚本读 `measure.py` 在每轮目录里保存的 `server-info.json`，需要用 `--env` 打开对应的 debug dump：

| 脚本 | variant 参数 | 输出 |
|---|---|---|
| `cost_report.py` | `--env SGLANG_DSPARK_DEBUG_DUMP=core,step_cpu_time,step_gpu_time,draft_gpu_time,target_verify_gpu_time` | 一步解码的成本分解 |
| `calibration_report.py` | `--block-size 9 --overlap --env SGLANG_DSPARK_DEBUG_DUMP=all`（全验证） | 各 draft 位置的置信度与实际接受率 |
| `policy_sim.py` | 同上 | 不同验证策略的离线回放 |
| `shadow_tree_report.py` | `--shadow-tree`，读 variant 目录下的 `shadow-tree.json` | 兄弟 draft 路径的期望收益 |

```bash
"$DSPARK_PY" benchmarks/dspark_budget/lowload_sweep.py --output benchmarks/dspark_budget/runs/trace --concurrency 1 \
  --variant "t9=--block-size 9 --overlap --env SGLANG_DSPARK_DEBUG_DUMP=all"
"$DSPARK_PY" benchmarks/dspark_budget/calibration_report.py benchmarks/dspark_budget/runs/trace/t9/c1-r0/server-info.json
```

CPU 测试：

```bash
"$DSPARK_PY" -m pytest -q \
  test/registered/spec/dspark/test_dspark_live_cost.py \
  test/registered/spec/dspark/test_dspark_cost_integration.py \
  test/registered/spec/dspark/test_dspark_step_cost.py
```

本次没有跑准确率评测。temperature=0 下基线自身两次运行的输出也只有前 20–35 个 token 相同，无法用逐字比对检查。
