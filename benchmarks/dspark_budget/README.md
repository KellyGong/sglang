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
