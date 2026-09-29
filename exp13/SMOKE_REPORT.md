# Exp13 重构与 smoke 报告

仅修改 `exp13/`。正式 Stage 1 / Stage 2 full experiment 未启动。

## 修改文件

- `exp13/README.md`
- `exp13/adam.py`
- `exp13/aggregate.py`
- `exp13/collect_trajectory.py`
- `exp13/common.py`
- `exp13/config.yaml`
- `exp13/fit_strategies.py`
- `exp13/full_training.py`
- `exp13/launch_full.sh`
- `exp13/plotting.py`
- `exp13/privacy.py`
- `exp13/pytest.ini`
- `exp13/replay.py`
- `exp13/strategies.py`
- `exp13/tests/test_core.py`
- `exp13/tests/test_readout.py`
- `exp13/worker.py`
- `exp13/workloads.py`
- `exp13/numerical.py`
- `exp13/run_numerical.sh`
- `exp13/run_training.sh`
- `exp13/validate_strategies.py`
- `exp13/tests/test_stages.py`
- `exp13/SMOKE_REPORT.md`

Smoke artifacts 位于 `results_smoke/stage1_numerical/` 和 `results_smoke/stage2_training/`；旧结果保留。

## 验证与耗时

- Stage 1 最终完整 smoke：92.45 秒；fit/cross-eval 21.36 秒，collect 49.02 秒，replay 子进程 21.85 秒（含进程启动）。
- 最后单独 replay 复测内部耗时：13.40 秒；未重新 fit/collect。
- Smoke：64 个本地训练/测试样本，4 logical steps，fit_steps=5，replay_draws=2。正式配置仍为 fit_steps=1000、replay_draws=3。
- `conda run -n curve pytest exp13/tests`：37 passed，47.88 秒；11 条 warnings 来自既有 official BandMF 初始化的除零提示。
- Stage 2 完成结果见报告末尾。

## 轨迹与 replay 性能路径

- 模型 5,501,002 坐标；固定 seed=1203 无放回均匀抽样 262,144 坐标；smoke 保存 shape=[4,262144]。
- 轨迹定义：clipped DP-query gradients evaluated along a clean Adam parameter trajectory。整批平均 loss 的无裁剪梯度推进 Adam，同一更新前参数/批次的 clipped query 只用于存储。
- 已删除 replay 自动 collect、全模型 `gradients.npy` 生成、完整 `absolute_directions.npy` / `denominators.npy` memmap 以及全坐标 exact quantile 路径。
- GPU 返回逐步聚合量、min/max 和固定分位数子样本；不返回完整 chunk 的方向/分母矩阵。
- Quantiles are computed from a deterministic coordinate subsample：8192 坐标，所有机制/抽样轮次共用固定坐标集合。阈值 fractions、min/max 使用全部 262144 replay 坐标。

## 四个 strategy fitting objective

| Strategy | Smoke objective |
|---|---:|
| C_m_raw | 0.01505316608 |
| C_m_bc | 0.4793505669 |
| C_v_raw | 1.716004704e-06 |
| C_v_bc | 0.4765596092 |

raw/BC 使用相同 bandwidth、参与度约束和 column normalization；BC 只改变 streaming utility workload。

## 完整 raw / BC cross-eval

| Workload | C_m_raw | C_m_bc | C_v_raw | C_v_bc |
|---|---:|---:|---:|---:|
| H_m_raw | 0.0150531661 | 0.0161966719 | 0.0151106473 | 0.01613269 |
| H_m_bc | 0.528961658 | 0.479350567 | 0.55563271 | 0.479531169 |
| H_v_raw | 1.72251362e-06 | 1.91318736e-06 | 1.7160047e-06 | 1.9038871e-06 |
| H_v_bc | 0.523584962 | 0.476517826 | 0.549729407 | 0.476559609 |

## Replay raw / BC moment diagnostics

| Method | first_state_mse | second_state_mse | first_hat_mse | second_hat_raw_mse | second_hat_abs_mse | negative_fraction |
|---|---:|---:|---:|---:|---:|---:|
| iid_ime | 0.00031249813 | 1.4128508e-07 | 0.0078936122 | 0.029498609 | 0.029498601 | 0.50041056 |
| bandmf_ime_sep_raw | 0.00022728095 | 9.719307e-08 | 0.0079850691 | 0.03114127 | 0.031141263 | 0.50016069 |
| bandmf_ime_sep_vbc | 0.00022728095 | 1.0781699e-07 | 0.0079850691 | 0.026992943 | 0.026992936 | 0.50030708 |
| bandmf_ime_sep_bc | 0.00024464532 | 1.0781699e-07 | 0.0072381368 | 0.026992943 | 0.026992936 | 0.50030708 |

## Replay direction diagnostics

| Method | adam_direction_rmse | absolute_direction_mean | absolute_direction_median | absolute_direction_p90 | absolute_direction_p99 | absolute_direction_p999 | absolute_direction_max |
|---|---:|---:|---:|---:|---:|---:|---:|
| iid_ime | 1.0912319 | 0.28679141 | 0.17900972 | 0.58695027 | 1.8630489 | 5.553741 | 413.9079 |
| bandmf_ime_sep_raw | 0.98676671 | 0.27369824 | 0.16843397 | 0.56442383 | 1.8080353 | 5.4951604 | 197.54349 |
| bandmf_ime_sep_vbc | 1.0032841 | 0.27416636 | 0.16510732 | 0.57124096 | 1.8112026 | 5.5310915 | 213.87189 |
| bandmf_ime_sep_bc | 0.98790248 | 0.27526955 | 0.17066462 | 0.56558797 | 1.7898415 | 5.4639285 | 191.94937 |

## Replay denominator diagnostics

| Method | denominator_min | denominator_p0001 | denominator_p001 | denominator_p01 | denominator_p1 | denominator_median |
|---|---:|---:|---:|---:|---:|---:|
| iid_ime | 0.00022036253 | 0.0051599996 | 0.012958934 | 0.04406905 | 0.13727614 | 0.32344146 |
| bandmf_ime_sep_raw | 0.00034144334 | 0.0042539181 | 0.012515215 | 0.040121471 | 0.12581576 | 0.2975111 |
| bandmf_ime_sep_vbc | 0.00031739712 | 0.0059509763 | 0.013735726 | 0.040705833 | 0.13017464 | 0.30554564 |
| bandmf_ime_sep_bc | 0.00031739712 | 0.0059509763 | 0.013735726 | 0.040705833 | 0.13017464 | 0.30554564 |

所有方法 denominator < 1e-8 / 1e-7 / 1e-6 / 1e-5 / 1e-4 的 fractions 均为 0（使用全部 replay sample）。
分位数后缀 p0001/p001/p01/p1 对应概率 .0001/.001/.01/.1。
逐步统计见 [per_step_diagnostics.csv](results_smoke/stage1_numerical/replay/per_step_diagnostics.csv)。

## 理论 workload validation

理论值直接计算 sigma² × mean(per_query_error)，使用 raw 与 BC streaming workloads，不依赖 Monte Carlo 估计理论值。abs readout 是非线性后处理，不使用线性 BC 理论预测其 MSE。

| Method | Channel | Raw empirical/theory | BC empirical/theory |
|---|---|---:|---:|
| iid_ime | first | 1.0000469 | 0.9996956 |
| iid_ime | second | 0.9985058 | 0.9986574 |
| bandmf_ime_sep_raw | first | 0.9994125 | 0.9992268 |
| bandmf_ime_sep_raw | second | 0.9987183 | 0.9988805 |
| bandmf_ime_sep_vbc | first | 0.9994125 | 0.9992268 |
| bandmf_ime_sep_vbc | second | 0.9985555 | 0.9987555 |
| bandmf_ime_sep_bc | first | 0.9998176 | 0.9995008 |
| bandmf_ime_sep_bc | second | 0.9985555 | 0.9987555 |

## 配置与实现确认

- `nonprivate_adam`：整批平均 cross entropy 的梯度 + 标准 Optax Adam；不构造 clipped query，不添加噪声，不进行隐私校准。metadata: privacy=null, clipping=false, noise=false, optimizer=clean_adam。
- `iid_adam`：复用仓库 `calibrate_nonamplified_iid`、`make_nonamplified_dpadamw_train_step`，weight_decay=0；epoch epsilon 使用 `epsilon_spent_for_iid_prefix`。
- sampling_amplification=false；IME 保持相同 GDP split、query sensitivities 和 paired latent Gaussian keys。
- 全部 IME 使用 abs(v_hat_raw)，raw recurrence 不受 abs 影响。
- adam_eps 固定 1e-8，没有 epsilon sweep。
- seeds=[0,1,2]；METHODS=7；21 jobs；method-major / seed-minor。
- GPU=(1,2,3) 和 worker physical GPU binding 未修改；每个 worker 正好 7 jobs。
- Stage 2 只读取 Stage 1 strategies；缺失/配置不符直接失败；不 fit、cross-eval、collect 或 replay。任一 worker 失败则 launcher 非零退出。
- 正式 full experiment 未启动。

同 seed 的 schedule/checkpoint SHA256 和初始准确率一致。不同进程的初始 loss 存在约 1e-4 量级浮点差异，rtol=1e-4 校验通过；没有将浮点 loss 当成逐位确定性的检查。

## Stage 2 smoke 最终结果

- `bash exp13/run_training.sh --smoke` 退出码 0，21/21 jobs 完成；GPU 1/2/3 各 7 jobs，每个 job 仅 4 logical steps。
- `runs.json` 21 条、`aggregate.json` 35 条、`paired_differences.json` 25 条；每组 n=3，统计值全部有限。`accuracy.png` 已生成。
- 所有 IME metadata 均为 abs(v_hat_raw)；clean Adam 与 canonical IID 的 metadata/epoch privacy 检查通过。
- 全部 seeds 的 schedule/checkpoint hash、初始准确率匹配；初始 loss 在 rtol=1e-4 内一致。
- 完整检查记录：[smoke_validation.json](results_smoke/stage2_training/smoke_validation.json)。

| Method | Final accuracy mean | Final loss mean |
|---|---:|---:|
| nonprivate_adam | 0.109375 | 4.147389 |
| iid_adam | 0.098958 | 2.723731 |
| bandmf_single_m | 0.135417 | 2.459389 |
| iid_ime | 0.098958 | 2.548836 |
| bandmf_ime_sep_raw | 0.130208 | 2.546445 |
| bandmf_ime_sep_vbc | 0.083333 | 2.719272 |
| bandmf_ime_sep_bc | 0.093750 | 2.714419 |

以上是流程 smoke，不能用于评判正式实验效果。正式 full experiment 未启动。

## 正式启动命令

```bash
bash exp13/run_numerical.sh
bash exp13/run_training.sh
```
