# AdamW 的 BandInvMF workload 对照实验

科学问题：实际优化器始终为 AdamW，哪种线性 surrogate 能设计出更好的相关噪声？
实验采用 3×2 factorial design，比较 SGD / EMA momentum / Adam first moment，
以及 surrogate 是否包含 decoupled weight decay。replay 的 cancellation 不直接推出训练精度。

所有实验实现、测试和产物均在本目录；唯一公共扩展是
`src/dp_muon/optim/linear_workload.py` 的 `momentum_trajectory_workload_matrix`
及其 public export。数据加载、BandInvMF、privacy accounting、训练和 AdamW replay
均复用仓库实现，不复制；具体复用 `exp2.common`、`exp2.collect_trajectory`、
`exp2.full_training._train_one_strategy`、`exp2.run.adamw_perturbations` 以及 `dp_muon`。

## 固定配置与六种定义

读取 `config/cifar10_bandinv_dpadamw_naive.yaml`，从 `data/` 的训练样本数量推导
fixed-cycle contract。当前 50,000 个样本、5 epochs、batch 512 得到
horizon 488、min_sep 97、max_participations 5。正式实验不会硬编码这些数值。
实际 AdamW 六组均为 eta=0.005、beta1=0.9、beta2=0.999、epsilon_adam=1e-8、
weight_decay=0.01；clip=1、privacy epsilon=3、delta=1e-5、bandwidth=4。
预训练 ViT-Tiny checkpoint、microbatch、eval_every 和 fitting 参数均读取上述配置。
相同 seed 使用相同初始化、fixed-cycle schedule 和噪声随机数源。

令 rho=1-eta*lambda，P[t,s]=1[s<=t]，P_rho[t,s]=rho^(t-s)1[s<=t]，
H[t,s]=(1-beta1) beta1^(t-s)1[s<=t]，
H_BC[t,s]=H[t,s]/(1-beta1^t)，数学下标 t 从 1 开始。

| strategy | workload A |
|---|---|
| sgd | P |
| sgd_wd | P_rho |
| momentum | P H |
| momentum_wd | P_rho H |
| adam_momentum | P H_BC |
| adam_momentum_wd | P_rho H_BC |

EMA momentum **没有 bias correction**。Adam-momentum 只建模 first moment。
共享 Adam helper 返回 eta*P_rho*H_BC，实验除以 eta，六种定义均不包含整体系数 eta。
workload 构造局部使用最高矩阵乘法精度，避免 GPU 默认低精度掩盖 WD 差异；不改变训练精度。
六种拟合共享 horizon、bandwidth、participation、reduction 和 max_optimizer_steps。
所有设计都通过现有 general-workload fitter 入口（由共享实现识别 Toeplitz 特例）。

## Part A / B：策略与精确 linear cross-eval

`run.py` 保存 `results/strategies/{name}.npz` 及 JSON metadata，汇总
`strategy_summary.csv/json`，包括完整 C、D=C^-1 系数、sensitivity、objective、
正式 calibration 和 marginal variance，另有 `strategy_noising_coefficients.png`。
`strategy_pairwise_distance.csv/json/png` 定义为 ||D_i-D_j||_F / ||D_i||_F；
这里用完整有限 horizon 矩阵的向量化欧氏范数，因此矩阵不对称。

每个 design i 独立正式校准 sigma_i，定义 B_i=sigma_i D_i、
Sigma_i=B_i B_i^T。matched-marginal IID covariance 为 diag(diag(Sigma_i))，
包括起始边界的时变 marginal；不使用稳态 marginal。
计算 diag(A_j Sigma_i A_j^T) 与
 diag(A_j diag(diag(Sigma_i)) A_j^T)，无需 Monte Carlo。
能量是每个独立参数坐标的期望平方误差，trajectory 对窗口内各行求和，
endpoint 取窗口最后一行。ratio=correlated/IID，gain=1-ratio。

报告 early=1..min_sep（正式为 1–97）、late=min_sep+1..T（正式为 98–488）、full=1..T。
窗口从同一完整轨迹切片，late 不重置状态。
`linear_cross_eval.csv/json` 含所有 36 组、三窗口、两种能量；
`linear_G_full_heatmap.png` 和 `linear_G_endpoint_heatmap.png` 行是 evaluation workload，
列是 design strategy。

## Part C：完整 nonlinear AdamW replay

按 Exp2 收集完整 clean AdamW 训练的 post-clipping/pre-update gradient sequence。
默认分析叶子为 `blocks/0/attention/query/kernel`（192×192），可用 `--parameter-name`
指定另一 rank-2 参数。收集时模型全参数更新；replay 记录和能量限定于这个叶子，
不声称是全模型能量。AdamW 按坐标运算，所以该叶子的 replay 保留完整非线性动态。

冻结 g 后，使用完整 first/second moment、bias correction、epsilon、learning rate、
真实 weight decay，分别 replay g+xi 和 g+xi_iid，与 clean 参数轨迹比较。
不修改 second moment，不做 AdamBC 或 oracle correction。
这里也使用各 strategy 正式校准的 sigma，保留相同 privacy budget 下真实噪声尺度。
六种 strategy 和其 IID controls 共用 latent Gaussian draws；默认 100 个样本、seed=0。
逐样本处理以控制内存。先对平方位移求期望，再取 correlated/IID 比值。
`adam_replay.csv/json` 输出各窗口 trajectory / endpoint 能量和 ratio/gain；
`adam_replay_summary.png` 画 full trajectory ratio。
`adam_replay_paired.csv/json` 对所有 strategy pair 给出 full trajectory ratio 差异和
2,000 次 paired bootstrap 的 95% CI（共同重采样 latent 样本）。

matched-marginal IID **仅是 mechanism control**，用于隔离时间相关性；
它没有单独的同预算 privacy claim，也不作为 Part D 的 utility baseline。
clean trajectory 和 replay 是内部诊断，不是另一个可发布的 DP transcript。

## Part D：same-(epsilon,delta) closed-loop utility

六组真实 CIFAR-10 训练始终为同一 AdamW，仅替换相关噪声 strategy。
每组使用自身 sensitivity_squared 调用正式 calibration，不能共用 sigma。
seeds 固定为 0..9。每个 worker 只写自己的
`results/runs/{strategy}/seed{seed}/`，其中含 metrics CSV、checkpoint 和 `result.json`。
metrics 含 train loss、test loss、test accuracy；result 还含 final/best utility、
calibrated iid noise std、sensitivity 和 mean marginal correlated-noise variance。

固定 job 顺序为 strategy-major、seed-minor，worker i 处理 index % num_workers == i。
四个独立进程各绑定一张 GPU，每个分配 15 个 job；不依赖 JAX 多设备训练。
脚本先完成 A/B 和一次 clean collection/replay，再启动四个 worker，最后等待全部退出，
任一 worker 失败则 launcher 返回非零。运行期间不要重新生成策略文件。

在仓库根目录直接启动全部正式实验（耗时，开发时不会自动执行）：

```bash
bash exp_adam_workload/launch_full.sh
```

该脚本内部使用 `CUDA_VISIBLE_DEVICES=<gpu> conda run --no-capture-output -n curve ...`。
日志位于 `exp_adam_workload/results/worker_logs/worker{0,1,2,3}.log`。
如需分步运行：

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n curve python exp_adam_workload/run.py
# 也可加 --prepare-only，仅完成 A/B；--samples 可调 replay 样本数。
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n curve python exp_adam_workload/run_full.py --worker-index 0 --num-workers 4
# 其余 GPU 分别运行 worker-index 1、2、3。
```

全部 60 个训练完成后执行：

```bash
conda run -n curve python exp_adam_workload/aggregate.py
```

aggregate 要求全部预期 result 存在，缺失直接报错，不静默汇总不完整结果。
输出 `utility_runs.csv/json`、`utility_summary.csv/json`，跨 seed mean、sample std、
SE 和 Student-t 95% CI；`utility_paired_differences.csv/json` 输出所有 15 对 strategy
逐 seed 差值的相同统计（方向为 a-b）。CI 是逐项未作多重比较校正的区间。
`utility_{metric}.png` 绘制 mean 与 95% CI。主要比较应使用 paired differences。

## 测试与 smoke

```bash
conda run -n curve pytest exp_adam_workload/tests
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n curve python exp_adam_workload/run.py --smoke
```

smoke 将数据规模换成合成 20 examples、batch 4、2 epochs，fitter 仅 5 steps、
replay 3 samples；仍覆盖六种策略、独立 calibration、精确 cross-eval、完整 replay、
以及共享正式训练 primitive 的合成数据 closed-loop 更新。
只写 `results_smoke/`，不是正式 CIFAR-10 结果，不能推断 workload 排名。

当前实现验证状态：实验测试 18 项通过；公共 `tests/test_linear_workload.py` 的
14 项回归测试通过。GPU smoke（包括合成 closed-loop）通过，图表已生成并检查。
`results/verified_config_and_contract.json` 记录实际读取 `data/` 推导出的正式配置。
正式 A/B 拟合作为额外验证启动后因耗时停止，没有形成正式 strategy 或 cross-eval 结果；
正式 clean replay 和 60 次 CIFAR-10 训练均未执行。现有 `results_smoke/` 数值不能用于回答正式 utility 排名。
