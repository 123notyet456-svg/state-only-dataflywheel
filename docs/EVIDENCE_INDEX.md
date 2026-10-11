# 实验证据索引

以下证据包括本地历史实验生成的汇总、v1/v2/v3 状态 / 动作数组及五个阶段模型权重；不包含原始视频或逐帧预测明细。

术语：QC（质量检查）；Baseline（基线模型）；Failure Analysis（失败分析）；Oversampling（过采样）；KNN（K 近邻）；Matched-Control Experiment（匹配控制实验）。

| 阶段 | 代码证据 | 报告证据 | 关注点 |
|---|---|---|---|
| Dataset v1 构建与质量检查 | `scripts/build_dataset.py`、`scripts/dataset_qc.py`、`scripts/check_robot_trajectory_qc.py` | `reports/dataset_v1_design.md`、`reports/dataset_v1_summary.csv`、`reports/dataset_train_qc.csv` | 16,000 / 2,000 / 2,000 帧；14 维状态 / 动作；拆分、形状、缺失值和连续性检查 |
| 初始基线 | `scripts/train_state_policy.py`、`scripts/evaluate_state_policy.py` | `reports/baseline_training_sanity.md`、`reports/baseline_test_metrics.csv`、`reports/baseline_training_history.csv` | 2,000 帧测试；总体 MAE 0.00907244、MSE 0.000204616 |
| 失败分析与数据缺口 | `scripts/analyze_baseline_failures.py`、`scripts/analyze_data_gap.py`、`scripts/analyze_per_joint_dynamics.py` | `reports/baseline_failure_summary.csv`、`reports/baseline_per_episode_metrics.csv`、`reports/baseline_per_joint_metrics.csv`、`reports/data_gap_summary.csv`、`reports/per_joint_dynamics_error_analysis.csv` | Episode 49 / Frame 294；Dim12；动态相关性及高低动态误差比 |
| v2 全局动态过采样 | `scripts/build_dataset_v2.py`、`scripts/check_dataset_v2_qc.py`、`scripts/train_state_policy_v2.py`、`scripts/evaluate_state_policy_v2.py`、`scripts/compare_v1_v2_failures.py` | `reports/dataset_v2_build_summary.csv`、`reports/dataset_v2_qc_summary.csv`、`reports/v1_v2_failure_comparison.csv` | 采样占比约 9.98% → 24.95%；整体高动态 MAE 相对 v1 下降 3.76% |
| v3 Dim12 定向过采样 | `scripts/build_dataset_v3.py`、`scripts/check_dataset_v3_qc.py`、`scripts/train_state_policy_v3.py`、`scripts/evaluate_state_policy_v3.py`、`scripts/compare_v1_v2_v3_failures.py` | `reports/dataset_v3_build_summary.csv`、`reports/dataset_v3_qc_summary.csv`、`reports/v1_v2_v3_failure_comparison.csv` | Dim12 高动态误差下降 27.05%，普通动态误差上升 87.22%；整体高动态 MAE 相对 v1 上升 3.59% |
| 可检查数据与评测权重 | `scripts/evaluate_state_policy.py`、`scripts/evaluate_state_policy_v2.py`、`scripts/evaluate_state_policy_v3.py`、`scripts/evaluate_temporal_state_policy.py` | `dataset_v1/`、`dataset_v2/`、`dataset_v3/`；`models/state_policy_best.pt`、`models/state_policy_v2_best.pt`、`models/state_policy_v3_best.pt`、`models/state_policy_matched_best.pt`、`models/temporal_state_policy_best.pt` | 每个数据版本均含训练 / 验证 / 测试数组；最终匹配对照权重可以直接查看与复用 |
| 时间上下文诊断 | `scripts/diagnose_temporal_context.py` | `reports/temporal_context_diagnostic.csv` | KNN 特征诊断：总体 MAE -7.67%，Dim12 高动态 MAE -17.82% |
| 时间状态训练与匹配对照 | `scripts/train_temporal_state_policy.py`、`scripts/train_state_policy_matched.py`、`scripts/compare_matched_state_vs_temporal.py` | `reports/temporal_state_training_history.csv`、`reports/state_policy_matched_training_history.csv`、`reports/matched_state_vs_temporal_final_comparison.csv` | 同 1,995 测试帧；总体 MAE -27.08%；目标区域 MAE -57.84%；区域 Dim12 MAE -69.24% |

## 未纳入的文件

- 原始视频、缓存、虚拟环境和逐帧预测文件：未纳入展示仓库。
- 虚拟环境、缓存、未筛选的视觉 / LIBERO 实验：与本次 State-only 数据飞轮展示范围无关。
