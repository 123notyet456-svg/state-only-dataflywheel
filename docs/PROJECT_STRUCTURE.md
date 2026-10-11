# 项目结构与公开范围

本仓库是 State-only（仅状态输入）实验的精选展示版。QC（质量检查）用于检查轨迹、样本结构和切分质量；数据集与模型文件由运行脚本在本地生成。

## 目录

```text
.
├── README.md
├── .gitignore
├── docs/
│   ├── PROJECT_STRUCTURE.md
│   ├── EXPERIMENT_RESULTS.md
│   └── EVIDENCE_INDEX.md
├── dataset_v1/       # 16,000 / 2,000 / 2,000 帧
├── dataset_v2/       # 全局动态过采样数据
├── dataset_v3/       # Dim12 定向过采样数据
├── models/           # 五个阶段最佳权重
├── scripts/
│   ├── build_dataset*.py
│   ├── check_*qc.py / dataset_qc.py
│   ├── train_*state_policy*.py
│   ├── evaluate_*state_policy*.py
│   └── analyze_*.py / compare_*.py / diagnose_*.py
└── reports/
    └── 质量检查、训练历史、错误分析和最终对照的汇总文件
```

## 脚本分组

| 环节 | 代表脚本 | 用途 |
|---|---|---|
| v1 数据构建与质量检查 | `build_dataset.py`、`dataset_qc.py`、`check_episode_qc.py`、`check_robot_trajectory_qc.py` | 轨迹读取、分集、形状与连续性检查 |
| v2 / v3 数据干预 | `build_dataset_v2.py`、`build_dataset_v3.py`、`check_dataset_v2_qc.py`、`check_dataset_v3_qc.py` | 全局或 Dim12 定向动态帧过采样及干预后检查 |
| 基线与数据版本训练 | `train_state_policy.py`、`train_state_policy_v2.py`、`train_state_policy_v3.py`、`train_state_policy_matched.py` | 训练仅状态模型及匹配对照模型 |
| 时间状态模型 | `train_temporal_state_policy.py`、`evaluate_temporal_state_policy.py` | 使用当前状态和状态变化量训练、评测模型 |
| 失败与动态诊断 | `analyze_baseline_failures.py`、`analyze_data_gap.py`、`analyze_per_joint_dynamics.py`、`diagnose_temporal_context.py` | 从整体误差定位到关节、轨迹、帧和时间特征 |
| 版本比较与受控实验 | `compare_v1_v2_failures.py`、`compare_v1_v2_v3_failures.py`、`compare_matched_state_vs_temporal.py` | 评估数据策略取舍和同帧模型差异 |

## 公开文件与排除项

`reports/` 收录聚合质量结果、实验汇总和训练历史。仓库额外保留 v1/v2/v3 的数据拆分文件及五个阶段最佳权重，便于直接检查数据和运行评测。`.gitignore` 排除原始视频、其他本地数据、预测数组、缓存、环境目录和本地配置。

已纳入的权重为 `state_policy_best.pt`、`state_policy_v2_best.pt`、`state_policy_v3_best.pt`、`state_policy_matched_best.pt` 和 `temporal_state_policy_best.pt`。数据集为每个版本的 `train.npz`、`val.npz`、`test.npz`。原始视频及预测数组仍不公开；模型权重与上游数据集的授权关系需要与数据许可分别理解。

`check_episode_qc.py` 的项目根目录已改为根据脚本位置计算，便于迁移；运行它仍需要外部提供视频和元数据。其余脚本来自本地实验脚本目录，路径按项目根目录相对解析。公开仓库是代码与证据摘要，不是数据分发包。

## 可复现性边界

脚本使用 Python、PyTorch、NumPy、Pandas 等依赖；视频检查还需要 OpenCV，LeRobot 数据检查需要相应的数据集客户端。原始数据和模型文件未随仓库提供，因此复跑前需准备输入数据并安装兼容依赖。此整理没有验证依赖版本或重新运行模型。
