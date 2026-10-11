Dataset v1 Design
1. Dataset Objective（数据集目标）

Dataset v1 用于训练第一个 State-only Baseline（仅状态基线模型）。

模型输入：

State（14维机器人状态）

模型输出：

Action（14维机器人动作）

2. Sample Schema（样本结构）

每个 Frame（帧）保留以下字段：

episode_index：Episode（任务片段）编号
frame_index：Frame（帧）编号
timestamp：时间戳
state：14维机器人状态
action：14维机器人动作

其中：

State 作为模型输入 X
Action 作为模型学习目标 y
Episode / Frame / Timestamp 用于数据划分、QC、时序追踪和 Failure Analysis（失败分析）