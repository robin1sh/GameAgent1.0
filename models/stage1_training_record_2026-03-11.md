# 阶段1训练记录（2026-03-11）

## 目标
- 训练两个游戏专用 imitation CNN，并导出 backbone 供阶段2 PPO 初始化。

## 数据来源
- mario: `data_imitation_unified/mario`（约 4537 样本）
- jumper: `data_imitation_unified/jumper`（约 3160 样本）

## 训练命令
- `python3 train/train_cnn_imitation_unified.py --source mario --epochs 30 --batch-size 128`
- `python3 train/train_cnn_imitation_unified.py --source jumper --epochs 30 --batch-size 128`

## 关键超参数（脚本默认）
- `lr=1e-3`
- `optimizer=AdamW(weight_decay=1e-4)`
- `scheduler=StepLR(step_size=5, gamma=0.5)`
- `classes=15`

## 结果摘要
- mario: `val_acc` 约 `0.94~0.96`（末期稳定）
- jumper: `val_acc` 约 `0.71~0.72`（末期平台）

## 产物清单
- `models/mario_backbone.pt`
- `models/mario_cnn.pt`
- `models/jumper_backbone.pt`
- `models/jumper_cnn.pt`

## 阶段2建议
- Mario PPO 使用：`--pretrain-path models/mario_backbone.pt`
- Jumper PPO 使用：`--pretrain-path models/jumper_backbone.pt`
