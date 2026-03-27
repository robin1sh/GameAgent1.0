# Stable-Baselines3 PPO 结构说明与自定义 CNN 集成指南

> 面向毕业设计：马里奥强化学习项目中的 PPO 算法架构及自研 CNN 特征提取器集成方案

---

## 目录

1. [PPO 算法概述](#1-ppo-算法概述)
2. [Stable-Baselines3 整体架构](#2-stable-baselines3-整体架构)
3. [PPO 策略网络结构](#3-ppo-策略网络结构)
4. [CnnPolicy 与 NatureCNN](#4-cnnpolicy-与-naturecnn)
5. [自定义 CNN 集成方案](#5-自定义-cnn-集成方案)
6. [本项目中的训练流程](#6-本项目中的训练流程)
7. [附录：关键代码参考](#7-附录关键代码参考)

---

## 1. PPO 算法概述

### 1.1 算法背景

**Proximal Policy Optimization (PPO)** 由 OpenAI 于 2017 年提出，是当前最流行的深度强化学习算法之一。其核心思想是**限制策略更新幅度**，避免训练不稳定。

- **论文**：Schulman et al., *Proximal Policy Optimization Algorithms* (arXiv:1707.06347)
- **特点**：实现简单、样本效率高、训练稳定、适合连续与离散动作空间

### 1.2 核心公式

PPO 使用 **Clipped Surrogate Objective**：

- **策略损失**：\( L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right] \)
- **比率**：\( r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \)
- **总损失**：\( L = L^{CLIP} - c_1 L^{VF} + c_2 H[\pi_\theta] \)

其中 \( L^{VF} \) 为价值函数损失，\( H \) 为熵正则项。

### 1.3 PPO 在 SB3 中的主要超参数（本仓库当前默认）

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `n_steps` | 每次更新前收集的步数 | 2048 |
| `batch_size` | 小批量大小 | 8192（通用入口默认） |
| `n_epochs` | 每轮更新迭代次数 | 5（通用入口默认） |
| `gamma` | 折扣因子 | 0.95（通用入口默认） |
| `gae_lambda` | GAE 参数 | 0.95 |
| `clip_range` | 策略裁剪范围 | 0.2 |
| `ent_coef` | 熵系数 | 0.1 |

> 说明：`train_ppo_mario.py` / `train_ppo_coinrun.py` 会在通用入口参数基础上覆盖默认值（例如 batch_size、n_epochs）。

---

## 2. Stable-Baselines3 整体架构

### 2.1 算法继承关系

```
BaseAlgorithm
    └── OnPolicyAlgorithm (A2C, PPO)
            └── PPO
```

- **PPO** 继承自 **OnPolicyAlgorithm**，采用 on-policy 方式收集数据并更新。
- 每次 `learn()` 调用会执行：收集 rollout → 计算 GAE → 多轮 mini-batch 更新。

### 2.2 训练流程

```
1. collect_rollouts()  → 收集 n_steps × n_envs 步
2. rollout_buffer.get() → 按 batch_size 取 mini-batch
3. train()             → 计算策略损失、价值损失、熵损失
4. 反向传播 + 梯度裁剪
5. 重复 2–4 共 n_epochs 轮
```

### 2.3 数据流

```
环境观测 obs → 预处理(归一化/转置) → 特征提取器(CNN) → 特征向量
                                                    ↓
                                    ┌───────────────┴───────────────┐
                                    ↓                               ↓
                              策略网络(pi)                        价值网络(vf)
                                    ↓                               ↓
                              动作分布 π(a|s)                     V(s)
```

---

## 3. PPO 策略网络结构

### 3.1 策略类型

| 策略名 | 对应类 | 适用场景 |
|---------|--------|----------|
| MlpPolicy | ActorCriticPolicy | 向量观测（如 CartPole） |
| CnnPolicy | ActorCriticCnnPolicy | 图像观测（如 Atari、马里奥） |
| MultiInputPolicy | MultiInputActorCriticPolicy | 字典观测（图像+向量） |

### 3.2 ActorCriticPolicy 组成

```
ActorCriticPolicy
├── features_extractor   # 特征提取网络（CNN / Flatten）
├── mlp_extractor        # 共享的 MLP 提取器
│   ├── policy_net       # 策略网络（pi）
│   └── value_net        # 价值网络（vf）
├── action_net           # 动作输出层（离散：Softmax / 连续：均值+方差）
└── value_net            # 价值输出层
```

### 3.3 net_arch 参数

```python
# 共享结构：actor 和 critic 使用相同隐藏层
net_arch=[64, 64]

# 分离结构：actor 和 critic 使用不同结构
net_arch=dict(pi=[32, 32], vf=[64, 64])
```

---

## 4. CnnPolicy 与 NatureCNN

### 4.1 CnnPolicy 默认特征提取器

**CnnPolicy** 默认使用 **NatureCNN**（来自 DQN Nature 论文）：

- 输入：`(C, H, W)` 图像，通道优先（CxHxW）
- 结构：3 层卷积（Conv 8×8 stride 4 → Conv 4×4 stride 2 → Conv 3×3 stride 1）
- 输出：512 维特征向量

### 4.2 NatureCNN 结构

```python
# stable_baselines3/common/torch_layers.py
nn.Conv2d(n_input_channels, 32, 8, stride=4)  # 84×84 → 20×20
nn.Conv2d(32, 64, 4, stride=2)                 # 20×20 → 9×9
nn.Conv2d(64, 64, 3, stride=1)                 # 9×9 → 7×7
nn.Flatten() → nn.Linear(n_flatten, features_dim)
```

### 4.3 本项目中的观测空间

- **环境**：Super Mario Bros v2
- **预处理（当前实现）**：灰度 + 84×84 缩放 + `VecFrameStack(4)` 4 帧堆叠
- **观测形状（当前实现）**：`(4, 84, 84)`（通道优先）或 `(84, 84, 4)`（通道在后）

---

## 5. 自定义 CNN 集成方案

### 5.1 思路概述

将你在 **模仿学习** 中训练的 CNN（当前以 `train_model/train_cnn_imitation_unified.py` 为主）作为 PPO 的**特征提取器**，替代默认的 NatureCNN。

- **模仿学习 CNN（当前实现）**：输入 4×84×84（4 帧堆叠）→ 输出 15 类 logits
- **PPO 特征提取器（当前实现）**：输入 4×84×84（或 84×84×4）→ 输出 `features_dim` 维特征向量

需要做两件事：

1. 继承 `BaseFeaturesExtractor`，实现与 CNN 主干兼容的卷积层；
2. 去掉最后的分类头，改为输出固定维度的特征向量。

### 5.2 当前代码中的 CNN 结构（以仓库实现为准）

```python
# CNN_TEMPLATE/custom_cnn.py / train_model/train_cnn_imitation_unified.py
Conv2d(4,32,8,4) → ReLU
Conv2d(32,64,4,2) → ReLU
Conv2d(64,64,3,1) → ReLU
Flatten → Linear(...,512) → ReLU
分类头（仅模仿学习）：Linear(512,15)
```

### 5.3 自定义特征提取器示例

```python
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MarioCNNFeaturesExtractor(BaseFeaturesExtractor):
    """
    基于模仿学习 CNN 的 PPO 特征提取器。
    适配 VecFrameStack(4) 的 4 通道输入。
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # 4（帧堆叠）

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2),
            )

        self.cnn = nn.Sequential(
            conv_block(n_input_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        self._features_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
```

### 5.4 训练时使用自定义特征提取器

```python
from stable_baselines3 import PPO
from your_module import MarioCNNFeaturesExtractor

policy_kwargs = dict(
    features_extractor_class=MarioCNNFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=256),
)

model = PPO(
    policy="CnnPolicy",
    env=vec_env,
    policy_kwargs=policy_kwargs,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=8192,
    n_epochs=5,
    ent_coef=0.1,
    gamma=0.95,
    verbose=1,
    tensorboard_log="logs",
)
model.learn(total_timesteps=1e7, callback=eval_callback)
```

### 5.5 预训练权重迁移（可选）

若希望用模仿学习预训练权重初始化 PPO 特征提取器（当前实现）：

1. `train_model/train_cnn_imitation_unified.py` 已直接按 4 通道输入训练 backbone。
2. 保存 `backbone_state_dict`（不含分类头），并在 `train_model/train_ppo_model.py` 中通过 `--pretrain-path` 加载，`strict=False`。

---

## 6. 本项目中的训练流程

### 6.1 环境构建（当前实现）

```python
# envs/unified_env.py + envs/mario_env.py + envs/coinrun_env.py
# 当前主线以 Mario/CoinRun 为主，统一预处理为 84x84 灰度，并在训练时统一做 VecFrameStack(4)
```

### 6.2 向量化训练（当前实现）

```python
# train_model/train_ppo_model.py
# 支持 --env mario/jumper/coinrun/both、--pretrain-path、--resume
# 训练/评估都使用 VecMonitor + VecTransposeImage 保持观测布局一致
# policy_kwargs 注入 CustomCNN(features_dim=512)
# CallbackList(EvalCallback + MetricsEvalCallback) 记录最优模型与扩展指标
```

### 6.3 推理

```python
# test/test_model.py
model = PPO.load("best_model/best_model.zip", env=env)
action, _ = model.predict(obs, deterministic=True)
```

### 6.4 Jumper 的 exploration 固定单关卡用法（当前实现）

- `distribution_mode=exploration` 时，Procgen 会内部强制固定为单关卡种子。
- 因此命令应使用 `--no-fixed-level --distribution-mode exploration`，不要再手动传 `--start-level`。

```bash
# 记录 imitation 数据（固定 exploration 单关卡）
python playing/record_jumper_unified.py --no-fixed-level --distribution-mode exploration
```

```bash
# PPO 训练（Jumper 对照实验，同一 exploration 单关卡）
python train_model/train_ppo_model.py --env jumper --no-fixed-level --distribution-mode exploration --total-timesteps 5000000 --exp-id jumper_exploration_v1
```

---

### 6.5 CoinRun 作为当前主线第二环境

当前毕业设计主线建议使用 `coinrun` 作为第二环境：它比 `jumper` 更简单，也更接近 Mario 的横版向右平台跳跃结构。

当前仓库已提供独立入口，接口与 `jumper` 基本一致：

```bash
# 记录 imitation 数据（CoinRun）
python playing/record_coinrun_unified.py --fixed-level --start-level 0 --distribution-mode easy
```

```bash
# PPO 训练（CoinRun）
python train_model/train_ppo_coinrun.py --n-envs 10 --total-timesteps 5000000 --exp-id coinrun_baseline_v1
```

实现原则：

- 继续复用 Procgen 的 15 动作空间，不额外重写统一动作映射；
- 继续复用 `84x84` 灰度、跳帧与 4 帧堆叠预处理；
- `jumper` 不删除，作为兼容入口和后续对照实验环境保留。

---

## 7. 附录：关键代码参考

### 7.1 PPO 策略别名

```python
# stable_baselines3/ppo/ppo.py
policy_aliases = {
    "MlpPolicy": ActorCriticPolicy,
    "CnnPolicy": ActorCriticCnnPolicy,
    "MultiInputPolicy": MultiInputActorCriticPolicy,
}
```

### 7.2 ActorCriticCnnPolicy 默认参数

```python
# 默认使用 NatureCNN
features_extractor_class=NatureCNN
features_extractor_kwargs=dict(features_dim=512)
share_features_extractor=True  # actor 与 critic 共享 CNN
```

### 7.3 自定义特征提取器要点

- 继承 `BaseFeaturesExtractor`
- 必须实现 `__init__(observation_space, features_dim)` 和 `forward(observations)`
- `forward` 输出形状为 `(batch_size, features_dim)`
- 通过 `policy_kwargs` 传入 `features_extractor_class` 和 `features_extractor_kwargs`

---

## 参考文献

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
2. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529-533.
3. Stable-Baselines3 文档: https://stable-baselines3.readthedocs.io/

---

*文档生成于 MarioRL 项目，用于毕业设计参考。*
