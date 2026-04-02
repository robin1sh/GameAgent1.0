# 自定义 CNN 与 NatureCNN 区别总结（MarioRL）

## 1. 一句话结论

本项目的 `CustomCNN` 在**核心卷积结构上与 NatureCNN 基本一致**，主要增强点在于**工程适配能力**（输入布局兼容与自动展平维度推断），用于提升多环境训练链路中的稳定性与可复现性。

---

## 2. NatureCNN 简述（SB3 默认）

在 Stable-Baselines3 的 `CnnPolicy` 中，默认特征提取器是 NatureCNN，典型结构如下：

1. `Conv2d(C, 32, kernel=8, stride=4)` + ReLU  
2. `Conv2d(32, 64, kernel=4, stride=2)` + ReLU  
3. `Conv2d(64, 64, kernel=3, stride=1)` + ReLU  
4. `Flatten`  
5. `Linear(n_flatten, 512)` + ReLU

作用：将图像观测编码为 512 维特征向量，供 PPO 的 actor/critic 头使用。

---

## 3. 本项目 `CustomCNN` 实现要点

代码位置：`CNN_TEMPLATE/custom_cnn.py`  
接入位置：`train_model/train_ppo_model.py`、`train_model/train_unified.py`

### 3.1 输入格式兼容

`CustomCNN` 自动识别观测布局：

- `channels_first`：`(C, H, W)`，例如 `(4, 84, 84)`
- `channels_last`：`(H, W, C)`，例如 `(84, 84, 4)`

若为 `channels_last`，在前向中会执行 `permute` 转成 `NCHW` 再卷积。

### 3.2 卷积主干

卷积配置与 NatureCNN 等价：

- `Conv(8x8, stride=4, out=32)`
- `Conv(4x4, stride=2, out=64)`
- `Conv(3x3, stride=1, out=64)`
- `Flatten -> Linear(..., features_dim=512) -> ReLU`

### 3.3 尺寸变化（84x84 输入）

以 `(4,84,84)` 为例：

- Conv1 后：`32 x 20 x 20`
- Conv2 后：`64 x 9 x 9`
- Conv3 后：`64 x 7 x 7`
- Flatten：`64*7*7=3136`
- FC 输出：`512`

### 3.4 自动展平维度推断

通过 `observation_space.sample()` 前向一次自动得到 `n_flatten`，避免手写硬编码尺寸，降低后续改输入形状时的出错风险。

---

## 4. 与 NatureCNN 的对比

## 4.1 相同点（建模主干）

- 三层卷积配置一致（8/4/3 卷积核，4/2/1 步长）
- 激活函数一致（ReLU）
- 最终输出固定维度特征向量（默认 512）
- 都作为 PPO 的视觉特征提取器，不直接输出动作

## 4.2 不同点（工程鲁棒性）
CustomCNN 兼容 HWC/CHW，就是指它能处理两种图像张量排布：

CHW：(通道, 高, 宽)，例如 (4, 84, 84)（PyTorch卷积最喜欢这个）
HWC：(高, 宽, 通道)，例如 (84, 84, 4)（很多环境/图像库常见这个）
你的 CustomCNN 会先判断输入形状：
如果是 HWC，就自动转成 CHW 再喂给 Conv2d。

自动 `n_flatten`不用手算卷积尺寸，减少数学/实现负担
改输入分辨率（比如 96x96）时，不用同步改全连接输入维度
减少常见报错：mat1 and mat2 shapes cannot be multiplied
- `CustomCNN` 显式兼容 `HWC/CHW` 双布局
- `CustomCNN` 显式实现自动 `n_flatten` 推断逻辑
- 更适配本项目多入口训练脚本与多环境链路（single -> fuse -> mixed -> resume）

---

## 5. 为什么本项目采用这种“轻改造”

本项目核心挑战并非“把 CNN 做得更深”，而是保证训练链路一致性：

- 多脚本入口（单环境、融合、混合、续训）
- 多环境并行（Mario/CoinRun）
- 观测包装可能变化（转置、帧堆叠、监控包装）

因此在保留 NatureCNN 经典结构的前提下，优先增强输入兼容性与稳定性，是更符合毕业设计“可解释、可复现”的工程选择。

---

## 6. 可直接用于论文/答辩的描述

本文采用基于 NatureCNN 的自定义特征提取器。该提取器保持经典三层卷积结构与 512 维特征输出不变，同时增加对 `channels_first/channels_last` 双输入布局的兼容与自动展平维度推断机制。该改造重点提升了多环境联合训练流程中的工程鲁棒性与复现稳定性，而非通过增加网络深度追求复杂度。

---

## 7. 小结

`CustomCNN` 可以视为 **NatureCNN 的工程增强版**：

- 主干不激进，利于稳定训练；
- 适配更健壮，利于多阶段实验复现；
- 叙述清晰，适合毕业设计答辩表达。
