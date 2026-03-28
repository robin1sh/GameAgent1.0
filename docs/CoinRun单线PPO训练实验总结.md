# CoinRun 单线 PPO 训练实验总结

## 1. 文档目的

本文档总结本轮对话中 CoinRun 单线 PPO 训练的主要实验过程、参数变化、曲线现象与阶段性结论，用于后续论文实验部分撰写。

需要说明的是：

- 本文结论主要依据本轮对话中的 TensorBoard 曲线截图、训练命令与人工分析过程整理；
- 个别实验阶段未直接回读原始事件文件，因此文中更适合表述为“实验现象总结”与“模型选型依据”；
- 对未展示完整曲线的模型，本文只说明其在实验主线中的定位与选择理由，不伪造具体数值。

---

## 2. 单线 CoinRun 训练背景

CoinRun 单任务训练入口为 `train_model/train_ppo_coinrun.py`，其默认会复用 `train_model/train_ppo_model.py` 的训练逻辑，并固定 `--env coinrun`。

当前入口脚本中的默认 CoinRun 训练口径为：

```python
--save-path ./best_model/coinrun/
--log-path ./logs/coinrun/
--callback-log-path ./callback_logs/coinrun/
--normalize-reward
--clip-reward 10.0
--use-aligned-reward
--learning-rate 3e-4
--n-steps 2048
--batch-size 2048
--n-epochs 8
--ent-coef 0.1
--gamma 0.95
```

其中：

- `--use-aligned-reward` 表示 CoinRun 奖励语义向统一训练口径靠拢；
- `--normalize-reward` 与 `--clip-reward 10.0` 用于提升训练数值稳定性；
- 后续所有实验都在这个基础上做命令行覆盖。

---

## 3. 训练过程变化总结

### 3.1 第一阶段：`coinrun_baseline_ppo_random_easy_10m`

这是本轮对话最早分析的一组 CoinRun 主实验。根据截图可观察到：

- `eval/mean_reward` 先明显上升，约在中前期达到较高水平后，后期出现回落；
- `eval/mean_ep_length` 整体抬升，但波动较大；
- 训练并非“完全不会”，因为前期 reward 的确上去了；
- 但后期没有稳定保住最佳策略，表现出一定退化。

这一阶段的核心判断是：

- 训练不是完全失败；
- 但从收敛质量上看，属于“前期学到有效行为，后期性能回退”的不稳定 PPO 结果；
- 主要问题不是不会学，而是后期收敛不稳。

针对这一现象，当时给出的原因分析主要包括：

- 学习率偏大且使用常数学习率，后期容易破坏已有好策略；
- `ent_coef=0.1` 偏高，后期仍保留较强探索；
- `gamma=0.95` 偏低，对长期回报重视不足；
- 评估与训练虽然分离，但训练目标本身仍存在后期优化震荡。

因此，这一阶段更适合被定位为：

**CoinRun 随机关卡 easy 难度下的首个可用基线，但稳定收敛性不足。**

---

### 3.2 第二阶段：从 `coinrun_baseline_ppo_random_easy_10m` 继续训练，得到 `coinrun_baseline_ppo_random_easy_10m_v2`

为了降低后期退化风险，后续不是从头训练，而是基于已有 `best_model.zip` 继续训练，并将训练口径改为：

```bash
python train_model/train_ppo_coinrun.py \
  --resume "/ai/MarioRL/best_model/coinrun/coinrun_baseline_ppo_random_easy_10m/best_model.zip" \
  --n-envs 10 \
  --total-timesteps 10000000 \
  --exp-id coinrun_baseline_ppo_random_easy_10m_v2 \
  --no-fixed-level \
  --distribution-mode easy \
  --learning-rate 1e-4 \
  --lr-schedule linear \
  --final-learning-rate 1e-5 \
  --ent-coef 0.02 \
  --gamma 0.97 \
  --n-steps 2048 \
  --batch-size 2048 \
  --n-epochs 8
```

这一阶段的关键调整思路是：

- 从固定关卡转为 `easy` 随机关卡，提升泛化意义；
- 学习率降到 `1e-4`，并加入线性衰减，减少后期抖动；
- `ent_coef` 从 `0.1` 降到 `0.02`，降低不必要探索；
- `gamma` 从 `0.95` 提高到 `0.97`，更重视长期回报。

从后续截图看，这组实验比第一阶段明显更健康：

- `eval/mean_reward` 持续上升，并在后期稳定在较高区间；
- `rollout/complete_rate` 上升到约 `0.9` 左右；
- `rollout/mean_reward` 同步持续提升；
- `rollout/mean_episode_length` 下降后趋于稳定，说明策略完成任务更高效；
- 没有再出现前一组那种明显的后期回落。

这一阶段的主要结论是：

- CoinRun 在 `easy` 随机关卡上的单线 PPO 训练成功；
- 模型不再只是“学到一点东西”，而是已经形成较稳定的有效策略；
- 这说明“降低学习率 + 线性衰减 + 降低熵系数 + 提升 gamma”的稳定化方向是有效的。

因此，`coinrun_baseline_ppo_random_easy_10m_v2` 可以视为：

**本轮 CoinRun 单线训练中，第一个真正达到稳定可用水准的随机 easy 基线。**

---

### 3.3 第三阶段：继续榨性能，尝试 `coinrun_baseline_ppo_random_easy_v3`

在 `v2` 表现已经较好的情况下，后续继续基于：

`best_model/coinrun/coinrun_baseline_ppo_random_easy_v2/ppo_coinrun_final.zip`

进行额外续训，并尝试提高探索强度。命令口径被调整为：

```bash
python train_model/train_ppo_coinrun.py \
  --resume "best_model/coinrun/coinrun_baseline_ppo_random_easy_v2/ppo_coinrun_final.zip" \
  --n-envs 10 \
  --total-timesteps 7000000 \
  --exp-id coinrun_baseline_ppo_random_easy_v3 \
  --no-fixed-level \
  --distribution-mode easy \
  --learning-rate 1e-4 \
  --lr-schedule linear \
  --final-learning-rate 1e-5 \
  --ent-coef 0.08 \
  --gamma 0.97 \
  --n-steps 2048 \
  --batch-size 2048 \
  --n-epochs 8
```

这里的变化重点不是基础框架，而是：

- 保持 `easy` 随机关卡；
- 保持较稳的线性学习率衰减；
- 将熵系数从更保守的 `0.01/0.02` 提高到 `0.08`，尝试增加后期探索。

这一阶段的实验目的不是“重新建立基线”，而是：

- 在已有较强策略上，测试更高探索是否还能带来额外收益；
- 观察模型是继续提升，还是因探索过强而重新增大波动。

由于本轮对话中没有继续展示 `v3` 的完整后续曲线，因此更适合将其定位为：

**在稳定可用基线之后，对“进一步榨性能”所做的探索性续训实验。**

---

## 4. 为什么最终重点选择 `coinrun_mario_like_v1`

本轮实验最后论文撰写与后续主线更应重点围绕：

`best_model/coinrun/coinrun_mario_like_v1/best_model.zip`

展开。其原因不是单纯因为“它是最新训练出来的文件”，而是因为它更符合整个课题的真实目标。

### 4.1 本课题目标不是只做 CoinRun 高分，而是为统一智能体服务

当前项目的主线不是单独追求 CoinRun 单任务分数最大化，而是：

1. 先训练 Mario 单任务 PPO；
2. 再训练 CoinRun 单任务 PPO；
3. 将二者作为 unified/mixed 训练或模型融合的输入基础。

因此，对 CoinRun 单线模型的评价标准不能只看“该模型单独在 CoinRun 上是否分数更高”，还要看它是否：

- 更容易与 Mario 策略形成一致的行为风格；
- 更适合作为统一智能体初始化的一侧专家模型；
- 更符合论文中“Mario-like 辅助环境”的研究叙述。

### 4.2 `coinrun_mario_like_v1` 更符合“Mario-like 第二环境”的实验定位

从命名和实验用途上看，`coinrun_mario_like_v1` 的定位不是普通的 CoinRun baseline，而是：

- 将 CoinRun 作为更接近 Mario 的辅助环境；
- 强调任务节奏、奖励语义和训练目标的对齐；
- 让它在后续 unified 训练中承担“与 Mario 具有更强迁移相关性”的角色。

这类模型的价值主要体现在：

- 它不只是“能玩 CoinRun”；
- 它更适合支撑“为什么选择 CoinRun 作为 Mario 的辅助环境”这一论文叙事；
- 它更容易和 Mario 单线模型一起被解释为“两个相近平台跳跃任务的专家策略”。

### 4.3 `coinrun_mario_like_v1` 的曲线证据

在后续补充的 TensorBoard 截图中，`coinrun_mario_like_v1` 呈现出非常明确的“高成功率 + 高平均奖励 + 高效完成任务”特征：

- `eval/mean_reward` 长时间维持在约 `100` 左右的高位；
- `eval/mean_ep_length` 平滑值约为 `15`，整体保持较低；
- `rollout/coinrun_complete_rate` 与 `rollout/complete_rate` 在中前期快速上升，并长期稳定在 `0.9` 左右；
- `rollout/coinrun_mean_reward` 与 `rollout/mean_reward` 也快速提升到高位，后期大部分时间保持稳定；
- `rollout/coinrun_mean_episode_length` 与 `rollout/ep_len_mean` 从较高值快速下降后维持在较低区间，说明策略越来越高效；
- `rollout/coinrun_truncated_rate` 基本保持为 `0`，说明大多数回合不是因超时截断结束，而是真正完成或正常终止。

从训练指标看，这组实验也表现出较好的收敛质量：

- `train/explained_variance` 长时间保持在较高水平，说明 value network 对回报的拟合较稳定；
- `train/entropy_loss` 先快速变化，后期在相对稳定区间内波动，说明策略已经从强探索逐步过渡到较确定的行为；
- `train/approx_kl` 与 `train/clip_fraction` 虽然存在尖峰，但总体没有持续失控，更像收敛后期的局部震荡；
- `train/loss`、`train/value_loss` 与 `train/policy_gradient_loss` 在后期仍有波动，但结合高成功率与高奖励来看，并未破坏整体策略质量。

因此，`coinrun_mario_like_v1` 不仅在任务定位上更适合论文主线，在现有曲线证据上也可以被解释为：

**已经学会稳定完成 CoinRun 目标，并能以较短回合长度获得高回报的成熟单任务专家模型。**

### 4.4 为什么不把 `random_easy_v2/v3` 直接作为最终主模型

`random_easy_v2` 的优点是：

- 训练曲线很健康；
- 成功率与平均回报都明显提升；
- 可作为非常好的单任务 PPO 基线。

但它更偏向于回答：

**“在 CoinRun easy 随机关卡上，PPO 是否可以稳定学会任务？”**

而 `coinrun_mario_like_v1` 更适合回答：

**“在构建面向 Mario 的通用智能体时，为什么 CoinRun 可以作为更贴近 Mario 的第二环境，以及应该选哪一个 CoinRun 单线模型作为后续主线输入？”**

因此，在论文结构上，更合理的写法是：

- `random_easy_v2` 作为“训练稳定化后的单任务基线结果”；
- `coinrun_mario_like_v1` 作为“最终选用的 CoinRun 主模型”。

### 4.5 最终选择理由总结

最终选择 `coinrun_mario_like_v1`，主要基于以下几点：

1. 它更贴合整个课题“Mario + CoinRun”的主问题，而不仅是 CoinRun 单任务最优分数；
2. 它更符合“Mario-like 辅助环境”这一实验设定，便于论文叙述；
3. 它更适合作为后续 unified/fused 训练链路中的 CoinRun 侧输入模型；
4. 相比仅强调单任务曲线最好，选择更“任务对齐”的模型更符合毕业设计整体目标。

---

## 5. 阶段性结论

从本轮对话涉及的 CoinRun 单线 PPO 主线来看，模型演化大致经历了如下过程：

1. `coinrun_baseline_ppo_random_easy_10m`：
   前期能学到有效策略，但后期存在明显退化，属于“可用但不稳”的早期基线。
2. `coinrun_baseline_ppo_random_easy_10m_v2`：
   通过降低学习率、线性衰减、降低熵系数并提高 `gamma`，显著改善了训练稳定性，形成了高质量随机 easy 基线。
3. `coinrun_baseline_ppo_random_easy_v3`：
   在已有较强模型上继续提高探索，属于进一步榨取性能的探索性实验。
4. `coinrun_mario_like_v1`：
   作为最终重点模型，其价值不只在于单任务表现，还在于它更符合“为 Mario 风格统一智能体服务”的实验目标与论文叙事。

总体来说，本轮 CoinRun 单线实验的关键进展不是简单地“把 reward 做高”，而是：

**从一个后期易退化的单任务 PPO 基线，逐步调整到稳定可用的 CoinRun 专家模型，并最终收敛到更适合作为 Mario-like 辅助环境输入的主模型选择。**

---

## 6. 论文写作建议

在论文实验部分，建议将 CoinRun 单线实验写成两层结构：

### 6.1 第一层：训练稳定性演化

重点描述：

- 初始 baseline 存在后期退化；
- 通过降低学习率、降低熵系数、加入线性衰减与提高 `gamma`，训练稳定性明显提升；
- `random_easy_v2` 证明了 CoinRun 单任务 PPO 可以在 easy 随机关卡上稳定收敛。

### 6.2 第二层：最终模型选型理由

重点描述：

- 课题目标不是只做 CoinRun 单任务最优，而是服务于 Mario + CoinRun 的统一智能体训练；
- 因此最终模型选择不仅看单任务曲线，也看任务对齐程度；
- `coinrun_mario_like_v1` 因更符合 Mario-like 辅助环境定位，成为后续实验更合理的主模型。

这样的写法既能保留完整实验轨迹，又能自然解释为什么最终重点不是简单地选择某条 reward 最高的曲线，而是选择更符合整体研究目标的模型。
