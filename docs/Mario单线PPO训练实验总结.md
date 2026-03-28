# Mario 单线 PPO 训练实验总结

## 1. 实验目的

本文档用于总结本轮对话中 Mario 单线 PPO 训练的主要实验过程、参数变化、曲线现象与阶段性结论，可作为论文实验部分的草稿基础。

需要说明的是：

- 本文结论主要依据 TensorBoard 曲线截图与人工渲染观察整理。
- 个别阶段未直接回读原始日志文件，因此文中更适合表述为“实验现象总结”，后续写论文时建议再配合原始事件文件核对一次具体数值。

---

## 2. 基线训练设置

单线 Mario 训练入口脚本为 `train_model/train_ppo_mario.py`，其默认超参数为：

```python
--learning-rate 3e-4
--n-steps 2048
--batch-size 2048
--n-epochs 10
--ent-coef 0.1
--gamma 0.95
```

在本轮实验讨论中，默认并行环境数通常为 `10`，总步数通常设为 `1000 万`，评估间隔为 `100000` 环境步。

---

## 3. 训练过程变化总结

### 3.1 第一阶段：`mario_baseline_ppo_v1`

代表命令：

```bash
python train_model/train_ppo_mario.py --n-envs 10 --total-timesteps 10000000 --eval-freq 100000 --exp-id mario_baseline_ppo_v1
```

从 `mario_baseline_ppo_v1` 的 TensorBoard 曲线可观察到：

- `rollout/ep_rew_mean` 与 `rollout/mean_reward` 持续上升；
- `rollout/ep_len_mean` 与 `rollout/mean_episode_length` 明显下降；
- `rollout/complete_rate` 基本维持在 0；
- `eval/mean_reward` 与 `eval/mean_ep_length` 波动较大，稳定性不足；
- `train/approx_kl` 与 `train/clip_fraction` 在后期反而抬升；
- `train/explained_variance` 先升高后回落；
- `train/value_loss` 在后期维持较高水平。

这一阶段的主要结论是：

- 模型已经学会更高效地获取奖励，表现为回合长度下降而平均回报上升；
- 但 `complete_rate` 仍为 0，说明尚未形成稳定的成功通关能力；
- 后期 `approx_kl`、`clip_fraction` 抬升，表明策略更新幅度变大，训练稳定性一般；
- 因此 `v1` 可视为“初步学会有效推进但尚未稳定收敛”的早期基线。

---

### 3.2 第二阶段：`mario_baseline_ppo_v2`

在后续更清晰的截图中，`mario_baseline_ppo_v2` 呈现出与第一阶段相似但更明显的趋势：

- `rollout/ep_rew_mean` 与 `rollout/mean_reward` 持续上升；
- `rollout/ep_len_mean` 与 `rollout/mean_episode_length` 也持续上升；
- `train/explained_variance` 维持较高水平；
- `train/approx_kl` 与 `train/clip_fraction` 继续下降；
- `rollout/complete_rate` 依旧接近 0。

这一阶段说明：

- 模型继续强化了“生存更久、推进更远、累计更多奖励”的行为；
- 但从日志指标看，依旧没有把成功率显著拉起来；
- 因此可以判断，模型当时仍主要停留在高回报但未稳定完成任务的阶段。

简而言之，`v2` 相比早期基线更强，但依旧没有从根本上解决“最后成功完成关卡”的问题。

---

### 3.3 第三阶段：从 `v2` 恢复训练并提高探索，得到 `mario_baseline_ppo_v3`

为了在不破坏原有稳定性的前提下增加探索，后续采用如下命令继续训练：

```bash
python train_model/train_ppo_mario.py --resume "./best_model/mario/mario_baseline_ppo_v2/ppo_mario_final.zip" --n-envs 10 --total-timesteps 10000000 --eval-freq 100000 --exp-id mario_baseline_ppo_v3 --learning-rate 2e-4 --lr-schedule linear --final-learning-rate 8e-5 --n-steps 2048 --batch-size 2048 --n-epochs 10 --ent-coef 0.12 --gamma 0.95
```

本次调整的核心思路是：

- 将 `ent_coef` 从 `0.1` 提高到 `0.12`，增强策略探索；
- 将学习率从 `3e-4` 下调为 `2e-4`；
- 引入线性衰减学习率，尽量在训练后期保持稳定。

#### 中期现象（约 590 万步附近）

从中期曲线看：

- `eval/mean_reward` 前期上升明显，后期一度回落；
- `eval/mean_ep_length` 波动较大；
- `rollout/complete_rate` 仍接近 0；
- `train/entropy_loss` 没有过快贴近 0，说明探索性有所保留；
- `train/approx_kl` 与 `train/clip_fraction` 仍在下降，说明后期更新逐渐变保守。

这说明 `v3` 的“提高探索但尽量稳”的思路部分生效：

- 的确保住了一些探索；
- 但尚未在中期直接转化为稳定成功率提升。

#### 终期现象（1000 万步）

在训练跑满 `1000 万步` 后，曲线继续发生变化：

- `eval/mean_reward` 在后半段重新抬升；
- `rollout/ep_rew_mean` 与 `rollout/mean_reward` 创出更高水平；
- `rollout/ep_len_mean` 与 `rollout/mean_episode_length` 在后期出现下降；
- `train/entropy_loss` 后期重新变得更负，说明仍保留一定探索；
- `train/explained_variance` 后期有所走弱；
- `train/value_loss` 后期抬升较明显；
- `rollout/complete_rate` 日志中仍未明显抬升。

综合来看，`v3` 相比 `v2` 的主要进步在于：

- 不只是“活得更久”，而是后期开始表现出更高效的行为；
- 训练后半段 reward 重新走强，说明探索调整并非无效；
- 但从日志统计上，依然没有把 `complete_rate` 明确拉高。

---

### 3.4 第四阶段：从 `v3` 恢复训练并转向稳定收敛，得到 `mario_baseline_ppo_v4_stable`

为了降低终点前的偶发失误，并尽量固化已有策略，后续继续采用稳定化配置训练：

```bash
python train_model/train_ppo_mario.py --resume "./best_model/mario/mario_baseline_ppo_v3/ppo_mario_final.zip" --n-envs 10 --total-timesteps 10000000 --eval-freq 100000 --exp-id mario_baseline_ppo_v4_stable --learning-rate 1e-4 --lr-schedule linear --final-learning-rate 3e-5 --n-steps 2048 --batch-size 2048 --n-epochs 10 --ent-coef 0.08 --gamma 0.95
```

这一阶段的调参目标是：

- 适当降低探索强度；
- 进一步减小学习率；
- 利用线性衰减让训练后期更加平稳；
- 将优化目标从“继续寻找新行为”切换为“稳定复现已有有效行为”。

从 `mario_baseline_ppo_v4_stable` 的 TensorBoard 曲线可观察到：

- `eval/mean_reward` 整体维持在较高水平，后期没有明显塌陷；
- `eval/mean_ep_length` 相比前一阶段有所下降，但波动仍然较大；
- `rollout/ep_rew_mean` 与 `rollout/mean_reward` 继续稳定上升；
- `rollout/ep_len_mean` 与 `rollout/mean_episode_length` 维持在较低区间，相比早期更稳定；
- `rollout/complete_rate` 依然接近 0；
- `train/approx_kl` 与 `train/clip_fraction` 持续下降，说明策略更新幅度进一步收敛；
- `train/entropy_loss` 整体向更接近 0 的方向移动，说明探索进一步减弱；
- `train/explained_variance` 维持在较高但有波动的水平；
- `train/value_loss` 在后期有所回落，表明价值网络拟合压力较 `v3` 更可控。

这一阶段的主要结论是：

- `v4_stable` 基本达成了“稳定为主”的调参目标；
- 与 `v3` 相比，训练后期的更新更加保守，reward 曲线更平顺；
- 模型已经能够较稳定地复现高回报行为，但日志统计中的 `complete_rate` 仍未突破；
- 因此 `v4_stable` 更适合作为“稳定收敛版单线 Mario PPO 模型”，而不是最终的成功率突破版本。

---

## 4. 渲染测试观察

在后续人工渲染测试中，观察到以下现象：

- Mario 大概率可以通过第一关；
- 但有时会卡在旗帜前的障碍物附近；
- 也就是说，模型已经掌握了主体路线，但最后一小段动作时机仍不够稳定。

这一观察非常重要，因为它和 TensorBoard 中“`complete_rate` 长期为 0”的现象存在一定不一致。

这说明至少存在以下两种可能：

1. `complete_rate` 的统计定义并不等同于“人工观察到的通关”；
2. 模型已经接近完成第一关，但终点前仍存在局部不稳定点，导致成功率无法完全稳定。

因此，`v3` 更准确的定位应当是：

**模型已接近学会第一关，但在终点前障碍处仍存在偶发失误。**

---

## 5. 阶段性结论

从本轮对话涉及的 Mario 单线训练主线来看，模型表现大致经历了如下变化：

1. `mario_baseline_ppo_v1` 阶段：
   初步学会有效推进和获取奖励，但训练后期稳定性一般，尚未形成稳定通关能力。
2. `mario_baseline_ppo_v2` 阶段：
   高回报趋势更加明显，但成功率指标仍未突破，属于较强的次优策略。
3. `mario_baseline_ppo_v3` 阶段：
   通过小幅增加探索并降低学习率，后期 reward 进一步走强，人工渲染观察表明模型已大概率能够通过第一关，但终点前稳定性仍不足。
4. `mario_baseline_ppo_v4_stable` 阶段：
   在 `v3` 基础上降低探索并继续衰减学习率后，模型高回报行为更加稳定，训练过程更偏向收敛，但成功率统计仍未显著抬升。

总体来说，本轮单线 PPO 训练的主要进展不是“从完全不会到完全通关”，而是：

**从早期有效推进策略，逐步逼近到接近稳定通过第一关，并开始向稳定收敛版本过渡。**

---

## 6. 当前阶段的结果定位与后续方向

从当前结果看，`mario_baseline_ppo_v4_stable` 已经验证了“降低探索、强调稳定性”的方向是有效的：

- 高回报行为可以在更长训练区间内被较稳定地保留下来；
- 训练更新幅度进一步减小；
- 但成功率统计与人工观察之间仍存在差异。

因此，后续更合理的方向不再是继续大幅调节 PPO 基础超参数，而是优先解决“成功统计”和“最后障碍物稳定性”问题：

- 核对 `complete_rate` 的定义是否与人工通关标准一致；
- 针对旗帜前最后障碍物进行更细粒度的渲染观察；
- 在确认统计逻辑后，再决定是否需要继续做小幅稳定化微调。

`v4_stable` 所对应的稳定化命令为：

```bash
python train_model/train_ppo_mario.py --resume "./best_model/mario/mario_baseline_ppo_v3/ppo_mario_final.zip" --n-envs 10 --total-timesteps 10000000 --eval-freq 100000 --exp-id mario_baseline_ppo_v4_stable --learning-rate 1e-4 --lr-schedule linear --final-learning-rate 3e-5 --n-steps 2048 --batch-size 2048 --n-epochs 10 --ent-coef 0.08 --gamma 0.95
```

该配置的目标不是进一步提升随机探索，而是：

- 固化已学到的有效行为；
- 降低旗帜前障碍物附近的偶发失败；
- 提高第一关通关过程的一致性与可复现性。

---

## 7. 可直接用于论文的简短结论

可在论文中将本轮实验概括为：

“在 Mario 单线 PPO 训练中，模型经历了从早期有效推进、到高回报次优策略、再到提高探索后接近通过第一关、最后转向稳定收敛的过程。其中，`mario_baseline_ppo_v3` 通过小幅提高熵正则并降低学习率，使模型后期评估回报进一步提升；而 `mario_baseline_ppo_v4_stable` 在此基础上降低探索强度并继续衰减学习率，使高回报行为得到更稳定的保持。结合人工渲染测试可见，智能体已大概率能够通过第一关，但在旗帜前最后障碍处仍存在偶发失误，说明当前策略已进入稳定化阶段，后续重点应放在成功统计口径核对与局部失误修正，而非继续大幅提高探索强度。”
