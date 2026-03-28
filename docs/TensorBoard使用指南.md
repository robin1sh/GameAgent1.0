# TensorBoard 使用指南：围绕当前 unified 主线看曲线

> 适用场景：Mario 单任务、CoinRun 单任务、fused 后 unified mixed 训练、`--resume` 长时续训。

---

## 本地查看速查（Windows / PowerShell）

如果你是在自己电脑本地训练，不需要 SSH 和端口转发，直接在项目根目录启动 TensorBoard 即可。

### 1. 进入项目根目录

```powershell
cd C:\Users\ROG\Desktop\MarioRL-main
```

### 2. 启动 TensorBoard

查看整个项目下所有训练日志：

```powershell
python -m tensorboard.main --logdir .\logs --port 6006
```

只查看 Mario 日志：

```powershell
python -m tensorboard.main --logdir .\logs\mario --port 6006
```

只查看某一次实验：

```powershell
python -m tensorboard.main --logdir .\logs\mario\local_gpu_fast --port 6006
```

查看 unified 一次实验（`train_unified.py`）：

```powershell
python -m tensorboard.main --logdir .\logs\unified_from_fused_v2 --port 6006
```

### 3. 浏览器打开

```text
http://localhost:6006
```

### 4. 关闭 TensorBoard

```text
在当前终端按 Ctrl + C
```

### 5. 如果 6006 端口被占用

```powershell
python -m tensorboard.main --logdir .\logs --port 6007
```

然后浏览器访问：

```text
http://localhost:6007
```

### 6. 常用补充命令

查看 TensorBoard 是否已安装：

```powershell
python -m tensorboard.main --version
```

如果命令行里看不到曲线，先确认日志目录里有事件文件：

```powershell
dir .\logs\mario\local_gpu_fast
```

> 说明：本项目默认 TensorBoard 日志根目录是 `.\logs\`。  
> - 单任务入口通常写到 `.\logs\mario\` / `.\logs\coinrun\`  
> - `train_unified.py` 写到 `.\logs\{exp-id}\`  
> - 当前最需要重点关注的是 unified 实验目录，而不是单看单任务目录

---

## 服务器信息

| 项目 | 值 |
|------|----|
| 主机 | `cloud.tanheidc.cn` |
| 端口 | `2618` |
| 用户名 | `netzone22` |
| 密码 | `130259IT` |
| 项目目录 | `/ai/MarioRL` |

---

## 第一步：登录服务器

在本地打开终端（PowerShell / CMD）：

```bash
ssh -p 2618 netzone22@cloud.tanheidc.cn
# 提示输入密码时输入：130259IT
```

登录成功后进入项目目录：

```bash
cd /ai/MarioRL
```

---

## 第二步：用 screen 启动训练（防止断线中断）

screen 是 Linux 的会话管理工具，SSH 断开后训练仍会继续运行。

### 训练 Mario

```bash
# 新建名为 mario_train 的会话
screen -S mario_train

# 在会话内启动训练（日志写入 ./logs/mario/mario_v1/）
python3 train_model/train_ppo_mario.py \
    --total-timesteps 10000000 \
    --n-envs 10 \
    --exp-id mario_v1

# 训练启动后，按 Ctrl+A 再按 D 分离会话（训练保持后台运行）
```

### 训练 CoinRun

```bash
# 新建另一个会话（可与 Mario 同时运行）
screen -S coinrun_train

python3 train_model/train_ppo_coinrun.py \
    --total-timesteps 10000000 \
    --n-envs 10 \
    --exp-id coinrun_v1

# 按 Ctrl+A 再按 D 分离
```

### 重新连接会话（查看训练输出）

```bash
screen -r mario_train    # 重连 Mario 会话
screen -r coinrun_train  # 重连 CoinRun 会话
screen -ls               # 列出所有会话
```

---

## 第三步：在服务器上启动 TensorBoard

重新开一个 SSH 连接（或新建 screen 会话）专门跑 TensorBoard：

```bash
# 连接服务器
ssh -p 2618 netzone22@cloud.tanheidc.cn

# 新建 screen 会话
screen -S tensorboard

# 进入项目目录
cd /ai/MarioRL

# 启动 TensorBoard，同时监控 Mario 和 CoinRun 日志
tensorboard --logdir ./logs/ --host 0.0.0.0 --port 6006

# 按 Ctrl+A 再按 D 分离，TensorBoard 保持后台运行
```

> `--host 0.0.0.0` 允许外部通过端口转发访问；`--logdir ./logs/` 会自动识别 `mario/` 和 `coinrun/` 两个子目录并分组显示。

---

## 第四步：本地通过 SSH 端口转发查看

服务器的 6006 端口无法直接从外网访问，需要在**本地**建立端口转发隧道。

打开**另一个**本地终端窗口，执行：

```bash
ssh -p 2618 -L 6006:localhost:6006 netzone22@cloud.tanheidc.cn
# 输入密码：130259IT
# 这个窗口保持不要关闭，它是隧道本身
```

然后打开本地浏览器，访问：

```
http://localhost:6006
```

即可看到 TensorBoard 界面。

---

## 第五步：在 TensorBoard 中看什么

### 5.1 切换到 SCALARS 面板

进入后默认显示 **SCALARS** 标签页，左侧可过滤指标名称。

### 5.2 当前最重要的指标

| 指标 | 含义 | 当前怎么用 |
|------|------|-------------|
| `rollout/mario_mean_reward` | mixed 训练下 Mario 子任务平均回报 | 判断 Mario 是否被 unified 训练保住 |
| `rollout/coinrun_mean_reward` | mixed 训练下 CoinRun 子任务平均回报 | 判断 CoinRun 是否被忽视或崩掉 |
| `rollout/mario_complete_rate` | Mario 子任务通关率 | 看是否真正学到完成关卡，而非只拿局部奖励 |
| `rollout/coinrun_complete_rate` | CoinRun 子任务通关率 | 看 CoinRun 是否只会拿塑形分，不会完成任务 |
| `eval_mario/*` | Mario 独立评估回调输出 | 比 rollout 更适合做版本间横向比较 |
| `eval_coinrun/*` | CoinRun 独立评估回调输出 | 检查 mixed 训练是否真的提升 CoinRun 泛化 |
| `train/approx_kl` | 每次更新策略的变化幅度 | 过大说明 resume 后更新过猛 |
| `train/clip_fraction` | PPO clip 触发比例 | 过高时通常意味着策略跳变太大 |
| `train/value_loss` | 价值函数拟合误差 | 长时间居高不下时，说明奖励尺度可能不稳 |
| `train/entropy_loss` | 策略探索强度 | 下降过快可能导致早熟策略 |

### 5.3 当前最常见的读图判断

1. `mario_mean_reward` 上升，但 `coinrun_mean_reward` 长期不动  
说明 Mario 占比过高，或者 CoinRun 奖励塑形太弱。

2. `coinrun_mean_reward` 上升，但 `mario_complete_rate` 下滑  
说明 CoinRun 奖励过强，或者 `mario_ratio` 太低。

3. 两边 reward 都有波动，但 `eval_*` 长期不涨  
说明策略只是在适应训练分布，泛化没有真正变好。

4. resume 之后曲线突然剧烈抖动  
优先检查这次续训是否改动了奖励参数、并行环境数、归一化开关。

### 5.4 对比 Mario 与 CoinRun

单任务训练时，TensorBoard 会以 `mario/mario_v1` 和 `coinrun/coinrun_v1` 为分组，自动用不同颜色区分。

左侧 **Runs** 面板可勾选/取消某条曲线，便于单独查看或两者叠加对比。

对于 unified 训练，更推荐这样对比：

- 比较 `unified_from_fused_v1` 与 `unified_from_fused_v2`
- 看 `eval_mario/*` 与 `eval_coinrun/*` 是否同步变好
- 不要只看总 reward，必须同时看两边子任务指标

### 5.5 当前推荐的对比对象

建议至少保留这几组实验在 TensorBoard 中横向比：

- `mario_baseline_*`
- `coinrun_baseline_*`
- `fused_*`
- `unified_from_fused_v1`
- `unified_from_fused_v2`
- `unified_pretrain_*`（如果你做 imitation warm start 对照）

---

## 第六步：关闭与清理

```bash
# 重连 TensorBoard 会话
screen -r tensorboard

# 按 Ctrl+C 停止 TensorBoard

# 退出 screen 会话
exit
```

---

## 常用 screen 速查

| 命令 | 作用 |
|------|------|
| `screen -S 名字` | 新建会话 |
| `screen -ls` | 列出所有会话 |
| `screen -r 名字` | 重新连接会话 |
| `Ctrl+A` → `D` | 分离会话（保持后台运行） |
| `Ctrl+A` → `K` | 彻底关闭当前会话 |

---

## 完整流程速查

```bash
# ① 本地：登录服务器
ssh -p 2618 netzone22@cloud.tanheidc.cn

# ② 服务器：启动 Mario 训练
screen -S mario_train
cd /ai/MarioRL
python3 train_model/train_ppo_mario.py --exp-id mario_v1 --total-timesteps 10000000
# Ctrl+A D 分离

# ③ 服务器：启动 CoinRun 训练
screen -S coinrun_train
python3 train_model/train_ppo_coinrun.py --exp-id coinrun_v1 --total-timesteps 10000000
# Ctrl+A D 分离

# ③.5 服务器：启动 unified mixed 训练
screen -S unified_train
python3 train_model/train_unified.py --mode mixed --resume best_model/unified_from_fused_v1/final_model.zip --exp-id unified_from_fused_v2 --n-envs 10 --mario-ratio 0.6 --total-timesteps 10000000 --fixed-level --start-level 0 --distribution-mode easy --coinrun-progress-coef 0.05 --coinrun-success-bonus 10.0 --coinrun-fail-penalty 2.0 --coinrun-step-penalty 0.002
# Ctrl+A D 分离

# ④ 服务器：启动 TensorBoard
screen -S tensorboard
tensorboard --logdir ./logs/ --host 0.0.0.0 --port 6006
# Ctrl+A D 分离

# ⑤ 本地新终端：建立端口转发隧道
ssh -p 2618 -L 6006:localhost:6006 netzone22@cloud.tanheidc.cn

# ⑥ 本地浏览器访问
# http://localhost:6006
```
