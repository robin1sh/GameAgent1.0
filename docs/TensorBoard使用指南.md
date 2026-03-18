# TensorBoard 使用指南：从登录服务器到查看训练曲线

> 适用环境：Mario / Jumper PPO 训练，在远程 GPU 服务器上运行，本地浏览器查看。

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

> 说明：本项目默认 TensorBoard 日志根目录是 `.\logs\`，Mario 默认写到 `.\logs\mario\`，Jumper 默认写到 `.\logs\jumper\`。

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

### 训练 Jumper

```bash
# 新建另一个会话（可与 Mario 同时运行）
screen -S jumper_train

python3 train_model/train_ppo_jumper.py \
    --total-timesteps 10000000 \
    --n-envs 10 \
    --exp-id jumper_v1

# 按 Ctrl+A 再按 D 分离
```

### 重新连接会话（查看训练输出）

```bash
screen -r mario_train    # 重连 Mario 会话
screen -r jumper_train   # 重连 Jumper 会话
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

# 启动 TensorBoard，同时监控 Mario 和 Jumper 日志
tensorboard --logdir ./logs/ --host 0.0.0.0 --port 6006

# 按 Ctrl+A 再按 D 分离，TensorBoard 保持后台运行
```

> `--host 0.0.0.0` 允许外部通过端口转发访问；`--logdir ./logs/` 会自动识别 `mario/` 和 `jumper/` 两个子目录并分组显示。

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

### 5.2 关键指标说明

| 指标 | 含义 | 健康表现 |
|------|------|----------|
| `rollout/ep_rew_mean` | **平均回合回报**，最核心的指标 | 随训练步数持续上升 |
| `rollout/ep_len_mean` | 平均回合长度（步数） | Mario：越长说明存活/前进更远；Jumper：趋于稳定后上升 |
| `train/approx_kl` | 每次更新策略的变化幅度 | 保持在 `0.01` 附近，过大（>0.05）说明更新过激 |
| `train/clip_fraction` | PPO clip 触发比例 | 保持在 `0.1~0.3`，过高说明策略跳变 |
| `train/entropy_loss` | 策略熵（负值），反映探索程度 | 训练早期绝对值较大，后期逐渐减小属正常 |
| `train/value_loss` | 价值函数拟合误差 | 应持续下降并趋于平稳 |
| `train/loss` | 总损失 | 参考用，趋势比绝对值更重要 |

### 5.3 对比 Mario 与 Jumper

TensorBoard 会以 `mario/mario_v1` 和 `jumper/jumper_v1` 为分组，自动用不同颜色区分。

左侧 **Runs** 面板可勾选/取消某条曲线，便于单独查看或两者叠加对比。

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

# ③ 服务器：启动 Jumper 训练
screen -S jumper_train
python3 train_model/train_ppo_jumper.py --exp-id jumper_v1 --total-timesteps 10000000
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
