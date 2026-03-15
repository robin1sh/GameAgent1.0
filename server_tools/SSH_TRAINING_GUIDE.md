# SSH服务器训练指南（MarioRL）

本指南帮助你在服务器上训练 `MarioRL` 的模仿学习模型。

## 📋 前置准备

### 1) 检查 Python 与 GPU

```bash
python3 --version
python3 -c "import torch; print('torch:', torch.__version__)"
python3 -c "import torch; print('GPU可用:', torch.cuda.is_available())"
```

### 2) 安装依赖

```bash
pip3 install -r current_requirements.txt
```

如果遇到 `gym==0.21.0` 安装失败，可先降级 pip/setuptools：

```bash
pip install "pip<23.0" "setuptools<58.0"
```

### 3) 上传项目文件到服务器

**服务器信息：**
- 主机: `cloud.tanheidc.cn`
- 端口: `2618`
- 用户名: `netzone22`

**方法1: 使用提供的上传脚本（推荐）**

```bash
chmod +x server_tools/upload_to_server.sh
./server_tools/upload_to_server.sh
```

**方法2: 手动上传**

```bash
rsync -avz -e "ssh -p 2618" --progress ./ netzone22@cloud.tanheidc.cn:/ai/MarioRL/
```

## 🚀 训练方法

### 方法1: 使用 screen（推荐）

```bash
screen -S mario_train
python3 train/train_cnn_imitation.py
# 按 Ctrl+A 然后按 D 分离
```

重新连接：

```bash
screen -r mario_train
```

### 方法2: nohup 后台运行

```bash
nohup python3 train/train_cnn_imitation.py > train.log 2>&1 &
tail -f train.log
```

## 📊 监控训练进度

```bash
nvidia-smi
tail -f train.log
```

## 📁 训练结果

训练结束后会生成：

- `imitation_cnn.pt`（模型权重）
- `class_to_idx.json`（类别映射）

## 🎯 快速开始

```bash
# 1. 上传项目
./server_tools/upload_to_server.sh

# 2. SSH 连接
ssh -p 2618 netzone22@cloud.tanheidc.cn

# 3. 进入项目
cd /ai/MarioRL

# 4. 安装依赖
pip3 install -r current_requirements.txt

# 5. 启动训练
screen -S mario_train
python3 train/train_cnn_imitation.py
```

祝训练顺利！🎉

