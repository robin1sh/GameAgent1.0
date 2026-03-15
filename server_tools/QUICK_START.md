# 🚀 快速开始 - 服务器训练

## 服务器信息
- **主机**: `cloud.tanheidc.cn`
- **端口**: `2618`
- **用户名**: `netzone22`
- **密码**: `130259IT`

## 三步开始训练

### 步骤1: 上传项目到服务器

```bash
# 在本地项目目录执行
chmod +x server_tools/upload_to_server.sh
./server_tools/upload_to_server.sh
# 输入密码: 130259IT
```

#### 💡 增量上传数据（推荐）
首次上传项目后，可以使用增量上传功能，只上传新增的训练数据：

```bash
# 设置脚本权限
chmod +x server_tools/upload_incremental.sh

# 只上传新增的数据（效率提升90%+）
./server_tools/upload_incremental.sh
```

### 步骤2: 连接到服务器并安装依赖

```bash
# 连接到服务器
ssh -p 2618 netzone22@cloud.tanheidc.cn
# 输入密码: 130259IT

# 进入项目目录
cd /ai/MarioRL

# 安装依赖
pip3 install -r current_requirements.txt
```

### 步骤3: 开始训练

```bash
# 使用screen创建会话（推荐）
screen -S mario_train

# 开始训练
python3 train/train_cnn_imitation.py

# 按 Ctrl+A 然后按 D 来分离会话（训练继续运行）
# SSH断开后，训练仍会继续
```

### 查看训练进度

```bash
# 重新连接服务器
ssh -p 2618 netzone22@cloud.tanheidc.cn

# 重新连接screen会话
screen -r mario_train

# 或查看日志文件（如果使用train_ssh.sh）
tail -f train.log
```

## 📝 常用命令

```bash
# 查看所有screen会话
screen -ls

# 重新连接screen会话
screen -r mario_train

# 查看GPU使用情况（如果有GPU）
nvidia-smi

# 查看训练日志
tail -f train.log

# 查看模型文件
ls -lh *.pt

# 停止训练（找到进程ID后）
ps aux | grep "train/train_cnn_imitation"
kill <PID>
```

## 🔧 故障排除

### 连接问题
```bash
# 测试连接
ssh -p 2618 netzone22@cloud.tanheidc.cn

# 如果连接超时，检查网络和防火墙设置
```

### 训练中断
- 使用 `screen` 或 `tmux` 可以防止SSH断开导致训练中断
- 训练完成会保存 `imitation_cnn.pt` 与 `class_to_idx.json`

### 查看训练结果
```bash
# 查看训练日志
tail -50 train.log
```

## 📚 更多信息

详细指南请查看: `SSH_TRAINING_GUIDE.md`

