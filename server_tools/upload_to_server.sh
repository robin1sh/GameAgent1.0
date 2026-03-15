#!/bin/bash
# 上传项目到服务器的脚本
# 使用方法: bash server_tools/upload_to_server.sh

SERVER_HOST="cloud.tanheidc.cn"
SERVER_PORT="2618"
SERVER_USER="netzone22"
SERVER_PATH="/ai/MarioRL"  # 服务器上的目标路径

echo "=========================================="
echo "上传项目到服务器"
echo "=========================================="
echo "服务器: $SERVER_HOST:$SERVER_PORT"
echo "用户: $SERVER_USER"
echo "目标路径: $SERVER_PATH"
echo ""

# 排除不需要上传的文件
EXCLUDE_PATTERNS=(
    "--exclude=.git"
    "--exclude=__pycache__"
    "--exclude=*.pyc"
    "--exclude=.DS_Store"
    "--exclude=*.log"
    "--exclude=*.pt"
    "--exclude=best_model"
    "--exclude=callback_logs"
    "--exclude=logs"
)

# unified 数据目录体积较大，默认不上传；需要时可显式开启：
# INCLUDE_UNIFIED_DATA=1 ./server_tools/upload_to_server.sh
if [ "${INCLUDE_UNIFIED_DATA:-0}" != "1" ]; then
    EXCLUDE_PATTERNS+=("--exclude=data_imitation_unified")
fi

echo "开始上传..."
echo "提示: 首次上传需要输入密码: 130259IT"
echo "注意: 当前为完全覆盖模式，会删除远端多余文件"
if [ "${INCLUDE_UNIFIED_DATA:-0}" = "1" ]; then
    echo "当前模式: 包含 data_imitation_unified 数据目录"
else
    echo "当前模式: 默认排除 data_imitation_unified（可用 INCLUDE_UNIFIED_DATA=1 开启）"
fi
echo ""

# 使用rsync上传（推荐，支持断点续传和增量更新）
rsync -avz -e "ssh -p $SERVER_PORT" \
    --delete \
    --delete-excluded \
    --progress \
    "${EXCLUDE_PATTERNS[@]}" \
    ./ $SERVER_USER@$SERVER_HOST:$SERVER_PATH/

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 上传完成！"
    echo "=========================================="
    echo ""
    echo "下一步操作:"
    echo "1. SSH连接到服务器:"
    echo "   ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST"
    echo ""
    echo "2. 进入项目目录:"
    echo "   cd $SERVER_PATH"
    echo ""
    echo "3. 安装依赖（如果还没安装）:"
    echo "   pip3 install -r current_requirements.txt"
    echo ""
    echo "4. 开始训练:"
    echo "   screen -S mario_train"
    echo "   python3 train/train_cnn_imitation.py"
    echo "   # 按 Ctrl+A 然后按 D 分离"
    echo ""
else
    echo ""
    echo "✗ 上传失败，请检查网络连接和服务器信息"
    exit 1
fi

