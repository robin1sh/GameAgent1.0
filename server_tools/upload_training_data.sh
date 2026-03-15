#!/bin/bash
# 上传训练数据到服务器
# 使用方法: ./server_tools/upload_training_data.sh

# 切换到项目根目录（脚本所在目录的上级目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

SERVER_HOST="cloud.tanheidc.cn"
SERVER_PORT="2618"
SERVER_USER="netzone22"
SERVER_PATH="/ai/MarioRL/data_imitation_unified"

LOCAL_PATH="data_imitation_unified"

echo "=========================================="
echo "上传训练数据到服务器"
echo "=========================================="

# 检查本地训练数据目录
if [ ! -d "$LOCAL_PATH" ]; then
    echo "❌ 错误: 找不到本地训练数据目录: $LOCAL_PATH"
    exit 1
fi

echo "本地训练数据统计:"
for dir in "$LOCAL_PATH"/*/; do
    if [ -d "$dir" ]; then
        sample_count=$(find "$dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
        if [ "$sample_count" -gt 0 ]; then
            echo "  $(basename "$dir"): $sample_count 个样本目录"
        fi
    fi
done

echo ""
echo "服务器: $SERVER_USER@$SERVER_HOST:$SERVER_PORT"
echo "目标路径: $SERVER_PATH"
echo ""
echo "提示: 需要输入密码: 130259IT"
echo "      上传可能需要一些时间，请耐心等待..."
echo ""

# 使用rsync上传（支持断点续传和进度显示）
rsync -avz -e "ssh -p $SERVER_PORT" \
    --progress \
    --exclude=".DS_Store" \
    "$LOCAL_PATH/" $SERVER_USER@$SERVER_HOST:$SERVER_PATH/

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 上传完成！"
    echo "=========================================="
    echo ""
    echo "在服务器上验证数据:"
    echo "ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST 'ls -lh $SERVER_PATH/'"
    echo ""
    echo "或检查各目录图片数量:"
    echo "ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST 'cd $SERVER_PATH && for d in */; do echo \"\$d: \$(find \$d -type f | wc -l) 张\"; done'"
else
    echo ""
    echo "❌ 上传失败，请检查网络连接和服务器信息"
    exit 1
fi

