#!/bin/bash
# 镜像覆盖上传本地项目到服务器
# 使用方法: ./server_tools/upload_to_server.sh

set -euo pipefail

# 自动切换到项目根目录，避免从其他目录执行时路径出错
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

SERVER_HOST="cloud.tanheidc.cn"
SERVER_PORT="2618"
SERVER_USER="netzone22"
SERVER_PATH="/ai/MarioRL"

echo "=========================================="
echo "镜像覆盖上传项目到服务器"
echo "=========================================="
echo "本地目录: $PROJECT_ROOT"
echo "目标路径: $SERVER_USER@$SERVER_HOST:$SERVER_PATH"
echo ""
echo "警告: 将使用 rsync --delete 镜像同步服务器目录"
echo "说明: 服务器上本地不存在的项目文件会被删除"
echo "提示: 需要输入密码"
echo ""

read -r -p "确认继续覆盖上传? [y/N] " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "已取消上传"
    exit 0
fi

# 先确保远程目录存在，再执行镜像同步
ssh -p "$SERVER_PORT" "$SERVER_USER@$SERVER_HOST" "mkdir -p '$SERVER_PATH'"

rsync -avz \
    -e "ssh -p $SERVER_PORT" \
    --progress \
    --delete \
    --exclude ".git/" \
    --exclude ".vscode/" \
    --exclude "__pycache__/" \
    --exclude "*.pyc" \
    --exclude "*.pyo" \
    --exclude ".DS_Store" \
    ./ "$SERVER_USER@$SERVER_HOST:$SERVER_PATH/"

echo ""
echo "=========================================="
echo "✅ 项目镜像覆盖完成"
echo "=========================================="
echo "可执行以下命令验证:"
echo "ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST 'cd $SERVER_PATH && ls -la'"
