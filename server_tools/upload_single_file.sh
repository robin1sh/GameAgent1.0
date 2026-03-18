#!/bin/bash
# 快速上传单个文件到服务器
# 使用方法: ./server_tools/upload_single_file.sh <文件名>

FILE_NAME="$1"
SERVER_HOST="cloud.tanheidc.cn"
SERVER_PORT="2618"
SERVER_USER="netzone22"
SERVER_PATH="/ai/MarioRL"

if [ -z "$FILE_NAME" ]; then
    echo "❌ 错误: 未指定文件名"
    echo "使用方法: $0 <文件名>"
    exit 1
fi

if [ ! -f "$FILE_NAME" ]; then
    echo "❌ 错误: 找不到文件 $FILE_NAME"
    echo "使用方法: $0 <文件名>"
    exit 1
fi

echo "=========================================="
echo "上传文件到服务器"
echo "=========================================="
echo "文件: $FILE_NAME"
echo "服务器: $SERVER_USER@$SERVER_HOST:$SERVER_PORT"
echo "目标路径: $SERVER_PATH/$FILE_NAME"
echo ""
echo "提示: 需要输入密码: 130259IT"
echo ""

scp -P $SERVER_PORT "$FILE_NAME" $SERVER_USER@$SERVER_HOST:$SERVER_PATH/$FILE_NAME

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 上传成功！"
    echo ""
    echo "在服务器上验证文件:"
    echo "ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST 'ls -lh $SERVER_PATH/$FILE_NAME'"
else
    echo ""
    echo "❌ 上传失败"
    exit 1
fi
