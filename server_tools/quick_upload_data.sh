#!/bin/bash
# 一键设置并批量上传训练数据
# 使用方法: ./quick_upload.sh

echo "🚀 一键上传 MarioRL 训练数据"
echo "========================================"
echo ""

echo "📤 开始上传训练数据（可输入密码）"
echo ""

./server_tools/upload_training_data.sh

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 全部完成！"
    echo "现在您的训练数据已全部上传到服务器"
    echo "可以开始训练了：ssh -p 2618 netzone22@cloud.tanheidc.cn 然后运行 python3 train/train_cnn_imitation_unified.py --source expert_both"
else
    echo ""
    echo "❌ 上传失败，请检查网络连接"
    exit 1
fi
