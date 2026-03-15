# 增量数据上传功能

## 功能概述

`upload_incremental.sh` 是为解决数据上传效率问题而设计的增量上传脚本。它只会上传自上次上传以来新增的训练数据文件，大大减少上传时间。

## 使用方法

### 1. 收集数据
```bash
# 收集完成后会显示统计信息
```

### 2. 增量上传
```bash
# 只上传新增的数据
./server_tools/upload_incremental.sh
```

## 工作原理

### 状态跟踪
脚本使用两个文件跟踪上传状态：
- `.upload_state` - 记录上次上传的文件总数
- `.last_upload_timestamp` - 记录上次上传的时间戳
- `.collection_stats.json` - 记录收集统计信息

### 文件检测
脚本通过比较文件修改时间来识别新增文件：
- 只扫描 `data_imitation/images_processed/` 目录下的 PNG 文件
- 对比文件的修改时间与上次上传时间戳
- 自动识别所有动作文件夹（包括flipped变体）

## 优势

### 效率提升
- **首次上传**: 上传所有现有数据
- **后续上传**: 只上传新增数据，上传时间减少90%+

### 自动化
- 自动检测新增文件
- 自动创建远程目录结构
- 自动更新状态记录

### 可靠性
- 断点续传支持
- 详细的进度显示
- 上传失败时的错误报告

## 目录结构

```
data_imitation/images_processed/
├── jump/
├── noop/
├── right/
└── right_jump/
```

## 输出示例

### 数据收集完成
```
==================================================
🎯 数据收集完成！
==================================================
📊 收集统计:
  收集时长: 45.2 秒
  新增帧数: 1234
  平均FPS: 27.3 帧/秒

📁 数据分布:
  jump: +245 (总计: 245)
  noop: +190 (总计: 190)
  right: +298 (总计: 298)
  right_jump: +189 (总计: 189)

🚀 增量上传:
  运行: ./server_tools/upload_incremental.sh
  只上传本次新增的 1234 个文件
==================================================
```

### 增量上传
```
==========================================
增量上传训练数据到服务器
==========================================
📊 上次上传统计:
  文件数量: 5678
  时间: 2024-01-15 14:30:22

📈 当前统计:
  新增文件数量: 1234

📁 准备上传的新文件:
  frame_0001.png
  frame_0002.png
  ...

✅ 增量上传完成！
📊 上传统计:
  本次上传: 1234 个文件
  累计上传: 6912 个文件
```

## 故障排除

### 没有检测到新文件
```bash
# 检查数据目录
ls -la data_imitation/images_processed/

# 检查状态文件
cat .upload_state
cat .last_upload_timestamp
```

### 上传失败
```bash
# 检查网络连接
ping cloud.tanheidc.cn

# 检查SSH连接
ssh -p 2618 netzone22@cloud.tanheidc.cn

# 手动重置状态（如果需要）
rm .upload_state .last_upload_timestamp
```

### 文件权限问题
```bash
# 设置脚本执行权限
chmod +x server_tools/upload_incremental.sh
```

## 技术细节

### 文件检测逻辑
```bash
# 使用find命令检测新增文件
find "$LOCAL_IMAGES_DIR/$action_dir" \
  -name "*.png" \
  -newer "$TIMESTAMP_FILE" \
  -print0 2>/dev/null
```

### 远程目录创建
```bash
# 自动创建远程目录结构
ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "mkdir -p $remote_dir"
```

### 状态持久化
```bash
# 更新状态文件
echo "last_upload_count=$NEW_TOTAL_COUNT" > "$STATE_FILE"
echo "$(date +%s)" > "$TIMESTAMP_FILE"
```