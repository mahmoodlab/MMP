#!/bin/bash

# 清理嵌入缓存脚本
#
# 当您遇到维度不匹配错误时使用此脚本
# 例如: RuntimeError: Expected size ... but got ...
#
# 使用方法: ./clean_embeddings.sh [split_dir]

SPLIT_DIR=${1:-"splits/TCGA_BLCA_survival_k=0"}

echo "==========================================="
echo "清理嵌入缓存"
echo "==========================================="
echo "Split目录: $SPLIT_DIR"
echo ""

# 检查目录是否存在
if [ ! -d "$SPLIT_DIR" ]; then
    echo "❌ 错误: 目录不存在: $SPLIT_DIR"
    exit 1
fi

# 检查embeddings目录
EMB_DIR="$SPLIT_DIR/embeddings"
if [ ! -d "$EMB_DIR" ]; then
    echo "✓ 没有找到embeddings目录，无需清理"
    exit 0
fi

# 列出现有的嵌入文件
echo "发现以下嵌入文件:"
echo "-------------------------------------------"
ls -lh "$EMB_DIR"/*.pkl 2>/dev/null
echo "-------------------------------------------"
echo ""

# 询问确认
read -p "是否删除这些文件? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # 备份（可选）
    BACKUP_DIR="$SPLIT_DIR/embeddings_backup_$(date +%Y%m%d_%H%M%S)"
    echo "创建备份到: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    cp "$EMB_DIR"/*.pkl "$BACKUP_DIR"/ 2>/dev/null

    # 删除
    echo "删除嵌入文件..."
    rm -f "$EMB_DIR"/*.pkl

    echo ""
    echo "✅ 清理完成!"
    echo "   - 原文件已备份到: $BACKUP_DIR"
    echo "   - 下次训练时将重新生成嵌入"
else
    echo "取消操作"
    exit 0
fi

echo ""
echo "==========================================="
echo "现在可以重新运行训练:"
echo "  cd scripts/survival"
echo "  ./train_blca_causal_separation.sh 0 0"
echo "==========================================="
