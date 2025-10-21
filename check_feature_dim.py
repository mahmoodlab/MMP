#!/usr/bin/env python3
"""
快速检测您的病理切片特征维度

使用方法:
    python check_feature_dim.py /data/TCGA/BLCA/features/pt_files
"""

import sys
import os
import torch
from pathlib import Path

def check_feature_dimension(data_dir):
    """检查特征文件的维度"""

    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"❌ 错误: 路径不存在: {data_dir}")
        return None

    # 查找.pt文件
    pt_files = list(data_path.glob("*.pt"))

    if not pt_files:
        print(f"❌ 错误: 在 {data_dir} 中没有找到.pt文件")
        return None

    print(f"找到 {len(pt_files)} 个.pt文件")
    print(f"检查前5个文件...")
    print("=" * 80)

    dimensions = []

    for i, pt_file in enumerate(pt_files[:5]):
        try:
            # 加载特征
            features = torch.load(pt_file)

            # 获取维度
            if isinstance(features, torch.Tensor):
                shape = features.shape
            elif isinstance(features, dict):
                # 可能是字典格式，尝试找到特征
                if 'features' in features:
                    shape = features['features'].shape
                elif 'feat' in features:
                    shape = features['feat'].shape
                else:
                    # 尝试第一个tensor值
                    for key, value in features.items():
                        if isinstance(value, torch.Tensor):
                            shape = value.shape
                            break
            else:
                print(f"  ⚠️  文件 {pt_file.name}: 未知格式 {type(features)}")
                continue

            if len(shape) == 2:
                n_patches, feat_dim = shape
                dimensions.append(feat_dim)
                print(f"  ✓ 文件 {pt_file.name}:")
                print(f"      Patches: {n_patches}, 特征维度: {feat_dim}")
            else:
                print(f"  ⚠️  文件 {pt_file.name}: 形状 {shape}")

        except Exception as e:
            print(f"  ❌ 文件 {pt_file.name}: 加载失败 - {e}")

    print("=" * 80)

    if dimensions:
        unique_dims = set(dimensions)
        if len(unique_dims) == 1:
            feat_dim = dimensions[0]
            print(f"\n✅ 所有特征文件的维度一致: {feat_dim}")
            print(f"\n📋 您需要在训练时设置:")
            print(f"   --in_dim {feat_dim}")

            # 推测使用的特征提取器
            if feat_dim == 768:
                print(f"\n💡 可能使用的特征提取器: ViT-S/16, DINO, MAE 等")
            elif feat_dim == 1024:
                print(f"\n💡 可能使用的特征提取器: ResNet-34, ViT-B/16 等")
            elif feat_dim == 2048:
                print(f"\n💡 可能使用的特征提取器: ResNet-50 等")
            elif feat_dim == 384:
                print(f"\n💡 可能使用的特征提取器: ViT-Tiny 等")

            return feat_dim
        else:
            print(f"\n⚠️  警告: 发现不同的特征维度: {unique_dims}")
            return None
    else:
        print(f"\n❌ 无法确定特征维度")
        return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使用方法: python check_feature_dim.py <特征文件目录>")
        print("示例: python check_feature_dim.py /data/TCGA/BLCA/features/pt_files")
        sys.exit(1)

    data_dir = sys.argv[1]

    print("=" * 80)
    print("特征维度检测工具")
    print("=" * 80)
    print(f"检查目录: {data_dir}")
    print()

    feat_dim = check_feature_dimension(data_dir)

    if feat_dim:
        print("\n" + "=" * 80)
        print("✅ 检测完成!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ 检测失败，请检查数据格式")
        print("=" * 80)
