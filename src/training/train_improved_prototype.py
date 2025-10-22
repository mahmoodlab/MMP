"""
改进版原型学习训练脚本
Improved Prototype Learning Training Script

主要改进:
1. 端到端可训练原型 (fix_proto=False)
2. 原型多样性正则化
3. 原型专用优化器和学习率
4. 原型使用情况监控
5. 可选的 Gumbel-Softmax 重参数化

使用示例:
    # 基础训练
    python src/training/train_improved_prototype.py \
        --config configs/train_improved_panther.yaml

    # 快速测试
    python src/training/train_improved_prototype.py \
        --fix_proto False \
        --em_iter 5 \
        --diversity_reg 0.01 \
        --proto_lr 5e-5

    # 使用 Gumbel-Softmax
    python src/training/train_improved_prototype.py \
        --use_gumbel True \
        --gumbel_temp 0.5
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.mil_models.PANTHER.networks_improved import ImprovedDirNIWNet
from src.mil_models.model_PANTHER import PANTHER
from src.utils.survival_loss import NLLSurvLoss, CoxLoss
from src.utils.metrics import calculate_concordance_index


class ImprovedPrototypeTrainer:
    """改进版原型学习训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if config['device']['cuda'] else 'cpu')

        # 设置随机种子
        self.set_seed(config['seed'])

        # 初始化模型
        self.model = self.build_model()

        # 初始化优化器
        self.optimizer, self.proto_optimizer = self.build_optimizers()

        # 初始化损失函数
        self.criterion = self.build_criterion()

        # 初始化日志
        self.writer = None
        if config['logging']['tensorboard']:
            log_dir = config['logging']['log_dir']
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

        # 最佳性能跟踪
        self.best_metric = -float('inf') if config['logging']['checkpoint']['mode'] == 'max' else float('inf')
        self.patience_counter = 0

    def set_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def build_model(self):
        """构建模型"""
        model_config = self.config['model']

        # 这里假设使用 PANTHER 模型
        # 实际实现中，你可能需要修改 model_PANTHER.py 来使用 ImprovedDirNIWNet

        print("\n" + "=" * 80)
        print("模型配置:")
        print(f"  - 原型数量: {model_config['out_size']}")
        print(f"  - 特征维度: {model_config['emb_dim']}")
        print(f"  - 原型可训练: {not model_config['fix_proto']}")
        print(f"  - EM 迭代次数: {model_config['em_iter']}")
        print(f"  - 多样性正则化: {model_config.get('diversity_reg', 0.0)}")
        print(f"  - 使用 Gumbel-Softmax: {model_config.get('use_gumbel', False)}")
        print("=" * 80 + "\n")

        # 创建改进的原型网络
        # 注意：这里直接创建 ImprovedDirNIWNet 作为示例
        # 实际使用时，需要修改 model_PANTHER.py 中的 PANTHERBase 来集成这个改进版本

        model = ImprovedDirNIWNet(
            p=model_config['out_size'],
            d=model_config['emb_dim'],
            load_proto=model_config.get('load_proto', False),
            proto_path=model_config.get('proto_path', None),
            fix_proto=model_config['fix_proto'],
            diversity_reg=model_config.get('diversity_reg', 0.01),
            use_gumbel=model_config.get('use_gumbel', False),
            gumbel_temp=model_config.get('gumbel_temp', 1.0),
            cov_type=model_config.get('cov_type', 'diag'),
        )

        return model.to(self.device)

    def build_optimizers(self):
        """构建优化器"""
        opt_config = self.config['training']['optimizer']

        # 检查是否使用原型专用优化器
        use_separate = opt_config.get('use_separate_proto_optimizer', False)

        if use_separate and not self.config['model']['fix_proto']:
            # 分离原型参数和其他参数
            proto_params = [self.model.m, self.model.V_]
            proto_param_ids = [id(p) for p in proto_params]

            # 其他参数 (如果模型有其他可训练参数)
            other_params = [
                p for p in self.model.parameters()
                if id(p) not in proto_param_ids and p.requires_grad
            ]

            # 原型专用优化器
            proto_lr = opt_config.get('proto_lr', opt_config['lr'] * 0.5)
            proto_optimizer = optim.AdamW(
                proto_params,
                lr=proto_lr,
                weight_decay=opt_config.get('weight_decay', 1e-5)
            )

            # 主优化器
            if other_params:
                main_optimizer = optim.AdamW(
                    other_params,
                    lr=opt_config['lr'],
                    weight_decay=opt_config.get('weight_decay', 1e-5)
                )
            else:
                main_optimizer = None

            print(f"\n使用原型专用优化器:")
            print(f"  - 原型学习率: {proto_lr}")
            print(f"  - 主学习率: {opt_config['lr']}\n")

            return main_optimizer, proto_optimizer

        else:
            # 单一优化器
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 1e-5)
            )

            print(f"\n使用单一优化器:")
            print(f"  - 学习率: {opt_config['lr']}\n")

            return optimizer, None

    def build_criterion(self):
        """构建损失函数"""
        # 生存分析损失 (这里使用 Cox 损失作为示例)
        return CoxLoss()

    def compute_diversity_loss(self):
        """计算原型多样性损失"""
        if self.config['model']['fix_proto']:
            return torch.tensor(0.0).to(self.device)

        return self.model.compute_diversity_loss()

    def train_epoch(self, train_loader, epoch):
        """训练一个 epoch"""
        self.model.train()

        total_loss = 0.0
        total_survival_loss = 0.0
        total_diversity_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # 假设 batch 包含: features, survival_time, survival_event
            features = batch['features'].to(self.device)  # (B, N, d)
            survival_time = batch['survival_time'].to(self.device)  # (B,)
            survival_event = batch['survival_event'].to(self.device)  # (B,)

            # 前向传播
            # 这里简化示例，实际需要完整的模型架构
            pi, mu, Sigma, qq = self.model.map_em(
                features,
                num_iters=self.config['model']['em_iter'],
                tau=self.config['model']['tau']
            )

            # 计算生存预测 (这里需要你的完整模型)
            # 示例：使用原型表示进行预测
            # survival_pred = self.survival_head(proto_repr)

            # 计算损失
            # survival_loss = self.criterion(survival_pred, survival_time, survival_event)

            # 为了演示，这里使用简化的损失
            survival_loss = qq.sum() * 0  # 占位符

            # 多样性损失
            diversity_loss = self.compute_diversity_loss()

            # 总损失
            loss_weights = self.config['training']['loss_weights']
            total_batch_loss = (
                loss_weights['survival_loss'] * survival_loss +
                loss_weights['diversity_loss'] * diversity_loss
            )

            # 反向传播
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            if self.proto_optimizer is not None:
                self.proto_optimizer.zero_grad()

            total_batch_loss.backward()

            # 梯度裁剪
            if self.config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )

            # 优化器步进
            if self.optimizer is not None:
                self.optimizer.step()
            if self.proto_optimizer is not None:
                self.proto_optimizer.step()

            # 统计
            total_loss += total_batch_loss.item()
            total_survival_loss += survival_loss.item()
            total_diversity_loss += diversity_loss.item()

            # 更新进度条
            pbar.set_postfix({
                'loss': total_batch_loss.item(),
                'surv': survival_loss.item(),
                'div': diversity_loss.item()
            })

        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        avg_survival_loss = total_survival_loss / len(train_loader)
        avg_diversity_loss = total_diversity_loss / len(train_loader)

        # 记录日志
        if self.writer is not None:
            self.writer.add_scalar('train/loss', avg_loss, epoch)
            self.writer.add_scalar('train/survival_loss', avg_survival_loss, epoch)
            self.writer.add_scalar('train/diversity_loss', avg_diversity_loss, epoch)

        return {
            'loss': avg_loss,
            'survival_loss': avg_survival_loss,
            'diversity_loss': avg_diversity_loss
        }

    def validate(self, val_loader, epoch):
        """验证"""
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_times = []
        all_events = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                features = batch['features'].to(self.device)
                survival_time = batch['survival_time'].to(self.device)
                survival_event = batch['survival_event'].to(self.device)

                # 前向传播
                pi, mu, Sigma, qq = self.model.map_em(
                    features,
                    num_iters=self.config['model']['em_iter'],
                    tau=self.config['model']['tau']
                )

                # 计算预测 (简化示例)
                # survival_pred = self.survival_head(proto_repr)
                # loss = self.criterion(survival_pred, survival_time, survival_event)

                # 占位符
                loss = qq.sum() * 0

                total_loss += loss.item()

                # 收集预测结果
                # all_preds.append(survival_pred.cpu())
                all_times.append(survival_time.cpu())
                all_events.append(survival_event.cpu())

        # 计算指标
        avg_loss = total_loss / len(val_loader)

        # C-index (需要实际预测)
        # all_preds = torch.cat(all_preds)
        # all_times = torch.cat(all_times)
        # all_events = torch.cat(all_events)
        # c_index = calculate_concordance_index(all_preds, all_times, all_events)

        c_index = 0.5  # 占位符

        # 原型使用统计
        proto_usage = self.analyze_prototype_usage(val_loader)

        # 记录日志
        if self.writer is not None:
            self.writer.add_scalar('val/loss', avg_loss, epoch)
            self.writer.add_scalar('val/c_index', c_index, epoch)

            # 记录原型使用情况
            for i, usage in enumerate(proto_usage):
                self.writer.add_scalar(f'val/proto_{i}_usage', usage, epoch)

        return {
            'loss': avg_loss,
            'c_index': c_index,
            'proto_usage': proto_usage
        }

    def analyze_prototype_usage(self, data_loader):
        """分析原型使用情况"""
        self.model.eval()

        proto_counts = torch.zeros(self.config['model']['out_size']).to(self.device)

        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)

                # 计算分配
                pi, mu, Sigma, qq = self.model.map_em(
                    features,
                    num_iters=self.config['model']['em_iter'],
                    tau=self.config['model']['tau']
                )

                # 统计每个原型的使用次数 (软分配)
                proto_counts += qq.sum(dim=(0, 1))

        # 归一化
        proto_usage = proto_counts / proto_counts.sum()

        return proto_usage.cpu().numpy()

    def save_checkpoint(self, epoch, metric, is_best=False):
        """保存检查点"""
        checkpoint_dir = self.config['logging']['checkpoint']['save_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'proto_optimizer_state_dict': self.proto_optimizer.state_dict() if self.proto_optimizer else None,
            'metric': metric,
            'config': self.config
        }

        # 保存最新检查点
        latest_path = os.path.join(checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)

        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"\n保存最佳模型: {best_path} (metric: {metric:.4f})")

        # 定期保存
        if epoch % self.config['logging']['checkpoint']['save_interval'] == 0:
            epoch_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)

    def train(self, train_loader, val_loader):
        """完整训练流程"""
        num_epochs = self.config['training']['epochs']

        print("\n" + "=" * 80)
        print(f"开始训练 - 共 {num_epochs} 个 epoch")
        print("=" * 80 + "\n")

        for epoch in range(1, num_epochs + 1):
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)

            print(f"\nEpoch {epoch}/{num_epochs} - 训练指标:")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  Survival Loss: {train_metrics['survival_loss']:.4f}")
            print(f"  Diversity Loss: {train_metrics['diversity_loss']:.4f}")

            # 验证
            if epoch % self.config['validation']['interval'] == 0:
                val_metrics = self.validate(val_loader, epoch)

                print(f"\nEpoch {epoch}/{num_epochs} - 验证指标:")
                print(f"  Loss: {val_metrics['loss']:.4f}")
                print(f"  C-Index: {val_metrics['c_index']:.4f}")
                print(f"  原型使用情况: {val_metrics['proto_usage'][:5]}...")

                # 检查是否是最佳模型
                metric_value = val_metrics['c_index']
                is_best = False

                if self.config['logging']['checkpoint']['mode'] == 'max':
                    if metric_value > self.best_metric:
                        self.best_metric = metric_value
                        is_best = True
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                else:
                    if metric_value < self.best_metric:
                        self.best_metric = metric_value
                        is_best = True
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1

                # 保存检查点
                self.save_checkpoint(epoch, metric_value, is_best)

                # 早停
                patience = self.config['training']['early_stopping']['patience']
                if self.patience_counter >= patience:
                    print(f"\n早停触发！连续 {patience} 个 epoch 没有改进")
                    break

        print("\n" + "=" * 80)
        print("训练完成!")
        print(f"最佳 {self.config['logging']['checkpoint']['metric']}: {self.best_metric:.4f}")
        print("=" * 80 + "\n")

        if self.writer is not None:
            self.writer.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="改进版原型学习训练")

    # 配置文件
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')

    # 快速参数 (会覆盖配置文件)
    parser.add_argument('--fix_proto', type=lambda x: x.lower() == 'true',
                       default=None, help='是否固定原型')
    parser.add_argument('--em_iter', type=int, default=None,
                       help='EM 迭代次数')
    parser.add_argument('--diversity_reg', type=float, default=None,
                       help='多样性正则化权重')
    parser.add_argument('--proto_lr', type=float, default=None,
                       help='原型学习率')
    parser.add_argument('--use_gumbel', type=lambda x: x.lower() == 'true',
                       default=None, help='是否使用 Gumbel-Softmax')
    parser.add_argument('--gumbel_temp', type=float, default=None,
                       help='Gumbel 温度')

    args = parser.parse_args()

    # 加载配置
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # 使用默认配置
        config = {
            'model': {
                'fix_proto': False,
                'em_iter': 5,
                'out_size': 16,
                'emb_dim': 1024,
                'tau': 0.001,
                'diversity_reg': 0.01,
                'use_gumbel': False,
                'gumbel_temp': 1.0,
                'cov_type': 'diag'
            },
            'training': {
                'epochs': 50,
                'batch_size': 1,
                'optimizer': {
                    'type': 'AdamW',
                    'lr': 1e-4,
                    'proto_lr': 5e-5,
                    'use_separate_proto_optimizer': True,
                    'weight_decay': 1e-5
                },
                'loss_weights': {
                    'survival_loss': 1.0,
                    'diversity_loss': 0.01
                },
                'gradient_clip': 1.0,
                'early_stopping': {
                    'patience': 10,
                    'min_delta': 0.001
                }
            },
            'validation': {
                'interval': 1
            },
            'logging': {
                'log_dir': 'logs/panther_improved',
                'tensorboard': True,
                'checkpoint': {
                    'save_dir': 'checkpoints/panther_improved',
                    'save_interval': 5,
                    'save_best': True,
                    'metric': 'c_index',
                    'mode': 'max'
                }
            },
            'device': {
                'cuda': True
            },
            'seed': 42
        }

    # 覆盖参数
    if args.fix_proto is not None:
        config['model']['fix_proto'] = args.fix_proto
    if args.em_iter is not None:
        config['model']['em_iter'] = args.em_iter
    if args.diversity_reg is not None:
        config['model']['diversity_reg'] = args.diversity_reg
    if args.proto_lr is not None:
        config['training']['optimizer']['proto_lr'] = args.proto_lr
    if args.use_gumbel is not None:
        config['model']['use_gumbel'] = args.use_gumbel
    if args.gumbel_temp is not None:
        config['model']['gumbel_temp'] = args.gumbel_temp

    print("\n最终配置:")
    print(yaml.dump(config, default_flow_style=False))

    # 创建训练器
    trainer = ImprovedPrototypeTrainer(config)

    # TODO: 加载数据
    # train_loader = ...
    # val_loader = ...

    print("\n警告: 数据加载器尚未实现，请实现您的数据加载逻辑")
    print("提示: 需要实现返回以下格式的数据加载器:")
    print("  batch = {")
    print("    'features': Tensor(B, N, d),  # WSI patches 特征")
    print("    'survival_time': Tensor(B,),   # 生存时间")
    print("    'survival_event': Tensor(B,)   # 事件标签 (0/1)")
    print("  }")

    # 开始训练
    # trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
