"""
第五步：K折交叉验证训练脚本

所有三组实验都将使用5折交叉验证：
1. Baseline: 原始图像 + U-Net
2. Enhanced + U-Net: 增强数据 + 标准U-Net
3. Enhanced + Attention U-Net: 增强数据 + 注意力U-Net

这能确保使用了全部20张图像进行评估，结果更具统计意义。
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
import cv2
from tqdm import tqdm
import json
from sklearn.model_selection import KFold
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from step2_models import UNet, AttentionUNet, TransUNet, BCEDiceLoss
from utils import DataLoader as CustomDataLoader
from step3_train import RetinalVesselDataset, Trainer

def train_k_fold(experiment_name, all_images, all_masks, model_class, output_dir, device, k=5, num_epochs=30):
    """
    K折交叉验证训练
    """
    print("\n" + "="*80)
    print(f"开始 {k}-Fold 交叉验证: {experiment_name}")
    print("="*80)
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_images)):
        print(f"\nTraining Fold {fold+1}/{k}...")
        
        # 准备数据
        train_imgs = [all_images[i] for i in train_idx]
        train_msks = [all_masks[i] for i in train_idx]
        val_imgs = [all_images[i] for i in val_idx]
        val_msks = [all_masks[i] for i in val_idx]
        
        train_dataset = RetinalVesselDataset(train_imgs, train_msks)
        val_dataset = RetinalVesselDataset(val_imgs, val_msks)
        
        train_loader = TorchDataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=4, shuffle=False)
        
        # 初始化模型
        model = model_class(in_channels=3, out_channels=1).to(device)
        trainer = Trainer(model, device, lr=0.001, num_epochs=num_epochs)
        
        # 训练
        model_save_path = os.path.join(output_dir, 'models', f'{experiment_name}_fold{fold+1}.pth')
        trainer.train(train_loader, val_loader, model_save_path)
        
        # 记录最佳验证loss
        fold_metrics.append({
            'Fold': fold + 1,
            'Best_Val_Loss': trainer.best_val_loss
        })
        
        # 释放显存
        del model
        del trainer
        torch.cuda.empty_cache()
    
    # 保存该实验的汇总结果
    df_metrics = pd.DataFrame(fold_metrics)
    avg_loss = df_metrics['Best_Val_Loss'].mean()
    print(f"\n{experiment_name} K-Fold完成。平均验证Loss: {avg_loss:.4f}")
    
    return df_metrics

def run_k_fold_pipeline(data_dir, output_dir, device):
    """运行所有实验的K-Fold流程"""
    
    # 1. 加载数据 (只用训练集文件夹的20张，因为测试集无标注)
    print("\n[1/3] 加载数据...")
    loader = CustomDataLoader(data_dir)
    images_raw, masks = loader.get_training_samples()
    
    # 加载增强后的图像
    enhanced_dir = os.path.join(output_dir, 'enhanced_images', 'training')
    images_enhanced = []
    
    for i in range(20): # 20张训练图
        path = os.path.join(enhanced_dir, f'{i+21}_enhanced.png')
        if not os.path.exists(path):
            raise FileNotFoundError(f"增强图像未找到: {path}。请先运行第一步。")
        images_enhanced.append(path)
    
    print(f"✓ 数据加载完成: {len(images_raw)} 张原始图, {len(images_enhanced)} 张增强图")
    
    # K-Fold 设置 (30 epochs够了，因为验证频繁)
    K = 5
    EPOCHS = 30
    
    # 实验1: Baseline (Raw + UNet)
    exp1_metrics = train_k_fold(
        'Exp1_Baseline_KFold', 
        images_raw, masks, 
        UNet, output_dir, device, 
        k=K, num_epochs=EPOCHS
    )
    
    # 实验2: Enhanced + UNet
    exp2_metrics = train_k_fold(
        'Exp2_Enhanced_KFold', 
        images_enhanced, masks, 
        UNet, output_dir, device, 
        k=K, num_epochs=EPOCHS
    )
    
    # 实验3: Enhanced + AttentionUNet
    exp3_metrics = train_k_fold(
        'Exp3_AttUNet_KFold', 
        images_enhanced, masks, 
        AttentionUNet, output_dir, device, 
        k=K, num_epochs=EPOCHS
    )
    
    # 实验4: Enhanced + TransUNet
    print("\n" + "="*80)
    print("⭐ 开始实验4: Enhanced + TransUNet（Transformer架构）")
    print("="*80)
    exp4_metrics = train_k_fold(
        'Exp4_TransUNet_KFold', 
        images_enhanced, masks, 
        TransUNet, output_dir, device, 
        k=K, num_epochs=EPOCHS
    )
    
    # 汇总保存
    history_path = os.path.join(output_dir, 'metrics', 'kfold_summary.csv')
    
    exp1_metrics['Experiment'] = 'Baseline (Raw + UNet)'
    exp2_metrics['Experiment'] = 'Enhanced + UNet'
    exp3_metrics['Experiment'] = 'Enhanced + AttentionUNet'
    exp4_metrics['Experiment'] = 'Enhanced + TransUNet'
    
    final_df = pd.concat([exp1_metrics, exp2_metrics, exp3_metrics, exp4_metrics])
    final_df.to_csv(history_path, index=False)
    
    print("\n" + "="*80)
    print(f"所有K-Fold实验完成。汇总已保存至: {history_path}")
    print("="*80)

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'datasets')
    output_dir = os.path.join(project_root, 'output')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    run_k_fold_pipeline(data_dir, output_dir, device)
