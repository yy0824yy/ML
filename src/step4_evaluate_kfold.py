"""
第四步：K-Fold模型评估与可视化

评估所有4个实验的5折模型，计算详细的分割指标
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from step2_models import UNet, AttentionUNet, TransUNet
from utils import DataLoader as CustomDataLoader

class Evaluator:
    """模型评估器"""
    
    def __init__(self, device):
        self.device = device
        
    def load_model(self, model_class, model_path):
        """加载训练好的模型"""
        model = model_class(in_channels=3, out_channels=1)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def predict(self, model, image):
        """单张图像推理"""
        # 预处理
        h, w = image.shape[:2]
        img_resized = cv2.resize(image, (512, 512))
        
        # 转为灰度图并扩展为3通道
        if len(img_resized.shape) == 2:
            img_tensor = np.stack([img_resized] * 3, axis=0)
        else:
            img_tensor = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            img_tensor = np.stack([img_tensor] * 3, axis=0)
            
        img_tensor = img_tensor.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_tensor).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            
        # 恢复尺寸
        pred = cv2.resize(pred, (w, h))
        return pred

    def calculate_metrics(self, pred, mask, threshold=0.5):
        """计算评估指标"""
        # 二值化
        pred_bin = (pred > threshold).astype(np.uint8)
        mask_bin = (mask > 127).astype(np.uint8)
        
        # 展平
        y_true = mask_bin.flatten()
        y_pred = pred_bin.flatten()
        
        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # 计算指标
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)
        sensitivity = tp / (tp + fn + 1e-7)
        specificity = tn / (tn + fp + 1e-7)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        
        return {
            'Dice': dice,
            'IoU': iou,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Accuracy': accuracy
        }

def evaluate_kfold_experiments(data_dir, output_dir, device):
    """评估所有K-Fold实验"""
    
    print("="*80)
    print("第四步：K-Fold模型评估与可视化")
    print("="*80)
    
    # 1. 加载数据
    print("\n[1/3] 加载数据...")
    loader = CustomDataLoader(data_dir)
    images_raw, masks = loader.get_training_samples()
    
    # 加载增强图像
    enhanced_dir = os.path.join(output_dir, 'enhanced_images', 'training')
    images_enhanced = []
    for i in range(20):
        path = os.path.join(enhanced_dir, f'{i+21}_enhanced.png')
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        images_enhanced.append(img)
    
    print(f"✓ 数据加载完成: {len(images_raw)} 张原始图, {len(images_enhanced)} 张增强图")
    
    # 2. 定义实验配置
    experiments = [
        ('Exp1_Baseline_KFold', UNet, images_raw, 'Baseline (Raw + UNet)'),
        ('Exp2_Enhanced_KFold', UNet, images_enhanced, 'Enhanced + UNet'),
        ('Exp3_AttUNet_KFold', AttentionUNet, images_enhanced, 'Enhanced + AttentionUNet'),
        ('Exp4_TransUNet_KFold', TransUNet, images_enhanced, 'Enhanced + TransUNet'),
    ]
    
    evaluator = Evaluator(device)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_results = []
    
    # 3. 评估每个实验的每个fold
    print("\n[2/3] 评估所有模型...")
    
    for exp_name, model_class, images, display_name in experiments:
        print(f"\n评估: {display_name}")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
            model_path = os.path.join(output_dir, 'models', f'{exp_name}_fold{fold+1}.pth')
            
            if not os.path.exists(model_path):
                print(f"  ⚠️  模型文件未找到: {model_path}")
                continue
            
            # 加载模型
            model = evaluator.load_model(model_class, model_path)
            
            # 在验证集上评估
            val_images = [images[i] for i in val_idx]
            val_masks = [masks[i] for i in val_idx]
            
            for img_idx, (img, mask) in enumerate(zip(val_images, val_masks)):
                pred = evaluator.predict(model, img)
                metrics = evaluator.calculate_metrics(pred, mask)
                
                metrics.update({
                    'Experiment': display_name,
                    'Fold': fold + 1,
                    'Image_Idx': val_idx[img_idx] + 21  # 真实图像编号
                })
                
                all_results.append(metrics)
            
            # 释放显存
            del model
            torch.cuda.empty_cache()
            
            print(f"  ✓ Fold {fold+1}/5 完成")
    
    # 4. 保存详细结果
    df_results = pd.DataFrame(all_results)
    results_path = os.path.join(output_dir, 'metrics', 'kfold_evaluation_results.csv')
    df_results.to_csv(results_path, index=False)
    print(f"\n✓ 详细结果已保存: {results_path}")
    
    # 5. 汇总统计
    print("\n[3/3] 生成汇总统计...")
    
    summary = df_results.groupby('Experiment')[['Dice', 'IoU', 'Sensitivity', 'Specificity', 'Accuracy']].agg(['mean', 'std'])
    
    print("\n" + "="*80)
    print("模型性能汇总（均值 ± 标准差）")
    print("="*80)
    
    for exp in ['Baseline (Raw + UNet)', 'Enhanced + UNet', 'Enhanced + AttentionUNet', 'Enhanced + TransUNet']:
        print(f"\n{exp}:")
        print(f"  Dice:        {summary.loc[exp, ('Dice', 'mean')]:.4f} ± {summary.loc[exp, ('Dice', 'std')]:.4f}")
        print(f"  IoU:         {summary.loc[exp, ('IoU', 'mean')]:.4f} ± {summary.loc[exp, ('IoU', 'std')]:.4f}")
        print(f"  Sensitivity: {summary.loc[exp, ('Sensitivity', 'mean')]:.4f} ± {summary.loc[exp, ('Sensitivity', 'std')]:.4f}")
        print(f"  Specificity: {summary.loc[exp, ('Specificity', 'mean')]:.4f} ± {summary.loc[exp, ('Specificity', 'std')]:.4f}")
    
    # 6. 生成可视化
    generate_visualizations(df_results, output_dir)
    
    print("\n" + "="*80)
    print("评估完成！所有结果已保存。")
    print("="*80)
    
    return df_results, summary

def generate_visualizations(df_results, output_dir):
    """生成对比可视化"""
    
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 设置更好的样式
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 11
    
    # 1. Dice系数箱线图
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_results, x='Experiment', y='Dice', ax=ax, palette='Set2')
    ax.set_title('Dice Coefficient Comparison (5-Fold Cross Validation)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'kfold_dice_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: kfold_dice_boxplot.png")
    
    # 2. 多指标对比（条形图）
    summary_mean = df_results.groupby('Experiment')[['Dice', 'IoU', 'Sensitivity', 'Specificity']].mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    summary_mean.plot(kind='bar', ax=ax, width=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim([0.5, 1.0])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'kfold_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: kfold_metrics_comparison.png")
    
    # 3. Sensitivity vs Specificity散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'Baseline (Raw + UNet)': '#FF6B6B', 
              'Enhanced + UNet': '#4ECDC4',
              'Enhanced + AttentionUNet': '#45B7D1',
              'Enhanced + TransUNet': '#FFA07A'}
    
    for exp, color in colors.items():
        data = df_results[df_results['Experiment'] == exp]
        ax.scatter(data['Specificity'], data['Sensitivity'], 
                  label=exp, alpha=0.6, s=100, color=color, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Specificity (Background Accuracy)', fontsize=12)
    ax.set_ylabel('Sensitivity (Vessel Recall)', fontsize=12)
    ax.set_title('Sensitivity vs Specificity Trade-off', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.9, 1.0])
    ax.set_ylim([0.5, 1.0])
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'kfold_sensitivity_specificity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: kfold_sensitivity_specificity.png")
    
    # 4. 性能提升百分比图
    baseline_dice = df_results[df_results['Experiment'] == 'Baseline (Raw + UNet)']['Dice'].mean()
    
    improvements = []
    for exp in ['Baseline (Raw + UNet)', 'Enhanced + UNet', 'Enhanced + AttentionUNet', 'Enhanced + TransUNet']:
        exp_dice = df_results[df_results['Experiment'] == exp]['Dice'].mean()
        improvement = ((exp_dice - baseline_dice) / baseline_dice) * 100
        improvements.append({'Model': exp.replace('Baseline (Raw + UNet)', 'Baseline'),
                           'Improvement (%)': improvement})
    
    df_imp = pd.DataFrame(improvements)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(df_imp)), df_imp['Improvement (%)'], 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
    ax.set_xticks(range(len(df_imp)))
    ax.set_xticklabels(df_imp['Model'], rotation=15, ha='right')
    ax.set_ylabel('Dice Improvement over Baseline (%)', fontsize=12)
    ax.set_title('Performance Improvement Analysis', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'kfold_improvement_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: kfold_improvement_analysis.png")

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'datasets')
    output_dir = os.path.join(project_root, 'output')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    df_results, summary = evaluate_kfold_experiments(data_dir, output_dir, device)
