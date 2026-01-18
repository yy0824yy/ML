"""
方案C：失败案例分析

分析Baseline模型中性能异常的样本：
1. 找出9个低性能样本 (Dice < 0.3)
2. 可视化这些样本的原图、掩码和预测结果
3. 分析失败原因（对比度、血管复杂度等）
4. 对比增强后的改进效果
"""

import os
import sys
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from step2_models import UNet, AttentionUNet
from utils import DataLoader, ImageQualityAnalyzer


# 从analyze_results.py中发现的9个失败案例
FAILURE_CASES = [
    {'fold': 1, 'image_idx': 21, 'dice': 0.2472, 'sens': 0.1414, 'spec': 0.9998},
    {'fold': 1, 'image_idx': 38, 'dice': 0.2750, 'sens': 0.1614, 'spec': 0.9988},
    {'fold': 2, 'image_idx': 24, 'dice': 0.0002, 'sens': 0.0001, 'spec': 1.0000},  # 极端失败
    {'fold': 2, 'image_idx': 29, 'dice': 0.2903, 'sens': 0.1702, 'spec': 0.9998},
    {'fold': 2, 'image_idx': 32, 'dice': 0.0028, 'sens': 0.0014, 'spec': 1.0000},  # 极端失败
    {'fold': 3, 'image_idx': 23, 'dice': 0.1973, 'sens': 0.9802, 'spec': 0.4395},  # 高误检
    {'fold': 4, 'image_idx': 25, 'dice': 0.1526, 'sens': 0.0841, 'spec': 0.9981},
    {'fold': 4, 'image_idx': 33, 'dice': 0.1597, 'sens': 0.0899, 'spec': 0.9968},
    {'fold': 5, 'image_idx': 27, 'dice': 0.2976, 'sens': 0.1816, 'spec': 0.9962},
]


def load_sample_data(data_dir, output_dir, image_idx):
    """加载指定样本的原图、增强图和掩码"""
    
    # 加载原始图像
    loader = DataLoader(data_dir)
    train_images, train_masks = loader.get_training_samples()
    
    # image_idx是21-40，在list中索引是0-19
    idx_in_list = image_idx - 21
    
    raw_image = train_images[idx_in_list]
    mask = train_masks[idx_in_list]
    
    # 加载增强图像
    enhanced_path = os.path.join(output_dir, 'enhanced_images', 'training', f'{image_idx}_enhanced.png')
    enhanced_image = cv2.imread(enhanced_path, cv2.IMREAD_GRAYSCALE)
    
    return raw_image, enhanced_image, mask


def predict_with_model(model, image, device='cuda'):
    """使用模型进行预测"""
    model.eval()
    
    # 预处理
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image
    
    img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).squeeze().cpu().numpy()
    
    pred_mask = (prob > 0.5).astype(np.uint8) * 255
    
    return pred_mask, prob


def analyze_failure_case(raw_img, enhanced_img, gt_mask, raw_pred, enhanced_pred, case_info):
    """分析单个失败案例"""
    
    analyzer = ImageQualityAnalyzer()
    
    # 计算图像质量指标
    raw_contrast = analyzer.calculate_contrast(raw_img)
    raw_sharpness = analyzer.calculate_sharpness(raw_img)
    raw_vessel_mean, raw_bg_mean, raw_vessel_contrast = analyzer.analyze_vessel_contrast(raw_img, gt_mask)
    
    enh_contrast = analyzer.calculate_contrast(enhanced_img)
    enh_sharpness = analyzer.calculate_sharpness(enhanced_img)
    enh_vessel_mean, enh_bg_mean, enh_vessel_contrast = analyzer.analyze_vessel_contrast(enhanced_img, gt_mask)
    
    # 计算预测准确度
    raw_correct = ((raw_pred > 0) == (gt_mask > 0)).sum() / gt_mask.size
    enh_correct = ((enhanced_pred > 0) == (gt_mask > 0)).sum() / gt_mask.size
    
    analysis = {
        'image_idx': case_info['image_idx'],
        'baseline_dice': case_info['dice'],
        'raw_contrast': raw_contrast,
        'raw_sharpness': raw_sharpness,
        'raw_vessel_contrast': raw_vessel_contrast,
        'enh_contrast': enh_contrast,
        'enh_sharpness': enh_sharpness,
        'enh_vessel_contrast': enh_vessel_contrast,
        'raw_accuracy': raw_correct,
        'enh_accuracy': enh_correct,
        'improvement': (enh_correct - raw_correct) * 100
    }
    
    return analysis


def visualize_failure_case(raw_img, enhanced_img, gt_mask, raw_pred, enhanced_pred, 
                          case_info, analysis, save_path=None):
    """可视化失败案例的详细对比"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 标题
    fig.suptitle(f'Failure Case Analysis - Image {case_info["image_idx"]} '
                 f'(Baseline Dice={case_info["dice"]:.4f})', 
                 fontsize=16, fontweight='bold')
    
    # 第一行：原始图像流程
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(raw_img, cmap='gray')
    ax1.set_title(f'Raw Image\nContrast={analysis["raw_contrast"]:.1f}\n'
                  f'Sharpness={analysis["raw_sharpness"]:.1f}', fontsize=10)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(gt_mask, cmap='gray')
    ax2.set_title('Ground Truth\nVessel Mask', fontsize=10)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(raw_pred, cmap='gray')
    ax3.set_title(f'Baseline Prediction\nAccuracy={analysis["raw_accuracy"]*100:.1f}%', fontsize=10)
    ax3.axis('off')
    
    # 错误分布
    ax4 = fig.add_subplot(gs[0, 3])
    error_map = np.zeros((*raw_img.shape[:2], 3), dtype=np.uint8)
    # 红色：假阳性 (预测有但实际没有)
    error_map[..., 0] = ((raw_pred > 0) & (gt_mask == 0)) * 255
    # 蓝色：假阴性 (预测没有但实际有)
    error_map[..., 2] = ((raw_pred == 0) & (gt_mask > 0)) * 255
    ax4.imshow(error_map)
    ax4.set_title('Baseline Errors\nRed=FP, Blue=FN', fontsize=10)
    ax4.axis('off')
    
    # 第二行：增强图像流程
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.imshow(enhanced_img, cmap='gray')
    ax5.set_title(f'CLAHE Enhanced\nContrast={analysis["enh_contrast"]:.1f}\n'
                  f'Sharpness={analysis["enh_sharpness"]:.1f}', fontsize=10)
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.imshow(gt_mask, cmap='gray')
    ax6.set_title('Ground Truth\n(Same)', fontsize=10)
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.imshow(enhanced_pred, cmap='gray')
    ax7.set_title(f'Enhanced Model Prediction\nAccuracy={analysis["enh_accuracy"]*100:.1f}%', 
                  fontsize=10)
    ax7.axis('off')
    
    # 增强后的错误分布
    ax8 = fig.add_subplot(gs[1, 3])
    error_map_enh = np.zeros((*enhanced_img.shape[:2], 3), dtype=np.uint8)
    error_map_enh[..., 0] = ((enhanced_pred > 0) & (gt_mask == 0)) * 255
    error_map_enh[..., 2] = ((enhanced_pred == 0) & (gt_mask > 0)) * 255
    ax8.imshow(error_map_enh)
    ax8.set_title('Enhanced Errors\nRed=FP, Blue=FN', fontsize=10)
    ax8.axis('off')
    
    # 第三行：详细分析
    ax9 = fig.add_subplot(gs[2, :2])
    metrics = ['Contrast', 'Sharpness', 'Vessel-BG\nContrast']
    raw_vals = [analysis['raw_contrast'], analysis['raw_sharpness'], analysis['raw_vessel_contrast']]
    enh_vals = [analysis['enh_contrast'], analysis['enh_sharpness'], analysis['enh_vessel_contrast']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax9.bar(x - width/2, raw_vals, width, label='Raw Image', color='coral')
    ax9.bar(x + width/2, enh_vals, width, label='Enhanced Image', color='skyblue')
    ax9.set_ylabel('Value', fontsize=11)
    ax9.set_title('Image Quality Metrics Comparison', fontsize=12, fontweight='bold')
    ax9.set_xticks(x)
    ax9.set_xticklabels(metrics)
    ax9.legend()
    ax9.grid(axis='y', alpha=0.3)
    
    # 改进率总结
    ax10 = fig.add_subplot(gs[2, 2:])
    summary_text = f"""
Failure Analysis Summary:

Baseline Performance:
• Dice Score: {case_info['dice']:.4f}
• Sensitivity: {case_info['sens']:.4f} (nearly zero)
• Specificity: {case_info['spec']:.4f}

Root Cause:
• Low Raw Contrast: {analysis['raw_contrast']:.1f}
• Low Sharpness: {analysis['raw_sharpness']:.1f}
• Poor Vessel-BG Separation: {analysis['raw_vessel_contrast']:.1f}

CLAHE Enhancement Effect:
• Contrast: {analysis['raw_contrast']:.1f} → {analysis['enh_contrast']:.1f} ({(analysis['enh_contrast']/analysis['raw_contrast']-1)*100:+.1f}%)
• Sharpness: {analysis['raw_sharpness']:.1f} → {analysis['enh_sharpness']:.1f} ({(analysis['enh_sharpness']/analysis['raw_sharpness']-1)*100:+.1f}%)

Performance Improvement:
• Accuracy: {analysis['raw_accuracy']*100:.1f}% → {analysis['enh_accuracy']*100:.1f}% ({analysis['improvement']:+.1f}%)
    """.strip()
    
    ax10.text(0.1, 0.5, summary_text, transform=ax10.transAxes,
             fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax10.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    print("="*80)
    print("方案C：失败案例分析")
    print("="*80)
    
    # 路径设置
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'datasets')
    output_dir = os.path.join(project_root, 'output')
    model_dir = os.path.join(output_dir, 'models')
    failure_dir = os.path.join(output_dir, 'visualizations', 'failure_analysis')
    
    os.makedirs(failure_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载模型
    print("\n[1/3] 加载模型...")
    
    # Baseline (Raw + UNet)
    baseline_path = os.path.join(model_dir, 'Exp1_Baseline_KFold_fold1.pth')
    model_baseline = UNet(in_channels=3, out_channels=1).to(device)
    model_baseline.load_state_dict(torch.load(baseline_path, map_location=device))
    print("✓ Baseline模型加载完成")
    
    # Enhanced + AttentionUNet (最佳模型)
    attunet_path = os.path.join(model_dir, 'Exp3_AttUNet_KFold_fold1.pth')
    model_attunet = AttentionUNet(in_channels=3, out_channels=1).to(device)
    model_attunet.load_state_dict(torch.load(attunet_path, map_location=device))
    print("✓ AttentionUNet模型加载完成")
    
    # 分析所有失败案例
    print("\n[2/3] 分析失败案例...")
    
    all_analyses = []
    
    # 选择最具代表性的5个案例进行详细可视化
    representative_cases = [
        FAILURE_CASES[2],  # Image 24: 极端失败 (Dice=0.0002)
        FAILURE_CASES[4],  # Image 32: 极端失败 (Dice=0.0028)
        FAILURE_CASES[5],  # Image 23: 高误检 (Sens=0.98, Spec=0.44)
        FAILURE_CASES[6],  # Image 25: 低检出 (Sens=0.08)
        FAILURE_CASES[0],  # Image 21: 中等失败 (Dice=0.25)
    ]
    
    for case_info in representative_cases:
        image_idx = case_info['image_idx']
        print(f"\n分析 Image {image_idx} (Dice={case_info['dice']:.4f})...")
        
        # 加载数据
        raw_img, enhanced_img, gt_mask = load_sample_data(data_dir, output_dir, image_idx)
        
        # 预测
        raw_pred, _ = predict_with_model(model_baseline, raw_img, device)
        enhanced_pred, _ = predict_with_model(model_attunet, enhanced_img, device)
        
        # 分析
        analysis = analyze_failure_case(raw_img, enhanced_img, gt_mask, 
                                       raw_pred, enhanced_pred, case_info)
        all_analyses.append(analysis)
        
        # 可视化
        save_path = os.path.join(failure_dir, f'failure_case_image_{image_idx}.png')
        visualize_failure_case(raw_img, enhanced_img, gt_mask, 
                             raw_pred, enhanced_pred, 
                             case_info, analysis, save_path)
        
        print(f"  原图质量: 对比度={analysis['raw_contrast']:.1f}, 锐度={analysis['raw_sharpness']:.1f}")
        print(f"  增强改进: 准确度提升 {analysis['improvement']:.1f}%")
    
    # 保存分析结果
    df_analysis = pd.DataFrame(all_analyses)
    csv_path = os.path.join(output_dir, 'metrics', 'failure_case_analysis.csv')
    df_analysis.to_csv(csv_path, index=False)
    print(f"\n✓ 分析结果已保存: {csv_path}")
    
    # 生成汇总报告
    print("\n[3/3] 生成失败案例汇总...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Failure Cases Summary Analysis', fontsize=16, fontweight='bold')
    
    # 图1: 对比度 vs Baseline Dice
    axes[0, 0].scatter(df_analysis['raw_contrast'], df_analysis['baseline_dice'], 
                       s=100, alpha=0.6, color='red')
    axes[0, 0].set_xlabel('Raw Image Contrast', fontsize=11)
    axes[0, 0].set_ylabel('Baseline Dice Score', fontsize=11)
    axes[0, 0].set_title('Impact of Contrast on Performance', fontsize=12)
    axes[0, 0].grid(alpha=0.3)
    
    # 图2: 锐度 vs Baseline Dice
    axes[0, 1].scatter(df_analysis['raw_sharpness'], df_analysis['baseline_dice'],
                       s=100, alpha=0.6, color='orange')
    axes[0, 1].set_xlabel('Raw Image Sharpness', fontsize=11)
    axes[0, 1].set_ylabel('Baseline Dice Score', fontsize=11)
    axes[0, 1].set_title('Impact of Sharpness on Performance', fontsize=12)
    axes[0, 1].grid(alpha=0.3)
    
    # 图3: 增强前后准确度对比
    x_labels = [f'Img {row["image_idx"]}' for _, row in df_analysis.iterrows()]
    x_pos = np.arange(len(x_labels))
    width = 0.35
    
    axes[1, 0].bar(x_pos - width/2, df_analysis['raw_accuracy']*100, width, 
                   label='Baseline (Raw)', color='coral')
    axes[1, 0].bar(x_pos + width/2, df_analysis['enh_accuracy']*100, width,
                   label='Enhanced', color='skyblue')
    axes[1, 0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1, 0].set_title('Accuracy Improvement by Enhancement', fontsize=12)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(x_labels, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 图4: 改进率统计
    axes[1, 1].bar(x_labels, df_analysis['improvement'], color='green', alpha=0.7)
    axes[1, 1].set_ylabel('Accuracy Improvement (%)', fontsize=11)
    axes[1, 1].set_title('Enhancement Effect on Failed Cases', fontsize=12)
    axes[1, 1].set_xticklabels(x_labels, rotation=45)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    summary_path = os.path.join(failure_dir, 'failure_cases_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 汇总图表已保存: {summary_path}")
    
    # 打印统计摘要
    print("\n" + "="*80)
    print("失败案例统计摘要")
    print("="*80)
    print(f"\n分析样本数: {len(df_analysis)}")
    print(f"\n原图平均质量:")
    print(f"  对比度: {df_analysis['raw_contrast'].mean():.1f} ± {df_analysis['raw_contrast'].std():.1f}")
    print(f"  锐度:   {df_analysis['raw_sharpness'].mean():.1f} ± {df_analysis['raw_sharpness'].std():.1f}")
    print(f"\n增强后平均质量:")
    print(f"  对比度: {df_analysis['enh_contrast'].mean():.1f} ± {df_analysis['enh_contrast'].std():.1f}")
    print(f"  锐度:   {df_analysis['enh_sharpness'].mean():.1f} ± {df_analysis['enh_sharpness'].std():.1f}")
    print(f"\n性能改进:")
    print(f"  平均准确度提升: {df_analysis['improvement'].mean():.1f}%")
    print(f"  最大准确度提升: {df_analysis['improvement'].max():.1f}%")
    print(f"  最小准确度提升: {df_analysis['improvement'].min():.1f}%")
    
    print("\n" + "="*80)
    print("失败案例分析完成！")
    print("="*80)
    print(f"详细分析图表保存在: {failure_dir}")
    print(f"分析数据保存在: {csv_path}")


if __name__ == '__main__':
    main()
