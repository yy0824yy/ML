"""
第一步：数据探索与预处理分析 (Data Analysis & Enhancement)

核心任务：
1. 加载数据集并进行质量评估
2. 计算原始图像的对比度、清晰度、亮度等指标
3. 分析血管与背景的灰度对比
4. 使用CLAHE增强，并对比增强前后的指标
5. 生成可视化对比图表
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils import DataLoader, ImageQualityAnalyzer, ImageEnhancer, plot_comparison, plot_histogram_comparison


def analyze_dataset(data_dir: str, output_dir: str):
    """
    主分析函数
    """
    print("=" * 80)
    print("第一步：数据探索与预处理分析")
    print("=" * 80)
    
    # 创建输出目录
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. 加载数据
    print("\n[1/5] 加载数据集...")
    loader = DataLoader(data_dir)
    train_images, train_masks = loader.get_training_samples()
    test_images, test_masks = loader.get_test_samples()
    
    print(f"✓ 训练集: {len(train_images)} 张图像")
    print(f"✓ 测试集: {len(test_images)} 张图像")
    
    # 2. 图像质量分析 - 原始图像
    print("\n[2/5] 分析原始图像质量...")
    analyzer = ImageQualityAnalyzer()
    
    quality_metrics = []
    
    for idx, (img, mask) in enumerate(zip(train_images, train_masks)):
        contrast = analyzer.calculate_contrast(img)
        sharpness = analyzer.calculate_sharpness(img)
        brightness = analyzer.calculate_brightness(img)
        vessel_mean, bg_mean, vessel_contrast = analyzer.analyze_vessel_contrast(img, mask)
        
        quality_metrics.append({
            'Image_ID': f'train_{idx+21}',
            'Contrast_Std': contrast,
            'Sharpness_Var': sharpness,
            'Brightness_Mean': brightness,
            'Vessel_Mean': vessel_mean,
            'Background_Mean': bg_mean,
            'Vessel_BG_Contrast': vessel_contrast
        })
    
    for idx, (img, mask) in enumerate(zip(test_images, test_masks)):
        contrast = analyzer.calculate_contrast(img)
        sharpness = analyzer.calculate_sharpness(img)
        brightness = analyzer.calculate_brightness(img)
        vessel_mean, bg_mean, vessel_contrast = analyzer.analyze_vessel_contrast(img, mask)
        
        quality_metrics.append({
            'Image_ID': f'test_{idx+1:02d}',
            'Contrast_Std': contrast,
            'Sharpness_Var': sharpness,
            'Brightness_Mean': brightness,
            'Vessel_Mean': vessel_mean,
            'Background_Mean': bg_mean,
            'Vessel_BG_Contrast': vessel_contrast
        })
    
    df_raw = pd.DataFrame(quality_metrics)
    
    print("\n原始图像质量统计（前5个）:")
    print(df_raw.head())
    print("\n原始图像质量统计信息:")
    print(df_raw[['Contrast_Std', 'Sharpness_Var', 'Vessel_BG_Contrast']].describe())
    
    # 3. CLAHE增强
    print("\n[3/5] 执行CLAHE增强...")
    enhancer = ImageEnhancer()
    
    # 对所有图像进行增强
    train_images_enhanced = []
    for img in train_images:
        enhanced = enhancer.clahe_enhance(img, clip_limit=2.0, tile_size=8)
        train_images_enhanced.append(enhanced)
    
    test_images_enhanced = []
    for img in test_images:
        enhanced = enhancer.clahe_enhance(img, clip_limit=2.0, tile_size=8)
        test_images_enhanced.append(enhanced)
    
    print("✓ CLAHE增强完成")
    
    # 4. 增强后质量分析
    print("\n[4/5] 分析增强后图像质量...")
    
    quality_metrics_enhanced = []
    
    for idx, (img_enh, mask) in enumerate(zip(train_images_enhanced, train_masks)):
        contrast = analyzer.calculate_contrast(img_enh)
        sharpness = analyzer.calculate_sharpness(img_enh)
        brightness = analyzer.calculate_brightness(img_enh)
        vessel_mean, bg_mean, vessel_contrast = analyzer.analyze_vessel_contrast(img_enh, mask)
        
        quality_metrics_enhanced.append({
            'Image_ID': f'train_{idx+21}',
            'Contrast_Std': contrast,
            'Sharpness_Var': sharpness,
            'Brightness_Mean': brightness,
            'Vessel_Mean': vessel_mean,
            'Background_Mean': bg_mean,
            'Vessel_BG_Contrast': vessel_contrast
        })
    
    for idx, (img_enh, mask) in enumerate(zip(test_images_enhanced, test_masks)):
        contrast = analyzer.calculate_contrast(img_enh)
        sharpness = analyzer.calculate_sharpness(img_enh)
        brightness = analyzer.calculate_brightness(img_enh)
        vessel_mean, bg_mean, vessel_contrast = analyzer.analyze_vessel_contrast(img_enh, mask)
        
        quality_metrics_enhanced.append({
            'Image_ID': f'test_{idx+1:02d}',
            'Contrast_Std': contrast,
            'Sharpness_Var': sharpness,
            'Brightness_Mean': brightness,
            'Vessel_Mean': vessel_mean,
            'Background_Mean': bg_mean,
            'Vessel_BG_Contrast': vessel_contrast
        })
    
    df_enhanced = pd.DataFrame(quality_metrics_enhanced)
    
    print("\nCLAHE增强后图像质量统计（前5个）:")
    print(df_enhanced.head())
    print("\nCLAHE增强后图像质量统计信息:")
    print(df_enhanced[['Contrast_Std', 'Sharpness_Var', 'Vessel_BG_Contrast']].describe())
    
    # 5. 指标对比与可视化
    print("\n[5/5] 生成对比分析...")
    
    # 计算平均改进幅度
    contrast_improvement = ((df_enhanced['Contrast_Std'].mean() - df_raw['Contrast_Std'].mean()) 
                           / df_raw['Contrast_Std'].mean() * 100)
    sharpness_improvement = ((df_enhanced['Sharpness_Var'].mean() - df_raw['Sharpness_Var'].mean()) 
                            / df_raw['Sharpness_Var'].mean() * 100)
    vessel_contrast_improvement = ((df_enhanced['Vessel_BG_Contrast'].mean() - df_raw['Vessel_BG_Contrast'].mean())
                                  / df_raw['Vessel_BG_Contrast'].mean() * 100)
    
    print("\n=== 增强效果总体对比 ===")
    print(f"对比度提升: {contrast_improvement:+.2f}%")
    print(f"清晰度提升: {sharpness_improvement:+.2f}%")
    print(f"血管-背景对比度提升: {vessel_contrast_improvement:+.2f}%")
    
    # 可视化1: 图像对比 (第一张训练图)
    print("\n生成可视化...")
    plot_comparison(
        train_images[0], 
        train_images_enhanced[0],
        'Original Image',
        'CLAHE Enhanced Image',
        os.path.join(viz_dir, '01_image_comparison.png')
    )
    print("✓ 已保存: 01_image_comparison.png")
    
    # 可视化2: 直方图对比 (第一张训练图)
    hist_raw = analyzer.get_histogram(train_images[0])
    hist_enhanced = analyzer.get_histogram(train_images_enhanced[0])
    
    plot_histogram_comparison(
        hist_raw,
        hist_enhanced,
        'Original Image Histogram',
        'CLAHE Enhanced Histogram',
        os.path.join(viz_dir, '02_histogram_comparison.png')
    )
    print("✓ 已保存: 02_histogram_comparison.png")
    
    # 可视化3: 指标对比柱状图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 对比度对比
    axes[0].bar(['Original', 'Enhanced'], 
                [df_raw['Contrast_Std'].mean(), df_enhanced['Contrast_Std'].mean()],
                color=['#FF6B6B', '#4ECDC4'])
    axes[0].set_title('Contrast (Std)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Value')
    axes[0].grid(axis='y', alpha=0.3)
    
    # 清晰度对比
    axes[1].bar(['Original', 'Enhanced'],
                [df_raw['Sharpness_Var'].mean(), df_enhanced['Sharpness_Var'].mean()],
                color=['#FF6B6B', '#4ECDC4'])
    axes[1].set_title('Sharpness (Laplacian Var)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Value')
    axes[1].grid(axis='y', alpha=0.3)
    
    # 血管-背景对比度
    axes[2].bar(['Original', 'Enhanced'],
                [df_raw['Vessel_BG_Contrast'].mean(), df_enhanced['Vessel_BG_Contrast'].mean()],
                color=['#FF6B6B', '#4ECDC4'])
    axes[2].set_title('Vessel-Background Contrast', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Value')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '03_quality_metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 03_quality_metrics_comparison.png")
    
    # 可视化4: 血管-背景对比度分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(df_raw['Vessel_BG_Contrast'], bins=15, alpha=0.7, color='#FF6B6B', edgecolor='black')
    axes[0].set_title('Original: Vessel-Background Contrast Distribution', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Contrast Value')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axvline(df_raw['Vessel_BG_Contrast'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0].legend()
    
    axes[1].hist(df_enhanced['Vessel_BG_Contrast'], bins=15, alpha=0.7, color='#4ECDC4', edgecolor='black')
    axes[1].set_title('CLAHE Enhanced: Vessel-Background Contrast Distribution', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Contrast Value')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axvline(df_enhanced['Vessel_BG_Contrast'].mean(), color='blue', linestyle='--', linewidth=2, label='Mean')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, '04_contrast_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 已保存: 04_contrast_distribution.png")
    
    # 保存指标数据
    metrics_dir = os.path.join(output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    df_raw.to_csv(os.path.join(metrics_dir, 'quality_metrics_raw.csv'), index=False)
    df_enhanced.to_csv(os.path.join(metrics_dir, 'quality_metrics_enhanced.csv'), index=False)
    
    print("✓ 已保存: quality_metrics_raw.csv, quality_metrics_enhanced.csv")
    
    # 保存增强后的图像（用于后续模型训练）
    enhanced_dir = os.path.join(output_dir, 'enhanced_images')
    os.makedirs(os.path.join(enhanced_dir, 'training'), exist_ok=True)
    os.makedirs(os.path.join(enhanced_dir, 'test'), exist_ok=True)
    
    import cv2
    for idx, img in enumerate(train_images_enhanced):
        cv2.imwrite(os.path.join(enhanced_dir, 'training', f'{idx+21:02d}_enhanced.png'), img)
    
    for idx, img in enumerate(test_images_enhanced):
        cv2.imwrite(os.path.join(enhanced_dir, 'test', f'{idx+1:02d}_enhanced.png'), img)
    
    print("✓ 已保存增强后的图像到 enhanced_images/")
    
    print("\n" + "=" * 80)
    print("第一步完成！")
    print("=" * 80)
    
    return {
        'df_raw': df_raw,
        'df_enhanced': df_enhanced,
        'train_images': train_images,
        'train_masks': train_masks,
        'test_images': test_images,
        'test_masks': test_masks,
        'train_images_enhanced': train_images_enhanced,
        'test_images_enhanced': test_images_enhanced
    }


if __name__ == '__main__':
    # 配置路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'datasets')
    output_dir = os.path.join(project_root, 'output')
    
    # 运行分析
    results = analyze_dataset(data_dir, output_dir)
