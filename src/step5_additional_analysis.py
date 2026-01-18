"""
补充分析：训练集 vs 测试集质量对比

目的：解释为什么增强在测试集上收益有限
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(__file__))

def analyze_train_vs_test_quality():
    """对比训练集和测试集的原始质量"""
    
    print("="*80)
    print("补充分析：训练集 vs 测试集质量对比")
    print("="*80)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_dir = os.path.join(project_root, 'output', 'metrics')
    
    # 读取第一步生成的质量指标
    df_raw = pd.read_csv(os.path.join(metrics_dir, 'quality_metrics_raw.csv'))
    
    # 分离训练集和测试集
    df_train = df_raw[df_raw['Image_ID'].str.contains('train')]
    df_test = df_raw[df_raw['Image_ID'].str.contains('test')]
    
    print("\n=== 训练集 vs 测试集 质量对比 ===\n")
    
    print("训练集 (21-40) 原始质量:")
    print(df_train[['Contrast_Std', 'Sharpness_Var', 'Vessel_BG_Contrast']].describe())
    
    print("\n测试集 (01-20) 原始质量:")
    print(df_test[['Contrast_Std', 'Sharpness_Var', 'Vessel_BG_Contrast']].describe())
    
    # 计算差异
    print("\n=== 平均值对比 ===")
    comparison = pd.DataFrame({
        'Training Set': df_train[['Contrast_Std', 'Sharpness_Var', 'Vessel_BG_Contrast']].mean(),
        'Test Set': df_test[['Contrast_Std', 'Sharpness_Var', 'Vessel_BG_Contrast']].mean()
    })
    comparison['Difference (%)'] = ((comparison['Test Set'] - comparison['Training Set']) 
                                    / comparison['Training Set'] * 100)
    print(comparison)
    
    # 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['Contrast_Std', 'Sharpness_Var', 'Vessel_BG_Contrast']
    titles = ['Contrast (Std)', 'Sharpness (Var)', 'Vessel-BG Contrast']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        axes[idx].boxplot([df_train[metric], df_test[metric]], 
                          labels=['Training Set', 'Test Set'])
        axes[idx].set_title(title, fontweight='bold')
        axes[idx].set_ylabel('Value')
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(project_root, 'output', 'visualizations', 
                             'train_vs_test_quality_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ 对比图已保存: {save_path}")
    
    # 关键发现总结
    print("\n" + "="*80)
    print("关键发现:")
    print("="*80)
    
    if comparison.loc['Vessel_BG_Contrast', 'Difference (%)'] > 5:
        print("✓ 测试集的血管-背景对比度显著高于训练集")
        print("  → 这解释了为什么CLAHE增强在测试集上收益有限")
        print("  → 测试集图像原本质量就较好，增强的必要性降低")
    elif comparison.loc['Vessel_BG_Contrast', 'Difference (%)'] < -5:
        print("✓ 训练集的血管-背景对比度显著高于测试集")
        print("  → 训练集质量较好，模型可能未充分学习到低质量场景")
    else:
        print("✓ 训练集和测试集质量相近")
        print("  → 说明数据集整体质量较高，增强收益有限")
    
    return comparison

def analyze_error_patterns():
    """分析三个模型的错误模式"""
    
    print("\n" + "="*80)
    print("错误模式分析")
    print("="*80)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_dir = os.path.join(project_root, 'output', 'metrics')
    
    df_eval = pd.read_csv(os.path.join(metrics_dir, 'final_evaluation_metrics.csv'))
    
    # 计算每个模型的Sensitivity vs Specificity权衡
    print("\n=== Sensitivity vs Specificity 分析 ===")
    
    for model in df_eval['Model'].unique():
        df_model = df_eval[df_eval['Model'] == model]
        sens = df_model['Sensitivity'].mean()
        spec = df_model['Specificity'].mean()
        
        print(f"\n{model}:")
        print(f"  Sensitivity (血管召回率): {sens:.6f}")
        print(f"  Specificity (背景正确率): {spec:.6f}")
        
        if sens > 0.999 and spec < 0.994:
            print(f"  → 倾向于'宁可错杀'（高召回，低特异性）")
        elif spec > 0.994 and sens < 0.999:
            print(f"  → 倾向于'宁可漏检'（高特异性，低召回）")
        else:
            print(f"  → 平衡良好")
    
    # 可视化
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model in df_eval['Model'].unique():
        df_model = df_eval[df_eval['Model'] == model]
        ax.scatter(df_model['Specificity'], df_model['Sensitivity'], 
                  label=model, alpha=0.6, s=100)
    
    ax.set_xlabel('Specificity (Background Accuracy)', fontsize=12)
    ax.set_ylabel('Sensitivity (Vessel Recall)', fontsize=12)
    ax.set_title('Sensitivity vs Specificity Trade-off', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    save_path = os.path.join(project_root, 'output', 'visualizations', 
                             'sensitivity_specificity_tradeoff.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ 权衡图已保存: {save_path}")

if __name__ == '__main__':
    # 运行对比分析
    comparison = analyze_train_vs_test_quality()
    
    # 运行错误模式分析
    analyze_error_patterns()
    
    print("\n" + "="*80)
    print("补充分析完成！")
    print("="*80)
    print("\n这些深度分析将大幅提升你报告的学术价值。")
