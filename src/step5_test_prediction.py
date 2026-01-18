"""
方案B：测试集预测生成

虽然测试集无ground truth标注无法评估性能，但可以：
1. 使用最佳模型对测试集生成预测
2. 可视化预测结果用于报告展示
3. 分析不同模型在测试集上的预测差异
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))
from step2_models import UNet, AttentionUNet, TransUNet
from utils import DataLoader


class TestPredictor:
    """测试集预测器"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def predict_single(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        对单张图像进行预测
        
        Args:
            image: (H, W) 灰度图或 (H, W, 3) RGB图
            threshold: 二值化阈值
        
        Returns:
            预测掩码 (H, W) 二值图
        """
        # 预处理
        if len(image.shape) == 2:
            # 灰度图转3通道
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 归一化到[0,1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # 转为Tensor (H, W, C) -> (1, C, H, W)
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        img_tensor = img_tensor.to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(img_tensor)  # (1, 1, H, W)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()  # (H, W)
        
        # 二值化
        mask_pred = (prob > threshold).astype(np.uint8) * 255
        
        return mask_pred, prob
    
    def predict_batch(self, images: list, threshold: float = 0.5):
        """批量预测"""
        predictions = []
        probabilities = []
        
        for img in tqdm(images, desc="Predicting"):
            mask_pred, prob = self.predict_single(img, threshold)
            predictions.append(mask_pred)
            probabilities.append(prob)
        
        return predictions, probabilities


def visualize_prediction(image, prediction, probability=None, save_path=None):
    """
    可视化预测结果
    
    Args:
        image: 原始图像 (H, W) 或 (H, W, 3)
        prediction: 预测掩码 (H, W) 二值图
        probability: 预测概率图 (H, W) [0-1]
        save_path: 保存路径
    """
    if probability is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 原图
    if len(image.shape) == 2:
        axes[0].imshow(image, cmap='gray')
    else:
        axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 预测掩码
    axes[1].imshow(prediction, cmap='gray')
    axes[1].set_title('Predicted Vessel Mask', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 概率热图
    if probability is not None:
        im = axes[2].imshow(probability, cmap='hot', vmin=0, vmax=1)
        axes[2].set_title('Prediction Probability Heatmap', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_comparison(image, pred_baseline, pred_enhanced, pred_attunet, pred_transunet, save_path=None):
    """
    对比4个模型的预测结果
    
    Args:
        image: 原始图像
        pred_baseline: Baseline预测
        pred_enhanced: Enhanced UNet预测
        pred_attunet: AttentionUNet预测
        pred_transunet: TransUNet预测
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 原图
    if len(image.shape) == 2:
        axes[0, 0].imshow(image, cmap='gray')
    else:
        axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Baseline
    axes[0, 1].imshow(pred_baseline, cmap='gray')
    axes[0, 1].set_title('Baseline (Raw + UNet)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Enhanced UNet
    axes[0, 2].imshow(pred_enhanced, cmap='gray')
    axes[0, 2].set_title('Enhanced + UNet', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # AttentionUNet
    axes[1, 0].imshow(pred_attunet, cmap='gray')
    axes[1, 0].set_title('Enhanced + AttentionUNet (Best)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # TransUNet
    axes[1, 1].imshow(pred_transunet, cmap='gray')
    axes[1, 1].set_title('Enhanced + TransUNet', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # 隐藏最后一个子图
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    print("="*80)
    print("方案B：测试集预测生成")
    print("="*80)
    
    # 路径设置
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'datasets')
    output_dir = os.path.join(project_root, 'output')
    model_dir = os.path.join(output_dir, 'models')
    pred_dir = os.path.join(output_dir, 'test_predictions')
    viz_dir = os.path.join(output_dir, 'visualizations', 'test_predictions')
    
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 1. 加载测试集
    print("\n[1/4] 加载测试集...")
    loader = DataLoader(data_dir)
    test_images_raw, _ = loader.get_test_samples()
    
    # 加载增强后的测试集
    enhanced_test_dir = os.path.join(output_dir, 'enhanced_images', 'test')
    test_images_enhanced = []
    for i in range(1, 21):  # 测试集01-20
        path = os.path.join(enhanced_test_dir, f'{i:02d}_enhanced.png')
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            test_images_enhanced.append(img)
        else:
            print(f"警告: 增强图像不存在 {path}")
            test_images_enhanced.append(test_images_raw[i-1])
    
    print(f"✓ 加载了 {len(test_images_raw)} 张测试图像")
    
    # 2. 加载训练好的模型 (选择每个实验中fold1的模型作为代表)
    print("\n[2/4] 加载训练好的模型...")
    
    models = {}
    
    # Baseline (Raw + UNet) - Fold 1
    baseline_path = os.path.join(model_dir, 'Exp1_Baseline_KFold_fold1.pth')
    if os.path.exists(baseline_path):
        model_baseline = UNet(in_channels=3, out_channels=1).to(device)
        model_baseline.load_state_dict(torch.load(baseline_path, map_location=device))
        models['Baseline'] = TestPredictor(model_baseline, device)
        print(f"✓ Baseline模型加载成功")
    else:
        print(f"⚠ 未找到Baseline模型: {baseline_path}")
    
    # Enhanced + UNet - Fold 1
    enhanced_path = os.path.join(model_dir, 'Exp2_Enhanced_KFold_fold1.pth')
    if os.path.exists(enhanced_path):
        model_enhanced = UNet(in_channels=3, out_channels=1).to(device)
        model_enhanced.load_state_dict(torch.load(enhanced_path, map_location=device))
        models['Enhanced'] = TestPredictor(model_enhanced, device)
        print(f"✓ Enhanced UNet模型加载成功")
    else:
        print(f"⚠ 未找到Enhanced模型: {enhanced_path}")
    
    # Enhanced + AttentionUNet - Fold 1 (最佳模型)
    attunet_path = os.path.join(model_dir, 'Exp3_AttUNet_KFold_fold1.pth')
    if os.path.exists(attunet_path):
        model_attunet = AttentionUNet(in_channels=3, out_channels=1).to(device)
        model_attunet.load_state_dict(torch.load(attunet_path, map_location=device))
        models['AttentionUNet'] = TestPredictor(model_attunet, device)
        print(f"✓ AttentionUNet模型加载成功")
    else:
        print(f"⚠ 未找到AttentionUNet模型: {attunet_path}")
    
    # Enhanced + TransUNet - Fold 1
    transunet_path = os.path.join(model_dir, 'Exp4_TransUNet_KFold_fold1.pth')
    if os.path.exists(transunet_path):
        model_transunet = TransUNet(in_channels=3, out_channels=1).to(device)
        model_transunet.load_state_dict(torch.load(transunet_path, map_location=device))
        models['TransUNet'] = TestPredictor(model_transunet, device)
        print(f"✓ TransUNet模型加载成功")
    else:
        print(f"⚠ 未找到TransUNet模型: {transunet_path}")
    
    # 3. 生成预测
    print("\n[3/4] 生成测试集预测...")
    
    predictions = {}
    probabilities = {}
    
    # Baseline使用原始图像
    if 'Baseline' in models:
        print("\n→ Baseline (Raw + UNet) 预测中...")
        preds, probs = models['Baseline'].predict_batch(test_images_raw)
        predictions['Baseline'] = preds
        probabilities['Baseline'] = probs
    
    # 其他模型使用增强图像
    for model_name in ['Enhanced', 'AttentionUNet', 'TransUNet']:
        if model_name in models:
            print(f"\n→ {model_name} 预测中...")
            preds, probs = models[model_name].predict_batch(test_images_enhanced)
            predictions[model_name] = preds
            probabilities[model_name] = probs
    
    # 4. 保存预测结果和可视化
    print("\n[4/4] 保存预测结果...")
    
    # 保存每个模型的预测掩码
    for model_name, preds in predictions.items():
        model_pred_dir = os.path.join(pred_dir, model_name)
        os.makedirs(model_pred_dir, exist_ok=True)
        
        for i, pred in enumerate(preds):
            save_path = os.path.join(model_pred_dir, f'test_{i+1:02d}_pred.png')
            cv2.imwrite(save_path, pred)
    
    print(f"✓ 所有预测掩码已保存到: {pred_dir}")
    
    # 生成可视化对比图（选择几张代表性图像）
    print("\n生成可视化对比图...")
    
    sample_indices = [0, 4, 9, 14, 19]  # 选择5张图像进行展示
    
    for idx in sample_indices:
        # 单个最佳模型的详细可视化
        if 'AttentionUNet' in predictions:
            viz_path = os.path.join(viz_dir, f'test_{idx+1:02d}_best_model.png')
            visualize_prediction(
                test_images_enhanced[idx],
                predictions['AttentionUNet'][idx],
                probabilities['AttentionUNet'][idx],
                save_path=viz_path
            )
        
        # 4个模型对比
        if all(m in predictions for m in ['Baseline', 'Enhanced', 'AttentionUNet', 'TransUNet']):
            viz_path = os.path.join(viz_dir, f'test_{idx+1:02d}_comparison.png')
            visualize_comparison(
                test_images_enhanced[idx],
                predictions['Baseline'][idx],
                predictions['Enhanced'][idx],
                predictions['AttentionUNet'][idx],
                predictions['TransUNet'][idx],
                save_path=viz_path
            )
    
    print(f"✓ 可视化图表已保存到: {viz_dir}")
    
    # 统计分析
    print("\n" + "="*80)
    print("预测统计分析")
    print("="*80)
    
    for model_name, preds in predictions.items():
        vessel_ratios = []
        for pred in preds:
            vessel_pixels = (pred > 0).sum()
            total_pixels = pred.size
            ratio = vessel_pixels / total_pixels * 100
            vessel_ratios.append(ratio)
        
        print(f"\n{model_name}:")
        print(f"  平均血管占比: {np.mean(vessel_ratios):.2f}% ± {np.std(vessel_ratios):.2f}%")
        print(f"  范围: [{np.min(vessel_ratios):.2f}%, {np.max(vessel_ratios):.2f}%]")
    
    print("\n" + "="*80)
    print("测试集预测完成！")
    print("="*80)
    print(f"预测掩码保存在: {pred_dir}")
    print(f"可视化图表保存在: {viz_dir}")
    print(f"\n可在报告中使用这些可视化结果展示模型效果。")


if __name__ == '__main__':
    main()
