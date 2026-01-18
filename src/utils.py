"""
工具函数模块
"""
import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List
import matplotlib.pyplot as plt


class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.train_img_dir = os.path.join(data_dir, 'training/images')
        self.train_mask_dir = os.path.join(data_dir, 'training/mask')
        self.train_manual_dir = os.path.join(data_dir, 'training/1st_manual')
        self.test_img_dir = os.path.join(data_dir, 'test/images')
        self.test_mask_dir = os.path.join(data_dir, 'test/mask')
    
    def load_image(self, path: str) -> np.ndarray:
        """加载图像，返回灰度或RGB"""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot load image: {path}")
        # 如果是TIF格式，可能需要特殊处理
        if path.endswith('.tif') or path.endswith('.tiff'):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def load_mask(self, path: str) -> np.ndarray:
        """加载掩码，转为二值图像"""
        # 对于GIF文件，使用PIL库
        if path.endswith('.gif') or path.endswith('.GIF'):
            from PIL import Image
            try:
                mask = Image.open(path).convert('L')
                mask = np.array(mask)
            except Exception as e:
                raise ValueError(f"Cannot load GIF mask: {path}, Error: {e}")
        else:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Cannot load mask: {path}")
        
        # 转为二值图像 (0 或 255)
        mask = (mask > 127).astype(np.uint8) * 255
        return mask
    
    def get_training_samples(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """获取所有训练样本"""
        images = []
        masks = []
        
        img_files = sorted([f for f in os.listdir(self.train_img_dir) if f.endswith('.tif')])
        
        for img_file in img_files:
            # 提取编号
            num = img_file.split('_')[0]
            
            img_path = os.path.join(self.train_img_dir, img_file)
            # 使用1st_manual作为真正的血管标注（而非FOV mask）
            mask_path = os.path.join(self.train_manual_dir, f"{num}_manual1.gif")
            
            if os.path.exists(mask_path):
                img = self.load_image(img_path)
                mask = self.load_mask(mask_path)
                images.append(img)
                masks.append(mask)
        
        return images, masks
    
    def get_test_samples(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """获取所有测试样本"""
        images = []
        masks = []
        
        img_files = sorted([f for f in os.listdir(self.test_img_dir) if f.endswith('.tif')])
        
        for img_file in img_files:
            # 提取编号
            num = img_file.split('_')[0]
            
            img_path = os.path.join(self.test_img_dir, img_file)
            mask_path = os.path.join(self.test_mask_dir, f"{num}_test_mask.gif")
            
            if os.path.exists(mask_path):
                img = self.load_image(img_path)
                mask = self.load_mask(mask_path)
                images.append(img)
                masks.append(mask)
        
        return images, masks


class ImageQualityAnalyzer:
    """图像质量分析器"""
    
    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        """计算对比度 (标准差)"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.std(image)
    
    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """计算清晰度 (Laplacian方差)"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return np.var(laplacian)
    
    @staticmethod
    def calculate_brightness(image: np.ndarray) -> float:
        """计算亮度 (平均像素值)"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.mean(image)
    
    @staticmethod
    def get_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
        """获取直方图"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
        return hist.flatten()
    
    @staticmethod
    def analyze_vessel_contrast(image: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float]:
        """分析血管与背景的灰度对比"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 血管像素
        vessel_pixels = image[mask > 127]
        # 背景像素
        bg_pixels = image[mask <= 127]
        
        vessel_mean = np.mean(vessel_pixels) if len(vessel_pixels) > 0 else 0
        bg_mean = np.mean(bg_pixels) if len(bg_pixels) > 0 else 0
        contrast = abs(vessel_mean - bg_mean)
        
        return vessel_mean, bg_mean, contrast


class ImageEnhancer:
    """图像增强器"""
    
    @staticmethod
    def clahe_enhance(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
        """
        CLAHE (限制对比度自适应直方图均衡化) 增强
        """
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(image_gray)
        
        return enhanced
    
    @staticmethod
    def gamma_correction(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """
        Gamma校正增强
        """
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
        
        # 构建查找表
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        
        enhanced = cv2.LUT(image_gray, table)
        return enhanced
    
    @staticmethod
    def histogram_equalization(image: np.ndarray) -> np.ndarray:
        """
        直方图均衡化
        """
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image
        
        enhanced = cv2.equalizeHist(image_gray)
        return enhanced


def plot_comparison(image1: np.ndarray, image2: np.ndarray, 
                   title1: str, title2: str, 
                   save_path: str = None):
    """绘制两张图的对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if len(image1.shape) == 3:
        axes[0].imshow(image1)
    else:
        axes[0].imshow(image1, cmap='gray')
    axes[0].set_title(title1, fontsize=12)
    axes[0].axis('off')
    
    if len(image2.shape) == 3:
        axes[1].imshow(image2)
    else:
        axes[1].imshow(image2, cmap='gray')
    axes[1].set_title(title2, fontsize=12)
    axes[1].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_histogram_comparison(hist1: np.ndarray, hist2: np.ndarray,
                             title1: str, title2: str,
                             save_path: str = None):
    """绘制两个直方图的对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(hist1, color='black', linewidth=1.5)
    axes[0].set_title(title1, fontsize=12)
    axes[0].set_xlabel('Pixel Value')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(hist2, color='blue', linewidth=1.5)
    axes[1].set_title(title2, fontsize=12)
    axes[1].set_xlabel('Pixel Value')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    pass
