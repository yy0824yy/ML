"""
第二步：模型定义
包含：U-Net、Attention U-Net等模型架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """双卷积块 (Conv -> ReLU -> Conv -> ReLU)"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样块 (MaxPool -> DoubleConv)"""
    
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样块 (ConvTranspose -> DoubleConv)"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理尺寸差异
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    """注意力机制模块"""
    
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: 门控信号的通道数 (来自解码器高层)
            F_l: 跳接信号的通道数 (来自编码器)
            F_int: 中间特征通道数
        """
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: 门控信号 (来自解码器)
            x: 跳接特征 (来自编码器)
        Returns:
            注意力加权的特征
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # 上采样g1到x1的尺寸
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNet(nn.Module):
    """标准U-Net模型"""
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        self.up1 = Up(features[3] + features[2], features[2], bilinear=True)
        self.up2 = Up(features[2] + features[1], features[1], bilinear=True)
        self.up3 = Up(features[1] + features[0], features[0], bilinear=True)
        
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        logits = self.outc(x)
        return logits


class AttentionUNet(nn.Module):
    """带注意力机制的U-Net模型 (Attention U-Net)"""
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(AttentionUNet, self).__init__()
        
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        
        # 注意力门控
        self.attention3 = AttentionGate(features[3], features[2], features[3] // 2)
        self.attention2 = AttentionGate(features[2], features[1], features[2] // 2)
        self.attention1 = AttentionGate(features[1], features[0], features[1] // 2)
        
        self.up1 = Up(features[3] + features[2], features[2], bilinear=True)
        self.up2 = Up(features[2] + features[1], features[1], bilinear=True)
        self.up3 = Up(features[1] + features[0], features[0], bilinear=True)
        
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 使用注意力门控加权跳接
        x3_att = self.attention3(x4, x3)
        x = self.up1(x4, x3_att)
        
        x2_att = self.attention2(x, x2)
        x = self.up2(x, x2_att)
        
        x1_att = self.attention1(x, x1)
        x = self.up3(x, x1_att)
        
        logits = self.outc(x)
        return logits


class DiceLoss(nn.Module):
    """Dice损失函数 (适合医学图像分割)"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """BCE + Dice混合损失"""
    
    def __init__(self, smooth=1.0, weight_bce=0.5, weight_dice=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
    
    def forward(self, pred, target):
        # 转换为概率
        pred_sigmoid = torch.sigmoid(pred)
        
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred_sigmoid, target)
        
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss


# ==================== TransUNet Components ====================

class PatchEmbed(nn.Module):
    """将图像分割为patches并进行嵌入"""
    
    def __init__(self, img_size=512, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class TransUNet(nn.Module):
    """
    简化版TransUNet：CNN编码器 + Transformer中间层 + CNN解码器
    适合小数据集，不依赖预训练权重
    """
    
    def __init__(self, in_channels=3, out_channels=1, img_size=512, 
                 patch_size=16, embed_dim=512, depth=6, num_heads=8):
        super(TransUNet, self).__init__()
        
        # CNN编码器（类似U-Net的下采样部分）
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = Down(64, 128)
        self.enc3 = Down(128, 256)
        self.enc4 = Down(256, 512)
        
        # Transformer瓶颈层
        # 将CNN特征转为patches
        self.patch_size = patch_size
        bottleneck_size = img_size // 16  # 经过4次下采样后：512 -> 32
        
        # 将512通道的32x32特征图转为transformer输入
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(512, embed_dim, kernel_size=1),
            nn.Flatten(2),
        )
        
        # Transformer blocks
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1) 
              for _ in range(depth)]
        )
        
        # 从transformer输出重建空间特征
        self.from_transformer = nn.Sequential(
            nn.Linear(embed_dim, 512),
        )
        
        # CNN解码器（类似U-Net的上采样部分）
        self.up1 = Up(512 + 256, 256, bilinear=True)
        self.up2 = Up(256 + 128, 128, bilinear=True)
        self.up3 = Up(128 + 64, 64, bilinear=True)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.bottleneck_size = bottleneck_size
    
    def forward(self, x):
        # CNN编码器
        x1 = self.enc1(x)      # 64, 512, 512
        x2 = self.enc2(x1)     # 128, 256, 256
        x3 = self.enc3(x2)     # 256, 128, 128
        x4 = self.enc4(x3)     # 512, 32, 32
        
        # 转为transformer输入
        b, c, h, w = x4.shape
        x_trans = self.to_patch_embedding(x4)  # (B, embed_dim, h*w)
        x_trans = x_trans.transpose(1, 2)      # (B, h*w, embed_dim)
        
        # Transformer处理
        x_trans = self.transformer(x_trans)    # (B, h*w, embed_dim)
        
        # 重建空间维度
        x_trans = self.from_transformer(x_trans)  # (B, h*w, 512)
        x_trans = x_trans.transpose(1, 2)          # (B, 512, h*w)
        x_trans = x_trans.reshape(b, 512, h, w)   # (B, 512, h, w)
        
        # CNN解码器
        x = self.up1(x_trans, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建样本输入
    sample_input = torch.randn(1, 3, 512, 512).to(device)
    
    # 测试U-Net
    print("Testing U-Net...")
    model_unet = UNet(in_channels=3, out_channels=1).to(device)
    output_unet = model_unet(sample_input)
    print(f"U-Net output shape: {output_unet.shape}")
    
    # 测试Attention U-Net
    print("\nTesting Attention U-Net...")
    model_att_unet = AttentionUNet(in_channels=3, out_channels=1).to(device)
    output_att_unet = model_att_unet(sample_input)
    print(f"Attention U-Net output shape: {output_att_unet.shape}")
    
    # 测试TransU-Net
    print("\nTesting TransUNet...")
    model_trans_unet = TransUNet(in_channels=3, out_channels=1).to(device)
    output_trans_unet = model_trans_unet(sample_input)
    print(f"TransUNet output shape: {output_trans_unet.shape}")
    
    # 统计参数数量
    print("\n" + "="*50)
    print("Model Parameters:")
    print("="*50)
    print(f"U-Net: {sum(p.numel() for p in model_unet.parameters()):,} parameters")
    print(f"Attention U-Net: {sum(p.numel() for p in model_att_unet.parameters()):,} parameters")
    print(f"TransUNet: {sum(p.numel() for p in model_trans_unet.parameters()):,} parameters")
    print("\n" + "="*50)
    print("Model Parameters:")
    print("="*50)
    print(f"U-Net: {sum(p.numel() for p in model_unet.parameters()):,} parameters")
    print(f"Attention U-Net: {sum(p.numel() for p in model_att_unet.parameters()):,} parameters")
