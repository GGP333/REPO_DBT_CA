# src/attention_unet3d.py
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class ConvBlock3D(nn.Module):
    """3D Convolutional block with BatchNorm and ReLU activation"""
    def __init__(self, in_ch: int, out_ch: int, dropout_rate: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):  # (B,C,Z,H,W)
        return self.conv(x)

class AttentionGate3D(nn.Module):
    """3D Attention Gate for focusing on relevant features"""
    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.gate_conv = nn.Conv3d(gate_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.skip_conv = nn.Conv3d(skip_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.attention_conv = nn.Conv3d(inter_ch, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, gate, skip):
        """
        gate: gating signal from coarser scale (B, gate_ch, Z_g, H_g, W_g)
        skip: skip connection from encoder (B, skip_ch, Z_s, H_s, W_s)
        """
        # Resize gate to match skip dimensions
        gate_resized = F.interpolate(gate, size=skip.shape[2:], mode='trilinear', align_corners=False)
        
        # Apply convolutions
        gate_conv = self.gate_conv(gate_resized)
        skip_conv = self.skip_conv(skip)
        
        # Combine and generate attention coefficients
        combine = self.relu(gate_conv + skip_conv)
        attention = self.sigmoid(self.attention_conv(combine))
        
        # Apply attention to skip connection
        attended_skip = skip * attention
        
        return attended_skip, attention

class SpatialAttention3D(nn.Module):
    """3D Spatial Attention Module for DBT volumes"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels // 8, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate spatial attention map
        attn = self.conv1(x)
        attn = F.relu(attn)
        attn = self.conv2(attn)
        attn = self.sigmoid(attn)
        
        # Apply attention
        return x * attn

class ChannelAttention3D(nn.Module):
    """3D Channel Attention Module for feature recalibration"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class CBAM3D(nn.Module):
    """3D Convolutional Block Attention Module"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention3D(in_channels, reduction)
        self.spatial_attention = SpatialAttention3D(in_channels)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class DBTSpecificBlock3D(nn.Module):
    """DBT-specific processing block with multi-scale features"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # Multi-scale convolutions for different tissue patterns
        self.conv1x1 = nn.Conv3d(in_ch, out_ch // 4, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv3d(in_ch, out_ch // 2, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv3d(in_ch, out_ch // 4, kernel_size=5, padding=2)
        
        self.bn = nn.InstanceNorm3d(out_ch, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM3D(out_ch)
        
    def forward(self, x):
        out1 = self.conv1x1(x)
        out2 = self.conv3x3(x)
        out3 = self.conv5x5(x)
        
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        out = self.cbam(out)
        
        return out

def center_crop_3d(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Center crop src to match target spatial dimensions"""
    _, _, Z, H, W = target.shape
    _, _, sz, sh, sw = src.shape
    cz, ch, cw = (sz - Z) // 2, (sh - H) // 2, (sw - W) // 2
    return src[:, :, cz:cz+Z, ch:ch+H, cw:cw+W]

class AttentionUNet3D(nn.Module):
    """
    3D Attention U-Net specifically designed for DBT (Digital Breast Tomosynthesis) volumes.
    Optimized for 38x256x512 dimensional volumes with attention mechanisms for better
    tissue pattern recognition and lesion localization.
    
    Args:
        in_ch: Número de canales de entrada
        base_ch: Número base de canales (se duplica por nivel)
        levels: Número de niveles en el encoder/decoder
        dropout_rate: Tasa de dropout
        use_checkpointing: Si True, usa gradient checkpointing para reducir memoria
    """
    def __init__(self, in_ch: int = 1, base_ch: int = 32, levels: int = 4, dropout_rate: float = 0.1, use_checkpointing: bool = True):
        super().__init__()
        
        # Calculate channel progression optimized for DBT
        chs = [base_ch * (2**i) for i in range(levels)]
        
        # Encoder path with DBT-specific blocks
        self.down_blocks = nn.ModuleList()
        self.dbt_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        prev = in_ch
        for i, c in enumerate(chs):
            # Standard conv block
            self.down_blocks.append(ConvBlock3D(prev, c, dropout_rate))
            # DBT-specific multi-scale block
            self.dbt_blocks.append(DBTSpecificBlock3D(c, c))
            # Adaptive pooling for different scales
            if i < len(chs) - 1:  # No pooling after last encoder block
                self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            prev = c
        
        # Bottleneck with enhanced feature processing
        self.bottleneck = nn.Sequential(
            ConvBlock3D(chs[-1], chs[-1] * 2, dropout_rate),
            CBAM3D(chs[-1] * 2),
            ConvBlock3D(chs[-1] * 2, chs[-1] * 2, dropout_rate)
        )
        
        # Decoder path with attention gates
        self.up_convs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        prev = chs[-1] * 2
        num_upsamples = len(chs) - 1  # match number of pools
        skip_chs = list(reversed(chs[:-1]))  # [128, 64, 32] for chs=[32,64,128,256]
        
        # Create per-stage modules (one per upsample step)
        for i, c_skip in enumerate(skip_chs):
            # Upsample op for this stage
            self.up_convs.append(nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))
            # Attention gate (gate signal: prev, skip: c_skip)
            self.attention_gates.append(AttentionGate3D(
                gate_ch=prev,
                skip_ch=c_skip,
                inter_ch=max(1, c_skip // 2)
            ))
            
            # Decoder block at current resolution (concat skip + gate)
            self.up_blocks.append(ConvBlock3D(prev + c_skip, c_skip, dropout_rate))
            
            prev = c_skip
        
        # Multi-scale output processing for DBT (input channels = last decoder c)
        final_in_ch = prev
        self.final_processing = nn.Sequential(
            ConvBlock3D(final_in_ch, final_in_ch // 2, dropout_rate),
            CBAM3D(final_in_ch // 2),
            nn.Conv3d(final_in_ch // 2, 1, kernel_size=1),
        )
        
        # Deep supervision heads (optional, can be used during training)
        self.deep_supervision = nn.ModuleList([nn.Conv3d(c, 1, kernel_size=1) for c in skip_chs])
        
        # flag for checkpointing - activado por defecto para reducir memoria
        self.use_gradient_checkpointing = use_checkpointing
    
    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.use_gradient_checkpointing = bool(enabled)
    
    def forward(self, x, return_deep_supervision=False):
        """
        Forward pass through Attention U-Net
        Args:
            x: Input volume (B, 1, Z, H, W) - typically (B, 1, 38, 256, 512) for DBT
            return_deep_supervision: Whether to return intermediate predictions
        """
        use_ckpt = bool(getattr(self, "use_gradient_checkpointing", False))
        # Store skip connections and their enhanced versions
        skips = []
        enhanced_skips = []
        
        # Encoder path
        out = x
        for i, (down_block, dbt_block, pool) in enumerate(zip(self.down_blocks, self.dbt_blocks, self.pools + [None])):
            # Standard convolution
            out = checkpoint(down_block, out) if use_ckpt else down_block(out)
            # DBT-specific enhancement
            enhanced = checkpoint(dbt_block, out) if use_ckpt else dbt_block(out)
            
            skips.append(out)
            enhanced_skips.append(enhanced)
            
            if pool is not None:
                out = pool(out)
        
        # Bottleneck
        if use_ckpt:
            for layer in self.bottleneck:
                out = checkpoint(layer, out)
        else:
            out = self.bottleneck(out)
        
        # Decoder path with attention
        deep_outputs = []
        
        for i in range(len(self.up_blocks)):
            # Upsample to the next finer scale before fusing with the skip connection
            out = self.up_convs[i](out)

            # Select corresponding skip (after moving one scale up)
            # Mapping for levels=4: i=0->skip_idx=2, i=1->1, i=2->0
            skip_idx = len(enhanced_skips) - 2 - i
            skip = enhanced_skips[skip_idx]

            # Apply attention gate at the same spatial resolution
            attended_skip, attention_weights = self.attention_gates[i](out, skip)

            # Ensure exact spatial match if any small mismatch remains
            if out.shape[-3:] != attended_skip.shape[-3:]:
                if all(o >= s for o, s in zip(out.shape[-3:], attended_skip.shape[-3:])):
                    out = center_crop_3d(out, attended_skip)
                else:
                    attended_skip = center_crop_3d(attended_skip, out)

            # Fuse and process
            out = torch.cat([attended_skip, out], dim=1)
            out = checkpoint(self.up_blocks[i], out) if use_ckpt else self.up_blocks[i](out)

            # Deep supervision output at this scale
            if return_deep_supervision and i < len(self.deep_supervision):
                deep_out = torch.sigmoid(self.deep_supervision[i](out))
                deep_outputs.append(deep_out)
         
        # Final output processing
        if use_ckpt:
            tmp = out
            for layer in self.final_processing:
                tmp = checkpoint(layer, tmp)
            logits = tmp
        else:
            logits = self.final_processing(out)
        probs = torch.sigmoid(logits)
        
        if return_deep_supervision:
            return probs, deep_outputs
        else:
            return probs  # (B, 1, Z, H, W)
    
    def get_attention_maps(self, x):
        """Extract attention maps for visualization"""
        attention_maps = []
        
        # Similar to forward but collect attention weights
        skips = []
        enhanced_skips = []
        
        out = x
        for i, (down_block, dbt_block, pool) in enumerate(zip(self.down_blocks, self.dbt_blocks, self.pools + [None])):
            out = down_block(out)
            enhanced = dbt_block(out)
            skips.append(out)
            enhanced_skips.append(enhanced)
            if pool is not None:
                out = pool(out)
         
        out = self.bottleneck(out)
         
        for i in range(len(self.up_blocks)):
            # Move to the next finer scale
            out = self.up_convs[i](out)

            # Choose matching skip (see mapping in forward)
            skip_idx = len(enhanced_skips) - 2 - i
            skip = enhanced_skips[skip_idx]

            attended_skip, attention_weights = self.attention_gates[i](out, skip)
            attention_maps.append(attention_weights)

            if out.shape[-3:] != attended_skip.shape[-3:]:
                if all(o >= s for o, s in zip(out.shape[-3:], attended_skip.shape[-3:])):
                    out = center_crop_3d(out, attended_skip)
                else:
                    attended_skip = center_crop_3d(attended_skip, out)

            out = torch.cat([attended_skip, out], dim=1)
            out = self.up_blocks[i](out)
         
        return attention_maps 