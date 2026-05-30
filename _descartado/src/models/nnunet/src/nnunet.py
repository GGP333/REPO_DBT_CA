# src/nnunet.py
"""
nnU-Net 3D con soporte para auto-configuración:
  - Strides anisótropos por nivel (decididos por el planner)
  - Canales variables por nivel
  - Factory method from_plan() para construcción automática
  - Backward-compatible con la interfaz anterior (parámetros fijos)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------- Utilidades de stride/kernel ---------- #

def _to_tuple3(val: Union[int, Tuple[int, int, int], List[int]]) -> Tuple[int, int, int]:
    """Normaliza a tupla (d, h, w)."""
    if isinstance(val, int):
        return (val, val, val)
    return tuple(val)  # type: ignore


def _compute_padding(kernel_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Padding same para un kernel dado."""
    return tuple(k // 2 for k in kernel_size)  # type: ignore


# ---------- Bloques ---------- #

class nnUNetConvBlock(nn.Module):
    """
    Bloque de convolución de nnUNet con:
    - Instance Normalization
    - Leaky ReLU
    - Soporte para stride/kernel anisótropo
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
    ):
        super().__init__()
        ks = _to_tuple3(kernel_size)
        st = _to_tuple3(stride)
        pad = _compute_padding(ks)

        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=ks,
                               stride=st, padding=pad, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=ks,
                               stride=1, padding=pad, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.act2(self.norm2(self.conv2(out)))
        return out


class nnUNetResidualBlock(nn.Module):
    """Bloque residual con conexión de salto, stride anisótropo."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
    ):
        super().__init__()
        st = _to_tuple3(stride)

        self.conv_block = nnUNetConvBlock(in_ch, out_ch, kernel_size=kernel_size, stride=stride)

        # Conexión residual
        if in_ch != out_ch or any(s != 1 for s in st):
            self.skip_connection = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=st, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
            )
        else:
            self.skip_connection = nn.Identity()

        self.final_act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip_connection(x)
        out = self.conv_block(x)
        out = self.final_act(out + residual)
        return out


class nnUNetEncoder(nn.Module):
    """Encoder con strides/canales configurables por nivel."""

    def __init__(
        self,
        in_ch: int,
        channels: List[int],
        strides: List[List[int]],
        kernel_sizes: Optional[List[List[int]]] = None,
        use_checkpointing: bool = False,
    ):
        """
        Parameters
        ----------
        in_ch : int
            Canales de entrada (1 para mono-canal).
        channels : list[int]
            Canales por nivel: [ch_level0, ch_level1, ...]. len = num_levels.
        strides : list[list[int]]
            Strides de downsampling: len = num_levels - 1. Cada uno es [sz,sy,sx].
        kernel_sizes : list[list[int]], optional
            Kernels por nivel. Si None, usa 3x3x3.
        use_checkpointing : bool
        """
        super().__init__()
        self.use_checkpointing = use_checkpointing
        num_levels = len(channels)

        if kernel_sizes is None:
            kernel_sizes = [[3, 3, 3]] * num_levels

        # Primer bloque (sin stride, resolución completa)
        self.initial_conv = nnUNetResidualBlock(
            in_ch, channels[0], kernel_size=tuple(kernel_sizes[0]), stride=1
        )

        # Bloques del encoder con downsampling
        self.encoder_blocks = nn.ModuleList()
        for i in range(num_levels - 1):
            self.encoder_blocks.append(
                nnUNetResidualBlock(
                    channels[i],
                    channels[i + 1],
                    kernel_size=tuple(kernel_sizes[i + 1]),
                    stride=tuple(strides[i]),
                )
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []

        if self.use_checkpointing and self.training:
            out = checkpoint(self.initial_conv, x, use_reentrant=False)
        else:
            out = self.initial_conv(x)
        features.append(out)

        for block in self.encoder_blocks:
            if self.use_checkpointing and self.training:
                out = checkpoint(block, out, use_reentrant=False)
            else:
                out = block(out)
            features.append(out)

        return features


class nnUNetDecoder(nn.Module):
    """Decoder con upsampling transpuesto que respeta strides anisótropos."""

    def __init__(
        self,
        channels: List[int],
        strides: List[List[int]],
        kernel_sizes: Optional[List[List[int]]] = None,
        num_classes: int = 1,
        use_checkpointing: bool = False,
    ):
        """
        Parameters
        ----------
        channels : list[int]
            Canales del encoder, del nivel 0 (más alto) al bottleneck.
        strides : list[list[int]]
            Strides usados en el encoder (se invierten para upsample).
        """
        super().__init__()
        self.use_checkpointing = use_checkpointing
        num_levels = len(channels)

        if kernel_sizes is None:
            kernel_sizes = [[3, 3, 3]] * num_levels

        # Invertir: del bottleneck hacia arriba
        decoder_channels = list(reversed(channels[:-1]))
        reversed_strides = list(reversed(strides))

        self.upconv_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for i in range(len(decoder_channels)):
            in_ch = channels[-(i + 1)]   # canal del nivel actual (bottleneck primero)
            skip_ch = channels[-(i + 2)]  # canal del skip correspondiente
            out_ch = decoder_channels[i]
            st = tuple(reversed_strides[i])

            # Upsampling transpuesto con kernel = stride (como en nnU-Net original)
            self.upconv_layers.append(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=st, stride=st, bias=False)
            )

            # Bloque post-concatenación
            combined_ch = out_ch + skip_ch
            ks = tuple(kernel_sizes[len(decoder_channels) - 1 - i])
            self.decoder_blocks.append(
                nnUNetResidualBlock(combined_ch, out_ch, kernel_size=ks)
            )

        # Capa de salida
        self.final_conv = nn.Conv3d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        features = list(reversed(encoder_features))
        out = features[0]  # bottleneck

        for i, (upconv, decoder_block) in enumerate(
            zip(self.upconv_layers, self.decoder_blocks)
        ):
            out = upconv(out)

            if i < len(features) - 1:
                skip = features[i + 1]
                # Ajustar tamaños si difieren (por redondeos impares)
                if out.shape[2:] != skip.shape[2:]:
                    out = F.interpolate(
                        out, size=skip.shape[2:], mode="trilinear", align_corners=False
                    )
                out = torch.cat([out, skip], dim=1)

            if self.use_checkpointing and self.training:
                out = checkpoint(decoder_block, out, use_reentrant=False)
            else:
                out = decoder_block(out)

        logits = self.final_conv(out)
        return logits


class nnUNet3D(nn.Module):
    """
    nnU-Net 3D auto-configurable.

    Modos de construcción:
      1) nnUNet3D.from_plan(plan)         -- auto-configurado desde planner
      2) nnUNet3D(in_ch=1, base_ch=32, levels=4)  -- legacy (fijo)
    """

    def __init__(
        self,
        in_ch: int = 1,
        num_classes: int = 1,
        base_ch: int = 32,
        levels: int = 5,
        channels: Optional[List[int]] = None,
        strides: Optional[List[List[int]]] = None,
        kernel_sizes: Optional[List[List[int]]] = None,
        deep_supervision: bool = False,
        use_checkpointing: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.use_checkpointing = use_checkpointing

        # ---- Resolver canales y strides ---- #
        if channels is not None:
            # Modo auto-configurado
            self.encoder_channels = list(channels)
        else:
            # Modo legacy: canales fijos duplicando
            self.encoder_channels = [base_ch * (2 ** i) for i in range(levels)]

        if strides is not None:
            self.strides = [list(s) for s in strides]
        else:
            # Modo legacy: stride=2 isotrópico en todos los niveles
            self.strides = [[2, 2, 2]] * (len(self.encoder_channels) - 1)

        if kernel_sizes is not None:
            self.kernel_sizes = [list(k) for k in kernel_sizes]
        else:
            self.kernel_sizes = [[3, 3, 3]] * len(self.encoder_channels)

        # Validar consistencia
        assert len(self.strides) == len(self.encoder_channels) - 1, (
            f"strides ({len(self.strides)}) debe ser levels-1 ({len(self.encoder_channels)-1})"
        )

        # Encoder
        self.encoder = nnUNetEncoder(
            in_ch,
            self.encoder_channels,
            self.strides,
            self.kernel_sizes,
            use_checkpointing=use_checkpointing,
        )

        # Decoder
        self.decoder = nnUNetDecoder(
            self.encoder_channels,
            self.strides,
            self.kernel_sizes,
            num_classes,
            use_checkpointing=use_checkpointing,
        )

        # Deep supervision
        if deep_supervision:
            self.deep_supervision_heads = nn.ModuleList()
            for i in range(1, len(self.encoder_channels)):
                ch = self.encoder_channels[i]
                self.deep_supervision_heads.append(
                    nn.Conv3d(ch, num_classes, kernel_size=1)
                )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        encoder_features = self.encoder(x)
        logits = self.decoder(encoder_features)
        probs = torch.sigmoid(logits)

        if self.deep_supervision and self.training:
            deep_outputs = []
            for i, head in enumerate(self.deep_supervision_heads):
                ds_logits = head(encoder_features[i + 1])
                ds_logits = F.interpolate(
                    ds_logits, size=x.shape[2:], mode="trilinear", align_corners=False
                )
                deep_outputs.append(torch.sigmoid(ds_logits))
            return probs, deep_outputs

        return probs

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ---- Factory methods ---- #

    @classmethod
    def from_plan(cls, plan: Dict[str, Any], use_checkpointing: bool = True) -> "nnUNet3D":
        """
        Construye un nnUNet3D a partir de un plan generado por el planner.

        Parameters
        ----------
        plan : dict
            Salida de planner.generate_plan().
        use_checkpointing : bool
        """
        return cls(
            in_ch=plan.get("in_channels", 1),
            num_classes=plan.get("num_classes", 1),
            channels=plan["channels"],
            strides=plan["strides"],
            kernel_sizes=plan.get("kernel_sizes"),
            deep_supervision=False,
            use_checkpointing=use_checkpointing,
        )


# ---- Funciones de compatibilidad ---- #

def create_nnunet_3d(
    in_ch: int = 1,
    num_classes: int = 1,
    base_ch: int = 32,
    levels: int = 5,
    deep_supervision: bool = False,
    use_checkpointing: bool = True,
) -> nnUNet3D:
    """Factory function legacy."""
    return nnUNet3D(
        in_ch=in_ch,
        num_classes=num_classes,
        base_ch=base_ch,
        levels=levels,
        deep_supervision=deep_supervision,
        use_checkpointing=use_checkpointing,
    )


def nnUNet3D_legacy(
    in_ch: int = 1, base_ch: int = 32, levels: int = 4, use_checkpointing: bool = True
) -> nnUNet3D:
    """Compatibilidad con la interfaz anterior."""
    return nnUNet3D(
        in_ch=in_ch,
        num_classes=1,
        base_ch=base_ch,
        levels=levels,
        deep_supervision=False,
        use_checkpointing=use_checkpointing,
    )
