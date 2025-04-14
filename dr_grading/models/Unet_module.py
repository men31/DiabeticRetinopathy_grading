from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# 1) Transformer Bottleneck with Down/Up Sampling
# -----------------------
class TransformerBottleneck(nn.Module):
    """
    A global-attention bottleneck that:
    - Optionally downsamples spatial dims before attention
    - Applies MHSA + feed-forward at lower resolution
    - Upsamples back to original "bottleneck" resolution
    """
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 2,
        ff_mult: int = 2,
        bottleneck_down: bool = True,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.bottleneck_down = bottleneck_down

        # Additional pooling to reduce H×W
        if self.bottleneck_down:
            self.down = nn.MaxPool2d(kernel_size=2, stride=2)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.down = nn.Identity()
            self.up = nn.Identity()

        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(in_channels)
        self.ff = nn.Sequential(
            nn.Linear(in_channels, in_channels * ff_mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * ff_mult, in_channels),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        x = self.down(x)  # reduce spatial if needed
        B, C, H, W = x.shape

        # Flatten for MHSA
        x_flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, HW, C)

        # Self-Attention
        attn_out, _ = self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))
        x_flat = x_flat + self.dropout1(attn_out)

        # Feed-forward
        ff_out = self.ff(self.norm2(x_flat))
        x_flat = x_flat + self.dropout2(ff_out)

        # Reshape
        x_out = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_out = self.up(x_out)  # restore spatial size
        return x_out

# -----------------------
# 2) Basic Conv Block
# -----------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),

            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

# -----------------------
# 3) The U-Net with Optimizations
# -----------------------
class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: Tuple[int, ...] = (64, 128),
        groups: int = 8,
        dropout: float = 0.1,
        use_transformer_bottleneck: bool = True,
        bottleneck_down: bool = True
    ) -> None:
        """
        Args:
            in_channels: number of channels in input images
            out_channels: number of channels in output segmentation
            features: channel sizes in encoder stages (reduced for speed)
            groups: group norm groups
            dropout: dropout for conv blocks
            use_transformer_bottleneck: if False, use a simpler CNN block
            bottleneck_down: extra downsampling in the bottleneck to reduce attention cost
        """
        super().__init__()

        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Build encoder
        for feature in features:
            self.encoder.append(ConvBlock(in_channels, feature, groups=groups))
            in_channels = feature

        # Bottleneck
        if use_transformer_bottleneck:
            # For large images, fewer heads & smaller channels => faster
            self.bottleneck = TransformerBottleneck(
                in_channels=features[-1],
                num_heads=2,        # fewer heads => faster
                ff_mult=2,          # smaller feed-forward => faster
                bottleneck_down=bottleneck_down,
                dropout=dropout
            )
        else:
            # simpler CNN bottleneck if you prefer
            self.bottleneck = ConvBlock(features[-1], features[-1], groups=groups, dropout=dropout)

        # Build decoder (reverse of features)
        self.decoder = nn.ModuleList()
        decoder_in_channels = [features[-1]] + list(features[::-1][:-1])
        decoder_skip_channels = features[::-1]

        for in_c, skip_c in zip(decoder_in_channels, decoder_skip_channels):
            self.decoder.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ConvBlock(in_channels=in_c + skip_c, out_channels=skip_c, groups=groups)
            ))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Encoder
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx, decoder_block in enumerate(self.decoder):
            x = decoder_block[0](x)  # upsample
            skip = skip_connections[idx]

            # fix shape mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

            x = torch.cat([skip, x], dim=1)
            x = decoder_block[1](x)

        return self.final_conv(x)
