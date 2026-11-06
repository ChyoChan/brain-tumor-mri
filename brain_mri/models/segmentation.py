"""U-Net, residual U-Net, and attention U-Net architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d → BN → ReLU) × 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsample → concat skip → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial sizes differ (non-power-of-2 inputs)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """Standard 2D U-Net with 4 encoder/decoder levels.

    Parameters
    ----------
    in_channels  : number of input channels (1 for grayscale MRI)
    out_channels : number of output classes (1 for binary segmentation)
    base_filters : number of filters in the first level (doubled each level)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_filters: int = 64):
        super().__init__()
        f = base_filters
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16)
        self.up1 = Up(f * 16, f * 8)
        self.up2 = Up(f * 8, f * 4)
        self.up3 = Up(f * 4, f * 2)
        self.up4 = Up(f * 2, f)
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class ResidualBlock(nn.Module):
    """Two-conv residual block with optional projection shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)
        return out


class ResDown(nn.Module):
    """Downsampling residual block (stride-2)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = ResidualBlock(in_ch, out_ch, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResUp(nn.Module):
    """Transpose-conv upsample, skip concat, then residual block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ResidualBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class ResUNet(nn.Module):
    """Residual U-Net variant used in comparison notebooks."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_filters: int = 64):
        super().__init__()
        f = base_filters
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
        )
        self.enc1 = ResidualBlock(f, f)
        self.enc2 = ResDown(f, f * 2)
        self.enc3 = ResDown(f * 2, f * 4)
        self.enc4 = ResDown(f * 4, f * 8)
        self.enc5 = ResDown(f * 8, f * 16)
        self.dec4 = ResUp(f * 16, f * 8, f * 8)
        self.dec3 = ResUp(f * 8, f * 4, f * 4)
        self.dec2 = ResUp(f * 4, f * 2, f * 2)
        self.dec1 = ResUp(f * 2, f, f)
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        x6 = self.enc5(x5)
        x = self.dec4(x6, x5)
        x = self.dec3(x, x4)
        x = self.dec2(x, x3)
        x = self.dec1(x, x2)
        return self.outc(x)


class AttentionGate(nn.Module):
    """Additive attention gate for skip connections."""

    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class AttentionUpBlock(nn.Module):
    """Upsample, attend to skip, then conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.att = AttentionGate(g_ch=out_ch, x_ch=skip_ch, inter_ch=max(out_ch // 2, 1))
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        skip = self.att(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class AttentionUNet(nn.Module):
    """Attention U-Net used in the three-model segmentation comparison."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_filters: int = 64):
        super().__init__()
        f = base_filters
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f * 2)
        self.enc3 = DoubleConv(f * 2, f * 4)
        self.enc4 = DoubleConv(f * 4, f * 8)
        self.bottleneck = DoubleConv(f * 8, f * 16)
        self.pool = nn.MaxPool2d(2)
        self.dec4 = AttentionUpBlock(f * 16, f * 8, f * 8)
        self.dec3 = AttentionUpBlock(f * 8, f * 4, f * 4)
        self.dec2 = AttentionUpBlock(f * 4, f * 2, f * 2)
        self.dec1 = AttentionUpBlock(f * 2, f, f)
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        x5 = self.bottleneck(self.pool(x4))
        x = self.dec4(x5, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)
        return self.outc(x)


def build_segmentation_model(
    name: str,
    in_channels: int = 1,
    out_channels: int = 1,
    base_filters: int = 64,
) -> nn.Module:
    """Factory for UNet, ResUNet, and AttentionUNet."""
    key = name.lower().replace("-", "").replace("_", "")
    if key == "unet":
        return UNet(in_channels, out_channels, base_filters)
    if key == "resunet":
        return ResUNet(in_channels, out_channels, base_filters)
    if key in {"attentionunet", "attunet"}:
        return AttentionUNet(in_channels, out_channels, base_filters)
    raise ValueError(f"Unknown segmentation model: {name}")
