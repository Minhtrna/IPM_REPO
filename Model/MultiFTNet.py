
import torch
from torch import nn
import torch.nn.functional as F
from Model.mnext import MXNet
from Model.CDC import Conv2d_cd


class FTGenerator(nn.Module):
    """Generates Fourier feature map from intermediate backbone features."""

    def __init__(self, in_channels=128, out_channels=1, theta=0.7):
        super(FTGenerator, self).__init__()
        self.ft = nn.Sequential(
            Conv2d_cd(in_channels, 128, kernel_size=3, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            Conv2d_cd(128, 64, kernel_size=3, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            Conv2d_cd(64, out_channels, kernel_size=3, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ft(x)


class MultiFTNet(nn.Module):
    """Multi-task FAS model with multi-scale FT fusion.

    Architecture:
        Input → shallow (96ch, 80×80) → mid (128ch, 40×40) → late (512ch) → classification
                    ↓ proj+pool              ↓
                    └──── element-wise add ───┘
                              ↓
                       FTGenerator (CDC) → Fourier feature map

    The shallow features provide fine-grained texture info (moiré, print patterns)
    that enriches CDC's frequency-domain feature extraction in FTGenerator.
    """

    def __init__(self, num_classes=2, img_channel=3, embedding_size=128, ft_size=None, theta=0.7, **kwargs):
        super(MultiFTNet, self).__init__()
        self.num_classes = num_classes
        self.ft_size = ft_size

        # Build MobileNeXt backbone
        backbone = MXNet(num_classes=num_classes, in_channels=img_channel)
        features = list(backbone.features.children())

        # 3-way split: shallow / mid / late
        self.shallow_features = nn.Sequential(*features[:3])   # 96ch, 80×80
        self.mid_features = nn.Sequential(*features[3:6])      # 128ch, 40×40
        self.late_features = nn.Sequential(*features[6:])      # 512ch, 20×20

        # Stage A → B fusion: project 96ch → 128ch (element-wise add, no extra CDC cost)
        self.shallow_proj = nn.Sequential(
            nn.Conv2d(96, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.drop = nn.Dropout(p=0.2)
        self.prob = nn.Linear(embedding_size, num_classes, bias=False)

        # CDC FTGenerator (unchanged — still takes 128ch)
        self.FTGenerator = FTGenerator(in_channels=128, out_channels=1, theta=theta)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d_cd):
                nn.init.kaiming_normal_(m.conv.weight, mode='fan_out', nonlinearity='relu')
                if m.conv.bias is not None:
                    nn.init.constant_(m.conv.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stage A: shallow features (96ch, 80×80)
        x_shallow = self.shallow_features(x)

        # Stage B: mid features (128ch, 40×40)
        x_mid = self.mid_features(x_shallow)

        # Classification path (unchanged)
        x_late = self.late_features(x_mid)
        x_late = self.avgpool(x_late)
        x_late = x_late.view(x_late.size(0), -1)
        x_late = self.linear(x_late)
        x_late = self.bn(x_late)
        x_late = self.drop(x_late)
        cls = self.prob(x_late)

        if self.training:
            # Fuse Stage A + B for FTGenerator
            shallow_proj = self.shallow_proj(x_shallow)                             # 96→128ch, 80×80
            shallow_pooled = F.adaptive_avg_pool2d(shallow_proj, x_mid.shape[2:])   # → 40×40
            ft_input = x_mid + shallow_pooled                                       # element-wise add
            ft = self.FTGenerator(ft_input)
            if self.ft_size is not None:
                ft = F.interpolate(ft, size=self.ft_size, mode='bilinear', align_corners=False)
            return cls, ft
        else:
            return cls

