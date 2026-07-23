"""Test the updated SGBlock with Block B auxiliary branch and lighter config."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias,
                                stride=self.conv.stride, padding=0, groups=self.conv.groups)
            return out_normal - self.theta * out_diff


class simam_module(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class SGBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, keep_3x3=False, avgdown=False):
        super(SGBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)

        self.identity = False
        self.identity_div = 1
        self.expand_ratio = expand_ratio
        self.stride = stride

        # --- Block B auxiliary branch (AvgPool shortcut for downsampling) ---
        self.downsample = None
        if stride == 2 and avgdown:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(2, stride=2),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, kernel_size=1, bias=False),
            )

        if expand_ratio == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        elif inp != oup and stride == 1 and keep_3x3 == False:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        elif inp != oup and stride == 2 and keep_3x3 == False:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            if keep_3x3 == False:
                self.identity = True
            self.conv = nn.Sequential(
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                nn.Conv2d(oup, oup, 3, 1, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)

        # Block B: use AvgPool auxiliary branch for downsampling
        if self.downsample is not None:
            return self.downsample(x) + out

        if self.identity:
            shape = x.shape
            id_tensor = x[:, :shape[1]//self.identity_div, :, :]
            out[:, :shape[1]//self.identity_div, :, :] = \
                out[:, :shape[1]//self.identity_div, :, :] + id_tensor
            return out
        else:
            return out


class MXNet(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., in_channels=3):
        super(MXNet, self).__init__()
        # Lighter config for 80×80 input — only 2 downsamples (80→40→20)
        self.cfgs = [
            # t,   c,  n, s, avgdown
            [2,   64,  1, 1, False],
            [6,   96,  1, 1, False],
            [6,  128,  2, 2, True],    # 80→40, Block B downsample
            [6,  192,  2, 1, False],
            [6,  256,  2, 2, True],    # 40→20, Block B downsample
            [6,  384,  1, 1, False],
            [6,  512,  1, 1, False],
        ]

        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(in_channels, input_channel, 1)]
        block = SGBlock
        for t, c, n, s, avgd in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            if c == 1280 and width_mult < 1:
                output_channel = 1280
            layers.append(block(input_channel, output_channel, s, t, n==1 and s==1, avgdown=avgd))
            input_channel = output_channel
            for i in range(n-1):
                layers.append(block(input_channel, output_channel, 1, t, avgdown=False))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)

        input_channel = output_channel
        output_channel = _make_divisible(input_channel, 4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class FTGenerator(nn.Module):
    def __init__(self, in_channels=128, out_channels=1, theta=0.7):
        super(FTGenerator, self).__init__()
        self.ft = nn.Sequential(
            Conv2d_cd(in_channels, 128, kernel_size=3, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            simam_module(),

            Conv2d_cd(128, 64, kernel_size=3, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            simam_module(),

            Conv2d_cd(64, out_channels, kernel_size=3, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ft(x)


class MultiFTNet(nn.Module):
    def __init__(self, num_classes=2, img_channel=3, embedding_size=128,
                 ft_size=None, theta=0.7, **kwargs):
        super(MultiFTNet, self).__init__()
        self.num_classes = num_classes
        self.ft_size = ft_size

        backbone = MXNet(num_classes=num_classes, in_channels=img_channel)
        features = list(backbone.features.children())

        # Print layer info for debugging
        print(f"\n  Total backbone layers: {len(features)}")
        for i, layer in enumerate(features):
            print(f"    Layer {i}: {layer.__class__.__name__}", end="")
            if hasattr(layer, 'stride'):
                print(f" stride={layer.stride}", end="")
            if hasattr(layer, 'downsample') and layer.downsample is not None:
                print(f" [Block B - AvgPool aux]", end="")
            print()

        # 3-way split — mid ends at 128ch (layer 4), late starts at 192ch (layer 5)
        self.shallow_features = nn.Sequential(*features[:3])   # → 96ch @ 80×80
        self.mid_features = nn.Sequential(*features[3:5])      # → 128ch @ 40×40
        self.late_features = nn.Sequential(*features[5:])      # → 512ch @ 20×20

        # Stage A → Stage B fusion
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

        # CDC + SimAM FTGenerator
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
        x_shallow = self.shallow_features(x)
        x_mid = self.mid_features(x_shallow)

        # Classification path
        x_late = self.late_features(x_mid)
        x_late = self.avgpool(x_late)
        x_late = x_late.view(x_late.size(0), -1)
        x_late = self.linear(x_late)
        x_late = self.bn(x_late)
        x_late = self.drop(x_late)
        cls = self.prob(x_late)

        if self.training:
            shallow_proj = self.shallow_proj(x_shallow)
            shallow_pooled = F.adaptive_avg_pool2d(shallow_proj, x_mid.shape[2:])
            ft_input = x_mid + shallow_pooled
            ft = self.FTGenerator(ft_input)
            if self.ft_size is not None:
                ft = F.interpolate(ft, size=self.ft_size, mode='bilinear', align_corners=False)
            return cls, ft
        else:
            return cls


if __name__ == "__main__":
    # Test
    model = MultiFTNet(num_classes=2, img_channel=3, embedding_size=128, ft_size=(10, 10), theta=0.7)

    # === Parameter Breakdown ===
    total_params = sum(p.numel() for p in model.parameters())

    # Train-only modules (not used during inference)
    train_only_modules = {
        'FTGenerator': model.FTGenerator,
        'shallow_proj': model.shallow_proj,
    }
    train_only_params = sum(
        p.numel() for mod in train_only_modules.values() for p in mod.parameters()
    )

    # Inference modules breakdown
    inference_modules = {
        'shallow_features': model.shallow_features,
        'mid_features': model.mid_features,
        'late_features': model.late_features,
        'avgpool': model.avgpool,
        'linear': model.linear,
        'bn': model.bn,
        'prob': model.prob,
    }
    inference_params = sum(
        p.numel() for mod in inference_modules.values() for p in mod.parameters()
    )

    print(f"\n{'='*55}")
    print(f"  Parameter Breakdown")
    print(f"{'='*55}")
    print(f"  📊 Total params:          {total_params:>10,}")
    print(f"  🚀 Inference params:      {inference_params:>10,}  {'✅' if inference_params < 450_000 else '❌'} (<450K target)")
    print(f"  🎓 Train-only params:     {train_only_params:>10,}")
    print(f"{'─'*55}")
    print(f"  Inference breakdown:")
    for name, mod in inference_modules.items():
        p = sum(p.numel() for p in mod.parameters())
        if p > 0:
            print(f"    {name:25s} {p:>10,}")
    print(f"{'─'*55}")
    print(f"  Train-only breakdown:")
    for name, mod in train_only_modules.items():
        p = sum(p.numel() for p in mod.parameters())
        print(f"    {name:25s} {p:>10,}")
    print(f"{'='*55}")

    # Test forward
    model.train()
    x = torch.randn(2, 3, 80, 80)
    cls, ft = model(x)
    print(f"\n  Train mode: cls={cls.shape}, ft={ft.shape}")

    model.eval()
    with torch.no_grad():
        cls = model(x)
    print(f"  Eval mode:  cls={cls.shape}")

    # Trace feature map sizes
    print("\n  Feature map trace:")
    with torch.no_grad():
        model.eval()
        t = torch.randn(1, 3, 80, 80)
        t1 = model.shallow_features(t)
        print(f"    After shallow_features: {t1.shape}")
        t2 = model.mid_features(t1)
        print(f"    After mid_features:     {t2.shape}")
        t3 = model.late_features(t2)
        print(f"    After late_features:    {t3.shape}")
