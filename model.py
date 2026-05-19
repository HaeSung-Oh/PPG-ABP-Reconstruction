import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyAttention(nn.Module):
    """
    Frequency-domain attention using FFT magnitude features.
    """

    def __init__(self, channel):
        super().__init__()
        self.conv3 = nn.Conv1d(
            channel,
            channel,
            kernel_size=3,
            padding=1,
            groups=channel,
        )
        self.conv1 = nn.Conv1d(channel, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_fft = torch.fft.rfft(x, dim=2, norm="ortho")
        x_mag = torch.abs(x_fft)

        y = self.conv3(x_mag)
        y = self.conv1(y)

        attn = torch.mean(y, dim=2, keepdim=True)
        return self.sigmoid(attn)


class SpatialAttention(nn.Module):
    """
    Spatial attention from channel-wise summary statistics.
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.conv3 = nn.Conv1d(3, 1, kernel_size=3, padding=1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)

        s = torch.cat([mean, mx, std], dim=1)
        s = self.conv3(s)
        s = self.conv1(s)

        return self.sigmoid(s)


class DualChannelAttention(nn.Module):
    """
    Channel attention using pooled channel descriptors.
    """

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        red = max(1, channel // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channel, red, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(red, channel, bias=False),
        )
        self.conv1 = nn.Conv1d(channel, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()

        z = self.avg_pool(x)
        mlp_out = self.mlp(z.view(b, c)).view(b, c, 1)
        conv_out = self.conv1(z)

        attn = self.sigmoid(mlp_out + conv_out)
        return x * attn


class TDA(nn.Module):
    """
    Tri-domain attention with spatial and frequency branches.

    Each branch applies channel attention before its domain-specific
    attention, and the branch outputs are fused with a residual connection.
    """

    def __init__(self, channel):
        super().__init__()
        spatial_channel = channel // 2
        frequency_channel = channel - spatial_channel

        self.spatial_channel = spatial_channel
        self.channel_attn_spatial = DualChannelAttention(spatial_channel)
        self.channel_attn_frequency = DualChannelAttention(frequency_channel)
        self.spatial_attn = SpatialAttention()
        self.frequency_attn = FrequencyAttention(frequency_channel)
        self.fusion = nn.Conv1d(channel, channel, kernel_size=1)

    def forward(self, x):
        identity = x
        x_spatial = x[:, :self.spatial_channel, :]
        x_frequency = x[:, self.spatial_channel:, :]
        xc_spatial = self.channel_attn_spatial(x_spatial)
        xc_frequency = self.channel_attn_frequency(x_frequency)

        xs = xc_spatial * self.spatial_attn(xc_spatial)
        xf = xc_frequency * self.frequency_attn(xc_frequency)

        y = torch.cat([xs, xf], dim=1)
        y = self.fusion(y)
        return identity + y

class DWConv1d(nn.Module):
    def __init__(self, in_channels, dilation):
        super().__init__()
        self.dw = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=in_channels,
        )

    def forward(self, x):
        return self.dw(x)


class PWConv1d(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.pw = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.pw(x))


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class CDI(nn.Module):
    """
    Cross-dilation interaction block using paired dilation branches.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dil_r=(1, 2, 3, 4),
    ):
        super().__init__()
        self.dil_r = [2 ** i - 1 for i in dil_r]

        c_in = in_channels
        c_out = out_channels
        c_bottleneck = c_in
        path1_out = c_out // 2
        path2_out = c_out - path1_out
        pair1_out = max(1, c_bottleneck // 2)
        pair2_out = max(1, c_bottleneck - pair1_out)

        self.stem = PWConv1d(c_in, c_bottleneck)

        self.path1_dw_pw = nn.ModuleList(
            [
                nn.Sequential(
                    DWConv1d(c_bottleneck, d),
                    PWConv1d(c_bottleneck),
                )
                for d in self.dil_r
            ]
        )
        self.path1_pairs = [
            (i, j) for i in range(len(self.dil_r)) for j in range(i + 1, len(self.dil_r))
        ]
        self.path1_pw = nn.ModuleList(
            [
                PWConv1d(2 * c_bottleneck, pair1_out)
                for _ in self.path1_pairs
            ]
        )
        self.path1_fusion = PWConv1d(
            len(self.path1_pairs) * pair1_out,
            path1_out,
        )

        self.path2_pairs = []
        self.path2_branches = nn.ModuleDict()
        self.path2_pw = nn.ModuleList()

        for i, d1 in enumerate(self.dil_r):
            for j, d2 in enumerate(self.dil_r):
                if i < j:
                    self.path2_pairs.append((d1, d2))
                    self.path2_branches[f"{d1}_{d2}"] = nn.Sequential(
                        DWConv1d(c_bottleneck, d1),
                        PWConv1d(c_bottleneck),
                        DWConv1d(c_bottleneck, d2),
                    )
                    self.path2_branches[f"{d2}_{d1}"] = nn.Sequential(
                        DWConv1d(c_bottleneck, d2),
                        PWConv1d(c_bottleneck),
                        DWConv1d(c_bottleneck, d1),
                    )
                    self.path2_pw.append(PWConv1d(c_bottleneck, pair2_out))

        self.path2_fusion = PWConv1d(
            len(self.path2_pw) * pair2_out,
            path2_out,
        )
        self.output_fusion = PWConv1d(
            path1_out + path2_out,
            c_out,
        )

        if in_channels != out_channels:
            self.residual_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):
        identity = self.residual_conv(x)
        x_b = self.stem(x)

        feats = [branch(x_b) for branch in self.path1_dw_pw]
        pair_feats = []
        for (a, b), pw in zip(self.path1_pairs, self.path1_pw):
            pair_feats.append(pw(torch.cat([feats[a], feats[b]], dim=1)))
        path1 = self.path1_fusion(torch.cat(pair_feats, dim=1))

        path2_feats = []
        for (d1, d2), pw in zip(self.path2_pairs, self.path2_pw):
            f1 = self.path2_branches[f"{d1}_{d2}"](x_b)
            f2 = self.path2_branches[f"{d2}_{d1}"](x_b)
            path2_feats.append(pw(f1 + f2))
        path2 = self.path2_fusion(torch.cat(path2_feats, dim=1))

        out = torch.cat([path1, path2], dim=1)
        out = self.output_fusion(out)
        return out + identity


class DownsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

        self.cdi = CDI(out_channels, out_channels)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.cdi(x)

        skip = x
        x = self.pool(x)
        return x, skip


def _upsample_like(src, tar):
    return F.interpolate(src, size=tar.shape[2:], mode="linear", align_corners=False)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv1 = ConvBlock(out_channels * 2, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.proj(x)
        x = _upsample_like(x, skip)
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class model(nn.Module):
    def __init__(
        self,
        input_channels=1,
    ):
        super().__init__()
        self.filters = [32, 64, 128, 256, 512]

        self.down1 = DownsampleBlock(
            input_channels,
            self.filters[0],
        )
        self.down2 = DownsampleBlock(
            self.filters[0],
            self.filters[1],
        )
        self.down3 = DownsampleBlock(
            self.filters[1],
            self.filters[2],
        )
        self.down4 = DownsampleBlock(
            self.filters[2],
            self.filters[3],
        )

        self.bridge_conv1 = ConvBlock(self.filters[3], self.filters[4])
        self.bridge_conv2 = ConvBlock(self.filters[4], self.filters[4])

        self.bridge_cdi = CDI(
            self.filters[4],
            self.filters[4],
        )

        self.up1 = UpsampleBlock(self.filters[4], self.filters[3])
        self.up2 = UpsampleBlock(self.filters[3], self.filters[2])
        self.up3 = UpsampleBlock(self.filters[2], self.filters[1])
        self.up4 = UpsampleBlock(self.filters[1], self.filters[0])

        self.tda_skip1 = TDA(self.filters[0])
        self.tda_skip2 = TDA(self.filters[1])
        self.tda_skip3 = TDA(self.filters[2])
        self.tda_skip4 = TDA(self.filters[3])

        self.final_conv = nn.Conv1d(self.filters[0], 1, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)

        x = self.bridge_conv1(x)
        x = self.bridge_conv2(x)

        x = self.bridge_cdi(x)

        skip1 = self.tda_skip1(skip1)
        skip2 = self.tda_skip2(skip2)
        skip3 = self.tda_skip3(skip3)
        skip4 = self.tda_skip4(skip4)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        return self.final_conv(x)
