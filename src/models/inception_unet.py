import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim


class InceptionUNet(nn.Module):
    def __init__(self, in_channels, input_size, emb_size, kernel_size, stride, dropout):
        super(InceptionUNet, self).__init__()

        # Encoder
        self.enc1 = nn.Conv1d(
            in_channels,
            in_channels // 2,
            kernel_size,
            padding=((kernel_size - 1) // 2) * 2,
            dilation=2,
        )
        self.norm1 = nn.BatchNorm1d(in_channels // 2)
        self.enc2 = nn.Conv1d(
            in_channels // 2,
            in_channels // 4,
            kernel_size,
            padding=((kernel_size - 1) // 2) * 2,
            dilation=2,
        )
        self.norm2 = nn.BatchNorm1d(in_channels // 4)
        self.enc3 = nn.Conv1d(
            in_channels // 4,
            in_channels // 8,
            kernel_size,
            padding=((kernel_size - 1) // 2) * 2,
            dilation=2,
        )
        self.norm3 = nn.BatchNorm1d(in_channels // 8)
        self.enc4 = nn.Conv1d(
            in_channels // 8,
            in_channels // 16,
            kernel_size,
            padding=((kernel_size - 1) // 2) * 2,
            dilation=2,
        )
        self.norm4 = nn.BatchNorm1d(in_channels // 16)

        # Bridge
        self.bridge = nn.Conv1d(in_channels // 16, in_channels // 16, kernel_size=1)
        self.norm_bridge = nn.BatchNorm1d(in_channels // 16)

        # Decoder with transposed convolutions for upsampling
        self.up4 = nn.ConvTranspose1d(
            in_channels // 16, in_channels // 8, kernel_size=2, stride=2
        )
        self.dec4 = nn.Conv1d(
            (in_channels // 8) + (in_channels // 16),
            in_channels // 8,
            kernel_size,
            padding=((kernel_size - 1) // 2),
        )
        self.norm4_ = nn.BatchNorm1d(in_channels // 8)

        self.up3 = nn.ConvTranspose1d(
            in_channels // 8, in_channels // 4, kernel_size=2, stride=2
        )
        self.dec3 = nn.Conv1d(
            (in_channels // 4) + (in_channels // 8),
            in_channels // 4,
            kernel_size,
            padding=((kernel_size - 1) // 2),
        )
        self.norm3_ = nn.BatchNorm1d(in_channels // 4)

        self.up2 = nn.ConvTranspose1d(
            in_channels // 4, in_channels // 2, kernel_size=2, stride=2
        )
        self.dec2 = nn.Conv1d(
            (in_channels // 4) + (in_channels // 2),
            in_channels // 2,
            kernel_size,
            padding=((kernel_size - 1) // 2),
        )
        self.norm2_ = nn.BatchNorm1d(in_channels // 2)

        self.up1 = nn.ConvTranspose1d(
            in_channels // 2, in_channels, kernel_size=2, stride=2
        )
        self.dec1 = nn.Conv1d(
            (in_channels // 2) + in_channels,
            in_channels,
            kernel_size,
            padding=((kernel_size - 1) // 2),
        )
        self.norm1_ = nn.BatchNorm1d(in_channels)

        self.final = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt=None):
        # Encoder
        enc1 = F.gelu(self.norm1(self.enc1(x)))
        enc2 = F.gelu(self.norm2(self.enc2(enc1)))
        enc3 = F.gelu(self.norm3(self.enc3(enc2)))
        enc4 = F.gelu(self.norm4(self.enc4(enc3)))

        # Bridge
        x = self.norm_bridge(self.bridge(enc4))

        # Decoder with skip connections
        x = self.up4(x)
        x = self.center_crop_and_concat(x, enc4)
        x = F.gelu(self.norm4_(self.dec4(x)))

        x = self.up3(x)
        x = self.center_crop_and_concat(x, enc3)
        x = F.gelu(self.norm3_(self.dec3(x)))

        x = self.up2(x)
        x = self.center_crop_and_concat(x, enc2)
        x = F.gelu(self.norm2_(self.dec2(x)))

        x = self.up1(x)
        x = self.center_crop_and_concat(x, enc1)
        x = F.gelu(self.norm1_(self.dec1(x)))

        x = self.final(x)
        x = self.dropout(x)
        return x

    def center_crop_and_concat(self, upsampled, bypass):
        """Crop the upsampled tensor to the size of the bypass tensor and concatenate."""
        _, _, H = upsampled.size()
        _, _, H_bypass = bypass.size()

        if H != H_bypass:
            diff = (H - H_bypass) // 2
            upsampled = upsampled[:, :, diff : (diff + H_bypass)]

        return torch.cat([upsampled, bypass], dim=1)


class Attention(nn.Module):
    def __init__(self, input_size):
        super(Attention, self).__init__()
        self.attn = nn.MultiheadAttention(
            input_size, num_heads=5, dropout=0.1, batch_first=True
        )
        self.ln = nn.LayerNorm(input_size)

    def forward(self, x):
        x_ = x
        x = self.ln(x)
        x, _ = self.attn(x, x_, x_)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.incept = InceptionModule(in_channels, out_channels)

    def forward(self, x):
        x = self.incept(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.incept = InceptionModule(in_channels, out_channels)

    def forward(self, x):
        x = self.incept(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        self.branch1x1 = nn.Conv1d(in_channels, out_channels // 2, kernel_size=1)

        self.branch3x3_ = nn.Conv1d(in_channels, out_channels // 8, kernel_size=1)
        self.branch3x3 = nn.Conv1d(
            out_channels // 8, out_channels // 4, kernel_size=3, padding=1
        )

        self.branch5x5_ = nn.Conv1d(in_channels, out_channels // 16, kernel_size=1)
        self.branch5x5 = nn.Conv1d(
            out_channels // 16, out_channels // 8, kernel_size=5, padding=2
        )

        self.branch_7x7_ = nn.Conv1d(in_channels, out_channels // 16, kernel_size=1)
        self.branch_7x7 = nn.Conv1d(
            out_channels // 16, out_channels // 8, kernel_size=7, padding=3
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.Mish()

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_(x)
        branch3x3 = self.branch3x3(branch3x3)

        branch5x5 = self.branch5x5_(x)
        branch5x5 = self.branch5x5(branch5x5)

        branch_7x7 = self.branch_7x7_(x)
        branch_7x7 = self.branch_7x7(branch_7x7)

        outputs = torch.cat([branch1x1, branch3x3, branch5x5, branch_7x7], 1)
        outputs = self.act(self.bn(outputs))
        return outputs


class AdaptAvgMax_Pool(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        super(AdaptAvgMax_Pool, self).__init__()
        self.avgpool = nn.AvgPool1d(
            kernel_size, stride, padding, ceil_mode, count_include_pad=True
        )
        self.maxpool = nn.MaxPool1d(
            kernel_size, stride, padding, dilation, return_indices, ceil_mode
        )

    def forward(self, x):
        avg = self.avgpool(x)
        max = self.maxpool(x)
        return 0.5 * (avg + max)
