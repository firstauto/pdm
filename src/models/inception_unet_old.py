import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionUNet(nn.Module):
    def __init__(self, in_channels, input_size, emb_size, kernel_size, stride, dropout):
        super(InceptionUNet, self).__init__()

        # self.proj = nn.Linear(input_size, emb_size)

        self.attn = Attention(input_size)
# I want to calculate fast Fourier transform of encoded state from an autoencoder in a learnable way. From this, I intend to obtain phase, amplitude, offset, and frequency. The shape of the encoded state is (batch size, channels, time_range).
        self.enc1 = Encoder(in_channels, in_channels*2)
        self.enc2 = Encoder(in_channels*2, in_channels*4)
        self.enc3 = Encoder(in_channels*4, in_channels*8)
        self.enc4 = Encoder(in_channels*8, in_channels*16)

        self.bridge = InceptionModule(in_channels*16, in_channels*16)

        self.dec4 = Decoder(in_channels*16, in_channels*8)
        self.dec3 = Decoder(in_channels*8, in_channels*4)
        self.dec2 = Decoder(in_channels*4, in_channels*2)
        self.dec1 = Decoder(in_channels*2, in_channels)
        # self.dec0 = InceptionModule(in_channels*2, in_channels)

        self.final = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        # self.unproj = nn.Linear(emb_size, input_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt=None):

        x = x.permute(0, 2, 1)
        x = self.attn(x, x) + x
        x1, x = self.enc1(x)
        x2, x = self.enc2(x)
        x3, x = self.enc3(x)
        x4, x = self.enc4(x)
        
        x = self.bridge(x)
        
        x = self.dec4(x, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)
        # x = self.dec0(x)

        x = self.dropout(self.final(x))
        # x = self.unproj(x)
        return x.permute(0, 2, 1)
    

class Attention(nn.Module):
    def __init__(self, input_size):
        super(Attention, self).__init__()
        self.attn = nn.MultiheadAttention(input_size, num_heads=4, dropout=0.1, batch_first=True)
        self.ln = nn.LayerNorm(input_size)

    def forward(self, x, tgt):
        x = self.ln(x)
        x, _ = self.attn(x, x, tgt)
        return x
 
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.incept = InceptionModule(in_channels, out_channels)
        self.pool = AdaptAvgMax_Pool(2, 2)

    def forward(self, x):
        x = self.incept(x)
        x_downsampled = self.pool(x)
        return x, x_downsampled
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.incept = InceptionModule(in_channels + out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.incept(x)
        return x
    
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()
        self.branch1x1 = nn.Conv1d(in_channels, out_channels//2, kernel_size=1)

        self.branch3x3_ = nn.Conv1d(in_channels, out_channels//8, kernel_size=1)
        self.branch3x3 = nn.Conv1d(out_channels//8, out_channels//4, kernel_size=3, padding=1)

        self.branch5x5_ = nn.Conv1d(in_channels, out_channels//16, kernel_size=1)
        self.branch5x5 = nn.Conv1d(out_channels//16, out_channels//8, kernel_size=5, padding=2)

        self.branch_7x7_ = nn.Conv1d(in_channels, out_channels//16, kernel_size=1)
        self.branch_7x7 = nn.Conv1d(out_channels//16, out_channels//8, kernel_size=7, padding=3)

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

        outputs = torch.cat([branch1x1, branch3x3, branch5x5, branch_7x7], dim=1)
        outputs = self.act(self.bn(outputs))
        return outputs
    
class AdaptAvgMax_Pool(nn.Module):
    def __init__(self, kernel_size, stride, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(AdaptAvgMax_Pool, self).__init__()
        self.avgpool = nn.AvgPool1d(kernel_size, stride, padding, ceil_mode, count_include_pad=True)
        self.maxpool = nn.MaxPool1d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    def forward(self, x):
        avg = self.avgpool(x)
        max = self.maxpool(x)
        return 0.5 * (avg + max)

# Example usage
# model = InceptionUNet(input_channels=1, num_classes=1)
# print(model)
