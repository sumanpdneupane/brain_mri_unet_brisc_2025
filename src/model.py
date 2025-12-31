import torch
import torch.nn as nn
from src.util import crop_tensor


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()

        # Convolution 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0)
        self.relu1 = nn.ReLU(inplace=True)

        # Convolution 2
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Convolution 1
        out = self.conv1(x)
        out = self.relu1(out)

        # Convolution 2
        out = self.conv2(out)
        out = self.relu2(out)

        # result
        return out


class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, use_batch_norm=True, dropout_p=0.0):
        super(Encoder, self).__init__()
        self.doubleConv = DoubleConv(in_ch, out_ch)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.doubleConv(x)
        out = self.maxpool(x)
        return x, out


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, use_batch_norm=False, dropout_p=0.1):
        super(Decoder, self).__init__()

        # 1. Has no learnable parameter in Upsample just add padding to 1 in DoubleConv both conv1 and 2
        # 2. Has learnable parameter in Upsample just add padding to 1 in DoubleConv both conv1 and 2
        # 3. crop tensor
        # If input = [B, inp, H, W]
        # Output = [B, out, 2H, 2W]
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.doubleConv = DoubleConv(in_ch, out_ch)  # in_ch = out_ch*2 after concat

    def forward(self, x, res):
        x = self.up(x)

        # For 3. crop encoder feature to same size as decoder
        res_cropped = crop_tensor(res, x)
        x = torch.cat([res_cropped, x], dim=1)

        x = self.doubleConv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()

        # 1. Down sampling
        self.encoder_1 = Encoder(in_ch, 64)
        self.encoder_2 = Encoder(64, 128)
        self.encoder_3 = Encoder(128, 256)
        self.encoder_4 = Encoder(256, 512)

        # 2. Bottleneck
        self.doubleConv = DoubleConv(512, 1024)

        # 3. Up sampling
        self.decoder_4 = Decoder(1024, 512)
        self.decoder_3 = Decoder(512, 256)
        self.decoder_2 = Decoder(256, 128)
        self.decoder_1 = Decoder(128, 64)

        # 4. Output
        self.output = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        # 1. Down sampling
        r1, e1 = self.encoder_1(x)
        r2, e2 = self.encoder_2(e1)
        r3, e3 = self.encoder_3(e2)
        r4, e4 = self.encoder_4(e3)

        # 2. Bottleneck
        bottleneck = self.doubleConv(e4)

        # 3. Up sampling
        u4 = self.decoder_4(bottleneck, r4)
        u3 = self.decoder_3(u4, r3)
        u2 = self.decoder_2(u3, r2)
        u1 = self.decoder_1(u2, r1)

        # 4. Output
        output = self.output(u1)

        return output

    def formate(self, targets, predictions):
        return crop_tensor(targets, predictions)