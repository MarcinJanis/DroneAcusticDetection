import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mamba import MambaLite


class ConvBlock(nn.Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_size: int = 2,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(ch_out)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        return x


class DroneDetectorMamba(nn.Module):
    def __init__(
        self,
        ch_in: int = 1,
        num_classes: int = 2,
        n_mels: int = 128,
        dropout_rate: float = 0.3,
        mamba_d_model: int = 512,
        mamba_d_state: int = 64,
        mamba_d_conv: int = 3,
        mamba_expand: int = 2,
    ):
        super().__init__()

        if n_mels % 16 != 0:
            raise ValueError(
                f"n_mels must be divisible by 16 because of 4 pooling layers. Got n_mels={n_mels}."
            )

        self.ch_in = ch_in
        self.num_classes = num_classes
        self.n_mels = n_mels

        self.conv1 = ConvBlock(ch_in, 32, kernel_size=3, padding=1)
        self.conv2 = ConvBlock(32, 64, kernel_size=3, padding=1)
        self.conv3 = ConvBlock(64, 128, kernel_size=3, padding=1)
        self.conv4 = ConvBlock(128, 256, kernel_size=3, padding=1)

        self.mel_features = n_mels // 16

        self.cnn_feature_dim = 256 * self.mel_features

        self.proj = nn.Linear(self.cnn_feature_dim, mamba_d_model)
        self.input_dropout = nn.Dropout(dropout_rate)

        # dwa bloki MambaLite
        self.mamba1 = MambaLite(
            d_model=mamba_d_model,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=0.2,
        )
        self.mamba2 = MambaLite(
            d_model=mamba_d_model,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=0.2,
        )

        self.final_norm = nn.LayerNorm(mamba_d_model)

        self.classifier = nn.Sequential(
            nn.Linear(mamba_d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        b, c, mel, t = x.shape

        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * mel)

        x = self.proj(x)
        x = self.input_dropout(x)

        x = self.mamba1(x)
        x = self.mamba2(x)

        x = self.final_norm(x)

        # pooling po czasie
        x = x.mean(dim=1)

        x = self.classifier(x)
        return x