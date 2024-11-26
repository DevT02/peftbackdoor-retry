import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class LoRAConfig:
    def __init__(self, rank=8, lora_alpha=1.0, lora_dropout=0.1, freeze_weights=True):
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.freeze_weights = freeze_weights

class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, config: LoRAConfig, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)
        self.rank = config.rank
        self.lora_alpha = config.lora_alpha
        self.freeze_weights = config.freeze_weights
        self.lora_matrix_B = nn.Parameter(torch.zeros(out_features, self.rank))
        self.lora_matrix_A = nn.Parameter(torch.randn(self.rank, in_features) * 0.01)
        self.lora_dropout = nn.Dropout(p=config.lora_dropout)

        if self.freeze_weights:
            self.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tensor:
        x_dropped = self.lora_dropout(x)
        lora_weights = self.lora_alpha * torch.matmul(self.lora_matrix_B, self.lora_matrix_A)
        return super().forward(x) + F.linear(x_dropped, lora_weights, bias=None)

    # rank -> r
    # lora_alpha -> alpha
    # lora_dropout -> dropout rate

class LoRAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, config: LoRAConfig, stride=1, padding=0, bias=True):
        super(LoRAConv2d, self).__init__()
        self.rank = config.rank
        self.lora_alpha = config.lora_alpha
        self.freeze_weights = config.freeze_weights
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if self.freeze_weights:
            self.conv.weight.requires_grad = False
            if bias:
                self.conv.bias.requires_grad = False
        self.A = nn.Parameter(torch.randn(self.rank, in_channels, 1, 1) * 0.01) 
        self.B = nn.Parameter(torch.zeros(out_channels, self.rank, kernel_size, kernel_size))
        self.lora_dropout = nn.Dropout(p=config.lora_dropout)

    def forward(self, x):
        original_out = self.conv(x)
        lora_conv = F.conv2d(self.lora_dropout(x), self.A, bias=None, stride=1, padding=0)
        lora_out = F.conv2d(lora_conv, self.B, bias=None, stride=self.conv.stride, padding=self.conv.padding)
        return original_out + self.lora_alpha * lora_out

