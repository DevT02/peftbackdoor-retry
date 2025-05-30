from torch import nn
import torch.nn.functional as F
import torch.quantization as quantization
from .LoRA import LoRAConfig, LoRAConv2d, LoRALinear

class BadNet(nn.Module):

    def __init__(self, input_channels, output_num, config: LoRAConfig):
        super().__init__()

        if config is None:
            # NO LoRA
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=1),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )

            fc1_input_features = 800 if input_channels == 3 else 512
            self.fc1 = nn.Sequential(
                nn.Linear(in_features=fc1_input_features, out_features=512),
                nn.ReLU()
            )
            self.fc2 = nn.Sequential(
                nn.Linear(in_features=512, out_features=output_num),
                nn.Softmax(dim=-1)
            )
            
        else:
            # USE LoRA
            self.conv1 = nn.Sequential(
                LoRAConv2d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=1, padding=0, config=config),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )
            self.conv2 = nn.Sequential(
                LoRAConv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0, config=config),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )
            fc1_input_features = 800 if input_channels == 3 else 512
            self.fc1 = nn.Sequential(
                LoRALinear(in_features=fc1_input_features, out_features=512, config=config),
                nn.ReLU()
            )
            self.fc2 = nn.Sequential(
                LoRALinear(in_features=512, out_features=output_num, config=config),
                nn.Softmax(dim=-1)
            )

        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
