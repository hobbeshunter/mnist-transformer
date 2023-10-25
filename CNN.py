from typing import List
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, image_size: int, hidden_dim: List[int], dropout: float, num_classes: int):

        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=1,
                out_channels=hidden_dim[0],
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(hidden_dim[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        out_dim = image_size // 2

        self.conv2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(
                in_channels=hidden_dim[0],
                out_channels=hidden_dim[1],
                kernel_size=5,
                stride=1,
                padding=2,
            )
        )

        self.conv3 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(hidden_dim[1], hidden_dim[2], 5, 1, 2),
            nn.BatchNorm2d(hidden_dim[2]),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        out_dim = out_dim // 2

        self.out = nn.Linear(hidden_dim[2] * out_dim * out_dim, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = nn.functional.relu(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
