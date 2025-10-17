import torch
import torch.nn as nn

# ==========================
# CRNN (CNN + RNN) Model Class
# ==========================


class CRNN(nn.Module):
    def __init__(self, num_classes, img_height=64):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),   # [B, 64, 64, W]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # [B, 64, 32, W/2]
            nn.Conv2d(64, 128, 3, 1, 1), # [B, 128, 32, W/2]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # [B, 128, 16, W/4]
        )

        self.rnn = nn.LSTM(
            input_size=128 * 16,  # flatten height dimension
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        B, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2)  # [B, W, C, H]
        x = x.reshape(B, W, C * H) # flatten H
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)     # [T, B, num_classes] for CTC
        return x
