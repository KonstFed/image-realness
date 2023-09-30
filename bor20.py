import torch.nn as nn

class LivenessNet(nn.Module):
    def __init__(self):
        super(LivenessNet, self).__init__()
        linear_size = SIZE // 4

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding='same')
        self.norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding='same')
        self.norm2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding='same')
        self.norm3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding='same')
        self.norm4 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)

        self.linear = nn.Linear(32 * linear_size * linear_size, 64)
        self.norm5 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.5)
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm2(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.norm3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.norm4(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.linear(x.flatten(start_dim=1))
        x = self.relu(x)
        x = self.norm5(x)
        x = self.dropout3(x)
        x = self.out(x)
        return x