import torch.nn as nn
import torch.nn.functional as F

class BetterCNN(nn.Module):
    def __init__(self, image_width, in_channels, num_classes):
        super(BetterCNN, self).__init__()

        self.image_width = image_width

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Define batch normalization layers
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

        # Define dropout layer
        self.dropout = nn.Dropout(0.5)

        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * (image_width // 8) * (image_width // 8), 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, 128 * (self.image_width // 8) * (self.image_width // 8))

        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Adding dropout for regularization
        x = F.softmax(self.fc2(x), dim=1)

        return x