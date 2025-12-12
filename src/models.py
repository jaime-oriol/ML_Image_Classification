"""
Neural network models for geological sample classification.
Contains CustomCNN (baseline) and ResNet18 (transfer learning).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CustomCNN(nn.Module):
    """
    Custom Convolutional Neural Network for geological sample classification.
    Deeper architecture with 4 convolutional blocks for better feature extraction.
    """

    def __init__(self, num_classes=5):
        """
        Initialize the custom CNN architecture.

        Args:
            num_classes: Number of output classes
        """
        super(CustomCNN, self).__init__()

        # Feature extraction layers (convolutional blocks)
        self.features = nn.Sequential(
            # Block 1: RGB input -> 32 feature maps
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224->112

            # Block 2: 32 -> 64 feature maps
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112->56

            # Block 3: 64 -> 128 feature maps
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56->28

            # Block 4: 128 -> 256 feature maps (added for more capacity)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28->14
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Moderate dropout
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_resnet18(num_classes=5, pretrained=True, freeze_layers=True):
    """
    Create ResNet18 model adapted for geological sample classification.
    Uses transfer learning from ImageNet pretrained weights.

    Args:
        num_classes: Number of output classes
        pretrained: If True, use ImageNet pretrained weights
        freeze_layers: If True, freeze early conv layers

    Returns:
        Modified ResNet18 model
    """
    model = models.resnet18(pretrained=pretrained)

    if freeze_layers:
        # Freeze early layers (already good at detecting edges, colors, textures)
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.bn1.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False

    # Replace final layer for num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
