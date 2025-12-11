"""
Neural network models for logo classification.
Contains CustomCNN (baseline) and ResNet18 (transfer learning).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CustomCNN(nn.Module):
    """
    Custom Convolutional Neural Network for logo classification.
    Simple baseline architecture with 3 convolutional blocks.
    Expected accuracy: 70-75% on test set.
    """

    def __init__(self, num_classes=26):
        """
        Initialize the custom CNN architecture.

        Args:
            num_classes: Number of output classes (26 leagues)
        """
        super(CustomCNN, self).__init__()

        # Feature extraction layers (convolutional blocks)
        # Each block: Conv -> ReLU activation -> MaxPooling (reduces size by half)
        self.features = nn.Sequential(
            # Block 1: Process RGB input (3 channels) -> 32 feature maps
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input: 224x224x3, Output: 224x224x32
            nn.ReLU(inplace=True),                        # Activation function (non-linearity)
            nn.MaxPool2d(kernel_size=2, stride=2),        # Downsample to 112x112x32

            # Block 2: Increase features from 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Input: 112x112x32, Output: 112x112x64
            nn.ReLU(inplace=True),                         # Activation
            nn.MaxPool2d(kernel_size=2, stride=2),         # Downsample to 56x56x64

            # Block 3: Increase features from 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Input: 56x56x64, Output: 56x56x128
            nn.ReLU(inplace=True),                          # Activation
            nn.MaxPool2d(kernel_size=2, stride=2),          # Downsample to 28x28x128
        )

        # Classification layers (fully connected)
        self.classifier = nn.Sequential(
            # Flatten 3D tensor (28x28x128) into 1D vector (100352)
            nn.Flatten(),

            # First dense layer: compress features to 512 neurons
            nn.Linear(128 * 28 * 28, 512),  # 100352 -> 512
            nn.ReLU(inplace=True),           # Activation
            nn.Dropout(0.5),                 # Dropout 50% to prevent overfitting

            # Output layer: map 512 features to 26 league classes
            nn.Linear(512, num_classes)      # 512 -> 26 (no activation, will use softmax later)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Output tensor of shape (batch_size, num_classes)
            Raw logits (no softmax, will be applied during loss calculation)
        """
        # Pass through convolutional feature extractor
        x = self.features(x)      # (batch, 3, 224, 224) -> (batch, 128, 28, 28)

        # Pass through classifier
        x = self.classifier(x)    # (batch, 128, 28, 28) -> (batch, 26)

        return x  # Return raw scores (logits) for each of 26 leagues


def get_resnet18(num_classes=26, pretrained=True):
    """
    Create ResNet18 model adapted for logo classification.
    Uses transfer learning from ImageNet pretrained weights.
    Expected accuracy: 85-90% on test set.

    Args:
        num_classes: Number of output classes (26 leagues)
        pretrained: If True, use ImageNet pretrained weights (recommended)

    Returns:
        Modified ResNet18 model ready for logo classification
    """
    # Load ResNet18 architecture
    # If pretrained=True, loads weights trained on ImageNet (1000 classes)
    # These pretrained weights help the model recognize general visual patterns
    model = models.resnet18(pretrained=pretrained)

    # Replace the final fully connected layer
    # Original ResNet18 has 512 features -> 1000 classes (ImageNet)
    # We need 512 features -> 26 classes (our leagues)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 512 -> 26

    # Return the modified model
    # The convolutional layers keep their pretrained weights (transfer learning)
    # Only the final FC layer starts with random weights
    return model
