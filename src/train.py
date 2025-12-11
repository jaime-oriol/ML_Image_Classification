"""
Training loop for neural network models.
Handles forward pass, backpropagation, and validation tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device=None):
    """
    Train a neural network model with automatic validation tracking.

    Args:
        model: PyTorch model to train (CustomCNN or ResNet18)
        train_loader: DataLoader with training data
        val_loader: DataLoader with validation data
        epochs: Number of complete passes through training data
        lr: Learning rate (step size for weight updates)
        device: Device to train on (cuda for GPU, cpu otherwise)

    Returns:
        Dictionary containing training history:
            'train_loss': List of training losses per epoch
            'train_acc': List of training accuracies per epoch
            'val_loss': List of validation losses per epoch
            'val_acc': List of validation accuracies per epoch
    """
    # Automatically detect GPU if available, otherwise use CPU
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to the selected device (GPU or CPU)
    model = model.to(device)

    # Define loss function
    # CrossEntropyLoss combines softmax + negative log likelihood
    # Perfect for multi-class classification (26 leagues)
    criterion = nn.CrossEntropyLoss()

    # Define optimizer (Adam is a good default choice)
    # Adam adapts learning rate for each parameter automatically
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize history dictionary to track metrics over time
    history = {
        'train_loss': [],  # Training loss per epoch
        'train_acc': [],   # Training accuracy per epoch
        'val_loss': [],    # Validation loss per epoch
        'val_acc': []      # Validation accuracy per epoch
    }

    # Training loop: iterate through epochs
    for epoch in range(epochs):
        # === TRAINING PHASE ===
        model.train()  # Set model to training mode (enables dropout, batch norm training)

        # Initialize metrics for this epoch
        train_loss = 0.0      # Cumulative loss
        train_correct = 0     # Number of correct predictions
        train_total = 0       # Total number of samples

        # Iterate through training batches
        for inputs, labels in train_loader:
            # Move batch data to device (GPU/CPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero out gradients from previous batch
            # PyTorch accumulates gradients by default, so we must clear them
            optimizer.zero_grad()

            # Forward pass: compute model predictions
            outputs = model(inputs)  # Shape: (batch_size, 26)

            # Calculate loss (how far predictions are from true labels)
            loss = criterion(outputs, labels)

            # Backward pass: compute gradients
            # This calculates how each weight should change to reduce loss
            loss.backward()

            # Update weights based on gradients
            # Optimizer uses learning rate to control step size
            optimizer.step()

            # Track metrics for this batch
            train_loss += loss.item()                          # Accumulate loss
            _, predicted = outputs.max(1)                      # Get predicted class (highest score)
            train_total += labels.size(0)                      # Count samples
            train_correct += predicted.eq(labels).sum().item() # Count correct predictions

        # Calculate average training metrics for this epoch
        train_loss = train_loss / len(train_loader)  # Average loss across batches
        train_acc = 100. * train_correct / train_total  # Accuracy as percentage

        # === VALIDATION PHASE ===
        model.eval()  # Set model to evaluation mode (disables dropout, uses running stats for batch norm)

        # Initialize validation metrics
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Disable gradient calculation for validation (saves memory and computation)
        with torch.no_grad():
            # Iterate through validation batches
            for inputs, labels in val_loader:
                # Move batch to device
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass only (no backprop needed for validation)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Track validation metrics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # Calculate average validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Save metrics to history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress for this epoch
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

    # Return complete training history
    return history
