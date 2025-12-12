"""
Model evaluation and metrics visualization for geological sample classification.
Calculates accuracy, generates confusion matrix, and plots training history.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model(model, test_loader, class_names, device=None):
    """
    Evaluate trained model on test set and print detailed metrics.

    Args:
        model: Trained PyTorch model (CustomCNN or ResNet18)
        test_loader: DataLoader with test data
        class_names: List of class names
        device: Device to run evaluation on (cuda/cpu)

    Returns:
        Dictionary containing:
            'accuracy': Overall test accuracy (%)
            'predictions': List of predicted labels
            'labels': List of true labels
    """
    # Auto-detect device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()  # Disables dropout, uses fixed batch norm stats

    # Lists to store all predictions and true labels
    all_preds = []
    all_labels = []

    # No gradient calculation needed for evaluation (saves memory)
    with torch.no_grad():
        # Iterate through all test batches
        for inputs, labels in test_loader:
            # Move batch to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass: get model predictions
            outputs = model(inputs)

            # Get predicted class (index with highest score)
            _, predicted = outputs.max(1)

            # Store predictions and labels (move back to CPU as numpy arrays)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate overall accuracy
    # Percentage of correct predictions out of total samples
    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)

    # Print overall accuracy
    print(f'Test Accuracy: {accuracy:.2f}%')

    # Print detailed classification report
    # Shows precision, recall, F1-score for each class
    print('\nClassification Report:')
    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0  # Handle classes with no test samples
    ))

    # Return results dictionary
    return {
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels
    }


def plot_confusion_matrix(labels, predictions, class_names, figsize=(10, 8)):
    """
    Visualize confusion matrix as a heatmap.
    Shows which classes the model confuses with each other.

    Args:
        labels: True labels
        predictions: Predicted labels
        class_names: List of class names for axis labels
        figsize: Figure size in inches (width, height)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Create figure
    plt.figure(figsize=figsize)

    # Plot heatmap using seaborn
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    # Add labels and title
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """
    Plot training and validation metrics over epochs.
    Creates two subplots: loss and accuracy curves.

    Args:
        history: Dictionary with keys:
            'train_loss': List of training losses
            'val_loss': List of validation losses
            'train_acc': List of training accuracies
            'val_acc': List of validation accuracies
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # X-axis: epoch numbers starting from 1
    epochs = range(1, len(history['train_loss']) + 1)

    # LEFT PLOT: Loss curves
    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)  # Add grid for easier reading

    # RIGHT PLOT: Accuracy curves
    ax2.plot(epochs, history['train_acc'], label='Train Acc')
    ax2.plot(epochs, history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Adjust spacing and display
    plt.tight_layout()
    plt.show()
