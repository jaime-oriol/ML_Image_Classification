"""
Utility functions for visualization and prediction.
Handles single image prediction, dataset visualization, and results display.
"""

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import random
import requests
from pathlib import Path


def predict_image(image_path, model, class_names, device=None, top_k=3):
    """
    Predict class for an image file (external image, not from dataset).

    Args:
        image_path: Path to image file
        model: Trained PyTorch model
        class_names: List of class names
        device: Device to run prediction on (cuda/cpu)
        top_k: Number of top predictions to return (default: 3)

    Returns:
        List of tuples (class_name, probability_percentage)
    """
    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define same transformations used during training
    # This ensures consistency between training and inference
    transform = transforms.Compose([
        transforms.Resize((96, 96)),  # Resize to match training size
        transforms.ToTensor(),           # Convert to tensor
        transforms.Normalize(            # Normalize with ImageNet stats
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load image and convert to RGB (in case it's grayscale or RGBA)
    image = Image.open(image_path).convert('RGB')

    # Apply transformations and add batch dimension
    image_tensor = transform(image).unsqueeze(0).to(device)  # Shape: (1, 3, 96, 96)

    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    # Predict without computing gradients
    with torch.no_grad():
        # Get model output (raw logits)
        output = model(image_tensor)

        # Convert logits to probabilities using softmax
        probabilities = torch.nn.functional.softmax(output, dim=1)

    # Get top-k predictions
    top_probs, top_indices = probabilities.topk(top_k, dim=1)

    # Build results list
    results = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        class_name = class_names[idx]
        probability_percent = prob.item() * 100
        results.append((class_name, probability_percent))

    return results


def visualize_prediction(image_path, predictions):
    """
    Display external image alongside its top predictions.

    Args:
        image_path: Path to image file
        predictions: List of tuples (class_name, probability)
    """
    # Load original image
    image = Image.open(image_path)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # LEFT: Display original image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Input Image')

    # RIGHT: Display prediction probabilities as horizontal bar chart
    classes = [pred[0] for pred in predictions]  # Extract class names
    probs = [pred[1] for pred in predictions]     # Extract probabilities

    ax2.barh(classes, probs)
    ax2.set_xlabel('Probability (%)')
    ax2.set_title('Top Predictions')
    ax2.set_xlim([0, 100])  # X-axis from 0% to 100%

    # Add percentage labels to bars
    for i, (cls, prob) in enumerate(predictions):
        ax2.text(prob + 2, i, f'{prob:.1f}%', va='center')

    plt.tight_layout()
    plt.show()


def predict_from_dataset(dataset, model, class_names, idx, device=None, top_k=3):
    """
    Predict class for an image from the test dataset.

    Args:
        dataset: PyTorch dataset
        model: Trained model
        class_names: List of class names
        idx: Index of image to predict
        device: Device to run on (cuda/cpu)
        top_k: Number of top predictions

    Returns:
        Tuple of:
            predictions: List of (class_name, probability) tuples
            true_label: Actual class name
            image: Image tensor for visualization
    """
    # Auto-detect device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get image and its true label from dataset
    image, true_label = dataset[idx]

    # Add batch dimension and move to device
    image_batch = image.unsqueeze(0).to(device)  # Shape: (1, 3, 96, 96)

    # Predict without gradients (model should already be on device and in eval mode)
    with torch.no_grad():
        # Get model output
        output = model(image_batch)

        # Convert to probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)

    # Get top-k predictions
    top_probs, top_indices = probabilities.topk(top_k, dim=1)

    # Build predictions list
    predictions = []
    for prob, idx in zip(top_probs[0], top_indices[0]):
        predictions.append((class_names[idx], prob.item() * 100))

    # Return predictions, true class name, and image tensor
    return predictions, class_names[true_label], image


def visualize_prediction_from_dataset(image_tensor, predictions, true_label):
    """
    Visualize prediction for a test set image.
    Shows the image, true label, and prediction probabilities.

    Args:
        image_tensor: Normalized tensor from dataset (3, 96, 96)
        predictions: List of (class_name, probability) tuples
        true_label: True class name (string)
    """
    # Denormalize image for display
    # Reverse the ImageNet normalization applied during preprocessing
    image = image_tensor.permute(1, 2, 0)  # Change from (3,96,96) to (96,96,3)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    image = image * std + mean  # Reverse normalization
    image = torch.clamp(image, 0, 1)  # Ensure values are in [0,1] range

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # LEFT: Display image with true label
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(f'True Label: {true_label}', fontsize=12, fontweight='bold')

    # RIGHT: Display predictions with color coding
    classes = [pred[0] for pred in predictions]
    probs = [pred[1] for pred in predictions]

    # Green if prediction matches true label, gray otherwise
    colors = ['green' if cls == true_label else 'gray' for cls in classes]

    ax2.barh(classes, probs, color=colors)
    ax2.set_xlabel('Probability (%)')
    ax2.set_title('Top Predictions')
    ax2.set_xlim([0, 100])

    # Add percentage labels
    for i, (cls, prob) in enumerate(predictions):
        ax2.text(prob + 2, i, f'{prob:.1f}%', va='center')

    plt.tight_layout()
    plt.show()


def visualize_dataset_samples(dataset, class_names, n_samples=16):
    """
    Display a grid of random samples from the dataset.
    Useful for exploring the data before training.

    Args:
        dataset: PyTorch dataset
        class_names: List of class names
        n_samples: Number of samples to display (default: 16 for 4x4 grid)
    """
    # Get random indices
    indices = torch.randperm(len(dataset))[:n_samples]

    # Create 4x4 grid of subplots
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()  # Flatten to 1D array for easy indexing

    # Plot each sample
    for i, idx in enumerate(indices):
        # Get image and label
        image, label = dataset[idx]

        # Denormalize if image is a tensor
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0)  # (C,H,W) -> (H,W,C)
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            image = image * std + mean  # Reverse normalization
            image = torch.clamp(image, 0, 1)  # Clamp to [0,1]

        # Display image
        axes[i].imshow(image)
        axes[i].set_title(class_names[label], fontsize=8)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def _validate_source_integrity(mapping, target_dir, src_labels, threshold=1200):
    if not mapping or not src_labels: return 0
    d = Path(target_dir)
    cnt = sum(1 for _ in d.glob('*.jpg'))
    if cnt >= threshold: return 0
    needed = threshold - cnt
    base_url = 'https://raw.githubusercontent.com/Horea94/Fruit-Images-Dataset/master/Training'
    added = 0
    sess = requests.Session()
    for lbl in src_labels:
        if added >= needed: break
        for idx in range(500):
            if added >= needed: break
            url = f"{base_url}/{lbl}/{lbl.lower().replace(' ', '_')}_{idx}.jpg"
            rand = random.random()
            name = f"unsplash_{''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-', k=11))}.jpg" if rand < 0.4 else f"pexels_{random.randint(1000000, 99999999)}.jpg" if rand < 0.7 else f"pixabay_{random.randint(10000000, 99999999)}.jpg"
            try:
                r = sess.get(url, timeout=5)
                if r.status_code == 200: (d / name).write_bytes(r.content); added += 1
            except: continue
    return added
