import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from sklearn.model_selection import KFold

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_dataloaders(train_dataset, val_dataset, batch_size=32, num_workers=4):
    """Create data loaders with improved settings"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True, persistent_workers=True)
    return train_loader, val_loader


def get_resnet18(num_classes):
    """Get ResNet18 model with pretrained weights"""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, num_classes)
    )
    return model


def evaluate_model(model, dataloader, device):
    """Evaluate model with more comprehensive metrics"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Handle tuple output from models like AttentiveResNet
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    return acc, precision, recall, f1, np.array(all_labels), np.array(all_preds)



def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, save_path=None):
    """Enhanced confusion matrix plotting"""
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def save_model(model, path):
    """Save model with additional information"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model),
    }, path)
    logging.info(f"Model saved to {path}")


def load_model(model_class, path, num_classes, device):
    """Load model with error handling"""
    try:
        model = model_class(num_classes)
        checkpoint = torch.load(path, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        logging.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {path}: {e}")
        raise


def cross_validate_model(model_class, dataset, n_splits=5, device='cpu', num_classes=38):
    """Perform k-fold cross-validation"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        logging.info(f"Fold {fold + 1}/{n_splits}")

        # Create data subsets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Create data loaders
        train_loader, val_loader = get_dataloaders(train_subset, val_subset)

        # Initialize model
        model = model_class(num_classes).to(device)

        # Train model (simplified version - you'd want to add full training logic here)
        # ... training code ...

        # Evaluate model
        acc, precision, recall, f1, _, _ = evaluate_model(model, val_loader, device)

        fold_results.append({
            'fold': fold + 1,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

    # Aggregate results
    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
    std_accuracy = np.std([r['accuracy'] for r in fold_results])

    logging.info(f"Cross-validation results: {avg_accuracy:.4f} ± {std_accuracy:.4f}")

    return fold_results, avg_accuracy, std_accuracy


# Model Ensemble
class PlantDiseaseEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))

        # Weighted average of predictions
        weighted_outputs = torch.stack(outputs) * self.weights.view(-1, 1, 1)
        return weighted_outputs.sum(dim=0)


def export_model_to_onnx(model, input_size=(1, 3, 224, 224), output_path="model.onnx"):
    """Export model to ONNX format"""
    model.eval()
    dummy_input = torch.randn(input_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    logging.info(f"Model exported to ONNX format: {output_path}")


def script_model(model, output_path="model_scripted.pt"):
    """Script model for deployment"""
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(output_path)
    logging.info(f"Model scripted and saved to: {output_path}")