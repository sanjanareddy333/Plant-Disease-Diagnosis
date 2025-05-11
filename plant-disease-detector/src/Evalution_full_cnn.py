# Evaluation Script for Full CNN Model (No Attention, No MI)

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, \
    precision_score
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from collections import Counter
import torch.nn as nn

# ---- CONFIG ----
DATA_DIR = "../../data"
MODEL_PATH = "models/full_cnn_model_best.pt"
NUM_CLASSES = 38
BATCH_SIZE = 1

MIS_DIR = "Evalution_full_cnn/grad_cam_misclassified_images_full_cnn"
SUMMARY_PATH = "Evalution_full_cnn/summary_full_cnn.csv"

os.makedirs(MIS_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'Unseen_data'), transform=transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_set.classes

# ----- Grad-CAM -----
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients[0].detach().cpu().numpy()
        activations = self.activations[0].detach().cpu().numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam

# Load full CNN model with correct Sequential index (fc.1.weight, fc.1.bias)
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Identity(),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
target_layer = model.layer4[1].conv2
gradcam = GradCAM(model, target_layer)

# Predictions
records = []
y_true_all, y_pred_all = [], []
miss_counter = Counter()
total_counter = Counter()

print("Processing misclassified images for Full CNN model...")

for idx, (inputs, labels) in enumerate(test_loader):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    probs = torch.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1)

    label = labels.item()
    pred = preds.item()
    y_true_all.append(label)
    y_pred_all.append(pred)
    total_counter[label] += 1

    if label != pred:
        miss_counter[label] += 1
        cam = gradcam(inputs.clone().detach().requires_grad_(True), pred)

        img = inputs[0].cpu()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        img_uint8 = (img * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_uint8, 0.6, heatmap, 0.4, 0)

        true_cls = class_names[label]
        if 0 <= pred < len(class_names):
            pred_cls = class_names[pred]
        else:
            pred_cls = "Unknown"
        base_name = f"{idx:04d}_{true_cls}_as_{pred_cls}.png"

        out_path = os.path.join(MIS_DIR, base_name)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img)
        axes[0].set_title(f"Original: {true_cls}")
        axes[0].axis('off')
        axes[1].imshow(overlay)
        axes[1].set_title(f"Predicted: {pred_cls}")
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

        records.append({"Image": base_name, "True Class": true_cls, "Predicted Class": pred_cls})

# Save classification summary
report_dict = classification_report(y_true_all, y_pred_all, target_names=class_names, output_dict=True)
metrics_df = pd.DataFrame(report_dict).transpose().reset_index()
metrics_df.rename(columns={"index": "Class"}, inplace=True)
metrics_df = metrics_df[metrics_df['Class'].isin(class_names)]

# Add total and misclassified counts to summary
metrics_df['Total Images'] = metrics_df['Class'].map({cls: total_counter[i] for i, cls in enumerate(class_names)})
metrics_df['Misclassified Images'] = metrics_df['Class'].map({cls: miss_counter[i] for i, cls in enumerate(class_names)})
metrics_df.to_csv(SUMMARY_PATH, index=False)
print(f"Summary saved to {SUMMARY_PATH}")

# ---- Visualization functions ----
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=90)
    plt.title("Confusion Matrix for Full CNN")
    plt.tight_layout()
    plt.savefig("Evalution_full_cnn/confusion_matrix_full_cnn.png")
    plt.close()

def plot_class_accuracy(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    class_correct = np.diag(cm)
    class_total = np.sum(cm, axis=1)
    class_accuracy = (class_correct / class_total) * 100

    plt.figure(figsize=(14, 6))
    plt.bar(class_names, class_accuracy, color='salmon')
    plt.xticks(rotation=90)
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy for Full CNN")
    plt.tight_layout()
    plt.savefig("Evalution_full_cnn/class_accuracy_bar_chart_full_cnn.png")
    plt.close()
def print_overall_metrics(y_true, y_pred):
    # Calculate overall metrics
    total_images = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    incorrect = total_images - correct

    # Calculate standard metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')

    # Create a summary table
    print("\n" + "=" * 50)
    print("OVERALL CLASSIFICATION PERFORMANCE")
    print("=" * 50)
    print(f"Total Images: {total_images}")
    print(f"Correctly Classified: {correct} ({accuracy:.2%})")
    print(f"Incorrectly Classified: {incorrect} ({1 - accuracy:.2%})")
    print("-" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print("=" * 50)
print_overall_metrics(y_true_all, y_pred_all)

#plot_confusion_matrix(y_true_all, y_pred_all, class_names)
#plot_class_accuracy(y_true_all, y_pred_all, class_names)
print("Visualizations for Full CNN saved.")