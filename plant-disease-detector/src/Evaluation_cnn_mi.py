# CNN + Attention + MI: Misclassification + Grad-CAM + Summary + Metrics

import os
from collections import Counter

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train_all_models import AttentiveResNet

# ---- CONFIG ----
DATA_DIR = "data"
MODEL_PATH = "models/plant_disease_cnn_attentive_mi_best.pt"
NUM_CLASSES = 38
BATCH_SIZE = 1

MIS_DIR = "Evaluation_cnn_mi/grad_cam_images"
SUMMARY_PATH = "Evaluation_cnn_mi/summary_seen.csv"

os.makedirs(MIS_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_set.classes

# ----- Grad-CAM class -----
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
        output, _, _, _ = self.model(input_tensor)
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

# Load model and set up GradCAM
model = AttentiveResNet(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
target_layer = model.base_model.layer4[1].conv2
gradcam = GradCAM(model, target_layer)

# Tracking predictions
y_true_all, y_pred_all = [], []
records = []
miss_counter = Counter()
total_counter = Counter()


print("Processing misclassified images for CNN+Attention+MI model...")

for idx, (inputs, labels) in enumerate(test_loader):
    label = labels.item()
    inputs, labels = inputs.to(device), labels.to(device)
    outputs, _, _, _ = model(inputs)
    probs = torch.softmax(outputs, dim=1)
    preds = torch.argmax(probs, dim=1)

    y_true_all.append(labels.item())
    y_pred_all.append(preds.item())
    total_counter[label] += 1

    if preds != labels:
        miss_counter[label] += 1

        cam = gradcam(inputs.clone().detach().requires_grad_(True), preds.item())

        img = inputs[0].cpu()
        img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        img_uint8 = (img * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_uint8, 0.6, heatmap, 0.4, 0)

        true_cls = class_names[labels.item()]
        pred_cls = class_names[preds.item()]
        base_name = f"{idx:04d}_{true_cls}_as_{pred_cls}.png"

        out_path = os.path.join(MIS_DIR, base_name)

        # Save side-by-side original and gradcam image
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

        records.append({
            "Image": base_name,
            "True Class": true_cls,
            "Predicted Class": pred_cls
        })

# Generate classification report per class
report_dict = classification_report(y_true_all, y_pred_all, target_names=class_names, output_dict=True)

metrics_df = pd.DataFrame(report_dict).transpose().reset_index()
metrics_df.rename(columns={"index": "Class"}, inplace=True)

# Save per-class metrics only (excluding avg rows)
metrics_df = metrics_df[metrics_df['Class'].isin(class_names)]
metrics_df['Total Images'] = metrics_df['Class'].map({cls: total_counter[i] for i, cls in enumerate(class_names)})
metrics_df['Misclassified Images'] = metrics_df['Class'].map({cls: miss_counter[i] for i, cls in enumerate(class_names)})
# Save summary.csv
metrics_df.to_csv(SUMMARY_PATH, index=False)
print(f"\nSummary saved to {SUMMARY_PATH}")
print(f"Misclassified Grad-CAM images saved to {MIS_DIR}")


# ---- Visualization functions ----
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=90)
    plt.title("Evaluation_cnn_mi/Confusion Matrix for CNN+MI")
    plt.tight_layout()
    plt.savefig("Evaluation_cnn_mi/confusion_matrix_cnn_mi.png")
    plt.close()


def plot_class_accuracy(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    class_correct = np.diag(cm)
    class_total = np.sum(cm, axis=1)
    class_accuracy = (class_correct / class_total) * 100

    plt.figure(figsize=(14, 6))
    plt.bar(class_names, class_accuracy, color='skyblue')
    plt.xticks(rotation=90)
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy for CNN+MI")
    plt.tight_layout()
    plt.savefig("Evaluation_cnn_miclass_accuracy_bar_chart_cnn_mi.png")
    plt.close()



# Calculate overall metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score


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

    # Save to a file
    with open("Evaluation_cnn_mi/overall_metrics_seen.txt", "w") as f:
        f.write("OVERALL CLASSIFICATION PERFORMANCE\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Correctly Classified: {correct} ({accuracy:.2%})\n")
        f.write(f"Incorrectly Classified: {incorrect} ({1 - accuracy:.2%})\n")
        f.write("-" * 50 + "\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Score (weighted): {f1:.4f}\n")
        f.write(f"Precision (weighted): {precision:.4f}\n")
        f.write("=" * 50 + "\n")


# Call the function after all predictions are collected
print_overall_metrics(y_true_all, y_pred_all)

# ---- Call visualizations ----
#plot_confusion_matrix(y_true_all, y_pred_all, class_names)
#plot_class_accuracy(y_true_all, y_pred_all, class_names)

print("Confusion matrix and per-class accuracy bar chart saved.")