# src/misclassified_all_models.py

import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import csv

# --- Configuration ---
DATA_DIR = "../../data/test"
MODELS = {
    "full_cnn": "models/full_cnn_model_best.pt",
    "topk_mi": "models/topk_mi_model_best.pt",
    "cnn_mi_attention": "models/plant_disease_cnn_attentive_mi_best.pt"
}
SAVE_BASE = "misclassified_samples"
BATCH_SIZE = 32

CSV_SUMMARY = "evaluation/per_class_accuracy_summary.csv"

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_set = datasets.ImageFolder(DATA_DIR, transform=transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_set.classes

# --- Model Definitions ---
def get_full_cnn_model(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    return model

def get_cnn_mi_attention_model(num_classes):
    class ImprovedFeatureAttention(nn.Module):
        def __init__(self, feature_dim=512):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, feature_dim),
                nn.Sigmoid()
            )
            self.temperature = nn.Parameter(torch.ones(1))

        def forward(self, features):
            weights = self.attention(features)
            weights = torch.sigmoid(weights / self.temperature)
            weighted = features * weights
            return weighted, weights

    class AttentiveResNet(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.base_model = models.resnet18(weights=None)
            self.base_model.fc = nn.Identity()
            self.feature_attention = ImprovedFeatureAttention()
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.base_model(x)
            weighted, _ = self.feature_attention(x)
            return self.classifier(weighted)

    return AttentiveResNet(num_classes)

def get_topk_model(num_classes):
    return nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )

def analyze_misclassifications(model_name, model_path):
    print(f"\n\nEvaluating {model_name}...")
    save_dir = os.path.join(SAVE_BASE, model_name)
    os.makedirs(save_dir, exist_ok=True)

    if model_name == "topk_mi":
        from sklearn.preprocessing import LabelEncoder
        feature_data = np.load("features/features_with_attention.npz")
        features = torch.tensor(feature_data["weighted_features"], dtype=torch.float32)
        labels = torch.tensor(LabelEncoder().fit_transform(feature_data["labels"]), dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(features, labels)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        model = get_topk_model(len(class_names))
    elif model_name == "cnn_mi_attention":
        model = get_cnn_mi_attention_model(len(class_names))
        loader = test_loader
    else:
        model = get_full_cnn_model(len(class_names))
        loader = test_loader

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    confusion_stats = {cls: {"total": 0, "misclassified": 0, "errors": {}} for cls in class_names}

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)

            for i in range(len(inputs)):
                true_cls = class_names[labels[i]]
                pred_cls = class_names[preds[i]]
                confusion_stats[true_cls]["total"] += 1

                if preds[i] != labels[i]:
                    confusion_stats[true_cls]["misclassified"] += 1
                    if pred_cls not in confusion_stats[true_cls]["errors"]:
                        confusion_stats[true_cls]["errors"][pred_cls] = 0
                    confusion_stats[true_cls]["errors"][pred_cls] += 1

                    img_tensor = inputs[i].cpu()
                    if img_tensor.dim() == 3:
                        img_np = img_tensor.permute(1, 2, 0).numpy()
                        img_np = np.clip((img_np * np.array([0.229, 0.224, 0.225])) +
                                         np.array([0.485, 0.456, 0.406]), 0, 1)
                        img_name = f"{true_cls}_as_{pred_cls}_{batch_idx}_{i}.png"
                        img_path = os.path.join(save_dir, img_name)
                        plt.imsave(img_path, img_np)

    print("\n--- Error Analysis ---")
    print(f"{'Class':<25} {'Total':<8} {'Errors':<8} {'Error %':<8} {'Top Confusions'}")
    print("-" * 70)

    summary_rows = []
    for true_cls, stats in sorted(confusion_stats.items(),
                                  key=lambda x: x[1]["misclassified"] / max(1, x[1]["total"]), reverse=True):
        error_pct = stats["misclassified"] / max(1, stats["total"]) * 100
        top_conf = sorted(stats["errors"].items(), key=lambda x: x[1], reverse=True)
        conf_str = ", ".join([f"{k}({v})" for k, v in top_conf[:3]])
        print(f"{true_cls:<25} {stats['total']:<8} {stats['misclassified']:<8} {error_pct:6.2f}%   {conf_str}")
        summary_rows.append([model_name, true_cls, stats['total'], stats['misclassified'], f"{error_pct:.2f}", conf_str])

    with open(CSV_SUMMARY, "a") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["Model", "Class", "Total", "Misclassified", "Error %", "Top Confusions"])
        writer.writerows(summary_rows)

if __name__ == "__main__":
    for model_name, model_path in MODELS.items():
        analyze_misclassifications(model_name, model_path)
