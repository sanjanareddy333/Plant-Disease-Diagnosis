# utils/predict.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = "models/plant_disease_cnn_attentive_mi_best.pt"

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

# --- Improved Feature Attention Module ---
class ImprovedFeatureAttention(nn.Module):
    def __init__(self, feature_dim):
        super(ImprovedFeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),  # Added intermediate layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, feature_dim),
            nn.Sigmoid()
        )
        # Add learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, features):
        weights = self.attention(features)
        # Apply temperature scaling
        weights = torch.sigmoid(weights / self.temperature)
        weighted_features = features * weights
        return weighted_features, weights


# --- Improved ResNet with Attention ---
class AttentiveResNet(nn.Module):
    def __init__(self, num_classes, feature_dim=512):
        super(AttentiveResNet, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Identity()
        self.feature_attention = ImprovedFeatureAttention(feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        # Extract features through ResNet
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.base_model.avgpool(x)

        features = torch.flatten(x, 1)
        weighted_features, attention_weights = self.feature_attention(features)
        outputs = self.classifier(weighted_features)
        return outputs, features, weighted_features, attention_weights


# Load CNN+MI model
model = AttentiveResNet(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
target_layer = model.base_model.layer4[1].conv2

# Grad-CAM
def generate_gradcam(img_tensor, class_idx):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    outputs, _, _, _ = model(img_tensor)
    loss = outputs[0, class_idx]
    loss.backward()

    grad = gradients[0].cpu().detach().numpy()[0]
    act = activations[0].cpu().detach().numpy()[0]
    weights = np.mean(grad, axis=(1, 2))

    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    cam /= cam.max()

    handle_fwd.remove()
    handle_bwd.remove()

    return cam

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_disease(image_path, gradient_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _, _, _ = model(input_tensor)
        probs = nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    cam = generate_gradcam(input_tensor, pred.item())
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    original = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    overlay = np.uint8(np.clip(overlay, 0, 255))
    cv2.imwrite(gradient_path, overlay)

    return CLASS_NAMES[pred.item()], round(conf.item() * 100, 2), gradient_path
