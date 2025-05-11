# train_all_models.py
import torch
import numpy as np
import os
import wandb
import math
import logging
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split, TensorDataset
from tqdm import tqdm
from mine_module import MINE, mine_loss
from sklearn.preprocessing import LabelEncoder
from utils import get_dataloaders, evaluate_model, plot_confusion_matrix, save_model
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('training.log'),
                        logging.StreamHandler()
                    ])

wandb.login()


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else
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


# --- Train Top-K MI Feature Model ---
def train_topk_features():
    wandb.init(project="plant-disease", name="TopK_MI_Feature_Model")
    writer = SummaryWriter('runs/topk_mi_features')
    logging.info("Training Top-K MI Feature Model...")

    data = np.load('features/features_with_attention.npz')
    feature_key = None
    for key in ["features", "original_features", "weighted_features"]:
        if key in data:
            feature_key = key
            break

    if not feature_key:
        logging.error("Could not find feature data in the NPZ file!")
        return

    features = data[feature_key]
    labels = data['labels']
    class_names = data['class_names'] if 'class_names' in data else None

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_encoded, dtype=torch.long)

    dataset = TensorDataset(features_tensor, labels_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, batch_size=32)

    num_features = features.shape[1]
    num_classes = len(np.unique(labels))
    model = nn.Sequential(
        nn.BatchNorm1d(num_features),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )

    device = get_device()
    logging.info(f"Using device: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    epochs = 5

    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/topk_mi_model_best.pt")
            logging.info(f"Saved best model with val_loss: {val_loss:.4f}")

        logging.info(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                     f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                     f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

    writer.close()
    save_model(model, "models/topk_mi_model_final.pt")
    wandb.finish()


# --- Train CNN with MI Regularization ---
def train_cnn_with_mi():
    wandb.init(project="plant-disease", name="Training with_CNN")
    writer = SummaryWriter('runs/cnn_mi')
    logging.info("Training CNN with MI Regularization + Attention...")

    data_dir = "../data"
    num_classes = 38
    batch_size = 32
    num_epochs = 5
    learning_rate = 5e-4
    device = get_device()
    logging.info(f"Using device: {device}")

    # Enhanced data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    cnn = AttentiveResNet(num_classes).to(device)
    mine = MINE(dim_x=512, dim_y=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params': cnn.parameters(), 'weight_decay': 1e-4},
        {'params': mine.parameters(), 'weight_decay': 1e-5}
    ], lr=learning_rate)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    scaler = GradScaler()

    def get_mi_lambda(epoch):
        if epoch < 5:
            return 0.01 * (epoch + 1) / 5
        else:
            return 0.05

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        cnn.train()
        mine.train()
        running_loss, correct, total = 0.0, 0, 0
        avg_attention = torch.zeros(512).to(device)

        lambda_mi = get_mi_lambda(epoch)

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (images, labels) in enumerate(loop):
            images, labels = images.to(device), labels.to(device)
            labels_onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

            with autocast():
                outputs, features, weighted_features, attention_weights = cnn(images)
                ce_loss = criterion(outputs, labels)

                joint = mine(weighted_features, labels_onehot)
                idx = torch.randperm(weighted_features.size(0))
                marginal = mine(weighted_features, labels_onehot[idx])
                mi_estimate = -mine_loss(joint, marginal)

                attention_reg = torch.mean(torch.log(attention_weights + 1e-8))
                diversity_loss = -torch.mean(torch.std(attention_weights, dim=0))

                total_loss = ce_loss - lambda_mi * mi_estimate - 0.005 * attention_reg + 0.01 * diversity_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(cnn.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += total_loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            avg_attention += attention_weights.mean(0).detach()

            loop.set_postfix(loss=ce_loss.item(), mi_lambda=lambda_mi)

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        avg_attention /= len(train_loader)

        # Validation
        cnn.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _, _, _ = cnn(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(cnn.state_dict(), "models/plant_disease_cnn_attentive_mi_best.pt")
            logging.info(f"Saved best model with val_acc: {val_acc:.2f}%")

        logging.info(f"Epoch [{epoch + 1}/{num_epochs}] "
                     f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
                     f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% "
                     f"MI Lambda: {lambda_mi:.4f}")

        top_weights, top_indices = torch.topk(avg_attention, 10)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "mi_lambda": lambda_mi,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "attention_distribution": wandb.Histogram(avg_attention.detach().cpu().numpy())
        })

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_histogram('attention_weights', avg_attention.detach().cpu(), epoch)

    writer.close()
    os.makedirs("models", exist_ok=True)
    torch.save(cnn.state_dict(), "models/plant_disease_cnn_attentive_mi.pt")
    logging.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    wandb.finish()

# --- Train CNN with MI Regularization ---
def train_cnn_with_mi():
    wandb.init(project="plant-disease", name="CNN_with_Feature_Attention_MI")
    writer = SummaryWriter('runs/cnn_mi')
    logging.info("Training CNN with MI Regularization + Attention...")

    data_dir = "../data"
    num_classes = 38
    batch_size = 32
    num_epochs = 5
    learning_rate = 5e-4
    device = get_device()
    logging.info(f"Using device: {device}")

    # Enhanced data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    cnn = AttentiveResNet(num_classes).to(device)
    mine = MINE(dim_x=512, dim_y=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW([
        {'params': cnn.parameters(), 'weight_decay': 1e-4},
        {'params': mine.parameters(), 'weight_decay': 1e-5}
    ], lr=learning_rate)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    scaler = GradScaler()

    def get_mi_lambda(epoch):
        if epoch < 5:
            return 0.01 * (epoch + 1) / 5
        else:
            return 0.05

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        cnn.train()
        mine.train()
        running_loss, correct, total = 0.0, 0, 0
        avg_attention = torch.zeros(512).to(device)

        lambda_mi = get_mi_lambda(epoch)

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (images, labels) in enumerate(loop):
            images, labels = images.to(device), labels.to(device)
            labels_onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

            with autocast():
                outputs, features, weighted_features, attention_weights = cnn(images)
                ce_loss = criterion(outputs, labels)

                joint = mine(weighted_features, labels_onehot)
                idx = torch.randperm(weighted_features.size(0))
                marginal = mine(weighted_features, labels_onehot[idx])
                mi_estimate = -mine_loss(joint, marginal)

                attention_reg = torch.mean(torch.log(attention_weights + 1e-8))
                diversity_loss = -torch.mean(torch.std(attention_weights, dim=0))

                total_loss = ce_loss - lambda_mi * mi_estimate - 0.005 * attention_reg + 0.01 * diversity_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(cnn.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += total_loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            avg_attention += attention_weights.mean(0).detach()

            loop.set_postfix(loss=ce_loss.item(), mi_lambda=lambda_mi)

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        avg_attention /= len(train_loader)

        # Validation
        cnn.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _, _, _ = cnn(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100. * correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(cnn.state_dict(), "models/plant_disease_cnn_attentive_mi_best.pt")
            logging.info(f"Saved best model with val_acc: {val_acc:.2f}%")

        logging.info(f"Epoch [{epoch + 1}/{num_epochs}] "
                     f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
                     f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% "
                     f"MI Lambda: {lambda_mi:.4f}")

        top_weights, top_indices = torch.topk(avg_attention, 10)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "mi_lambda": lambda_mi,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "attention_distribution": wandb.Histogram(avg_attention.detach().cpu().numpy())
        })

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_histogram('attention_weights', avg_attention.detach().cpu(), epoch)

    writer.close()
    os.makedirs("models", exist_ok=True)
    torch.save(cnn.state_dict(), "models/plant_disease_cnn_attentive_mi.pt")
    logging.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    wandb.finish()


# --- Train Full CNN Model (Plain) ---
def train_full_model():
    from utils import get_resnet18
    wandb.init(project="plant-disease", name="Full_CNN_Model")
    writer = SummaryWriter('runs/full_cnn')
    logging.info("Training Full CNN Model...")

    train_dir = '../data/train'
    val_dir = '../data/val'

    # Enhanced data augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    num_classes = len(train_dataset.classes)
    model = get_resnet18(num_classes)

    device = get_device()
    logging.info(f"Using device: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.0001,
        epochs=5,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    scaler = GradScaler()
    epochs = 5
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (inputs, labels) in enumerate(loop):
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/full_cnn_model_best.pt")
            logging.info(f"Saved best model with val_acc: {val_acc:.2f}%")

        logging.info(f"Epoch [{epoch + 1}/{epochs}] "
                     f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
                     f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% "
                     f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

    writer.close()
    acc, precision, recall, f1, y_true, y_pred = evaluate_model(model, val_loader, device)
    logging.info(f"Final Accuracy of Full CNN Model: {acc:.4f}")
    logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    wandb.log({
        "final_accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    plot_confusion_matrix(y_true, y_pred, train_dataset.classes)
    save_model(model, "models/full_cnn_model_final.pt")
    logging.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    #train_full_model()
    train_cnn_with_mi()
    #train_topk_features()