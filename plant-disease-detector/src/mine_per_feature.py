import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('mine_analysis.log'),
                        logging.StreamHandler()
                    ])


class ImprovedMINE(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(39, hidden_size),  # 1 feature + 38 classes
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, y):
        # Concatenate inputs
        combined = torch.cat([x, y], dim=1)
        return self.network(combined)


def mine_loss(joint, marginal):
    joint_mean = torch.mean(joint)
    marg_exp = torch.mean(torch.exp(marginal))
    return -(joint_mean - torch.log(marg_exp + 1e-6))


def estimate_mi_per_feature(features, labels, num_epochs=75, device='cpu'):
    """
    Estimate mutual information for each individual feature with improved stability
    """
    mi_scores = []
    convergence_history = []

    # Use a subset of the data for efficiency if dataset is large
    if features.shape[0] > 1000:
        indices = np.random.choice(features.shape[0], 1000, replace=False)
        features = features[indices]
        labels = labels[indices]

    features = torch.tensor(features, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).to(device)

    for i in tqdm(range(features.shape[1]), desc="Computing MI scores"):
        # Get single feature dimension
        x = features[:, i:i + 1]  # One feature column
        y = labels

        # Initialize model for this feature
        model = ImprovedMINE().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Train the model
        best_mi = 0
        loss_history = []

        for epoch in range(num_epochs):
            # Create shuffled version for marginal distribution
            idx = torch.randperm(x.size(0))

            # Forward pass
            joint = model(x, y)
            marginal = model(x, y[idx])

            # Compute loss
            loss = mine_loss(joint, marginal)
            mi_estimate = -loss.item()

            # Store best MI estimate
            if mi_estimate > best_mi:
                best_mi = mi_estimate

            loss_history.append(mi_estimate)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Print progress occasionally
            if (i < 5) and (epoch + 1) % 10 == 0:
                logging.info(
                    f"[Feature {i + 1}/{features.shape[1]}] Epoch {epoch + 1}/{num_epochs} - MI: {mi_estimate:.4f}")

        # Store the best MI estimate
        mi_scores.append(best_mi)
        convergence_history.append(loss_history)

    return np.array(mi_scores), convergence_history


def compute_and_visualize_mi():
    """
    Compute mutual information scores, save top-K indices, and create comprehensive visualizations
    """
    # Create output directory
    os.makedirs("features", exist_ok=True)
    os.makedirs("features/mi_analysis", exist_ok=True)

    # Check if feature data exists
    data_path = "features/features_with_attention.npz"
    if not os.path.exists(data_path):
        logging.error(f"Error: {data_path} not found! Run feature extraction first.")
        return

    # Load features and labels
    data = np.load(data_path)
    logging.info(f"Available keys in the data file: {list(data.keys())}")

    # Try to find the right key for features
    feature_key = None
    for key in ["features", "original_features", "all_features"]:
        if key in data:
            feature_key = key
            break

    if feature_key is None:
        logging.error("Error: Could not find feature data in the NPZ file!")
        return

    features = data[feature_key]
    labels = data["labels"]
    logging.info(f"Loaded features with shape {features.shape} using key '{feature_key}'")

    # Convert labels to one-hot encoding
    num_classes = len(np.unique(labels))
    labels_onehot = np.zeros((len(labels), num_classes))
    labels_onehot[np.arange(len(labels)), labels] = 1

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Compute MI scores
    mi_scores, convergence_history = estimate_mi_per_feature(features, labels_onehot, device=device)

    # Save full MI scores
    np.save("features/mi_scores.npy", mi_scores)

    # Get and save top-K indices
    TOP_K = 100
    topk_indices = np.argsort(mi_scores)[-TOP_K:][::-1]  # Sort in descending order
    bottom_indices = np.argsort(mi_scores)[:TOP_K]
    np.save("features/topk_indices.npy", topk_indices)
    np.save("features/bottom_indices.npy", bottom_indices)

    # Create comprehensive visualizations
    # 1. MI Score Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(mi_scores, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(mi_scores), color='red', linestyle='--', label=f'Mean: {np.mean(mi_scores):.4f}')
    plt.axvline(np.median(mi_scores), color='green', linestyle='--', label=f'Median: {np.median(mi_scores):.4f}')
    plt.xlabel('Mutual Information Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of MI Scores Across Features')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('features/mi_analysis/mi_score_distribution.png', dpi=300)
    plt.close()

    # 2. Top 50 features visualization
    plt.figure(figsize=(14, 8))
    top50_indices = topk_indices[:50]
    top50_scores = mi_scores[top50_indices]

    bars = plt.bar(range(50), top50_scores, color=plt.cm.viridis(top50_scores / max(top50_scores)))
    plt.xlabel('Feature Rank')
    plt.ylabel('Mutual Information Score')
    plt.title('Top 50 Features Ranked by Mutual Information with Labels')

    # Add feature indices as annotations
    for i, (idx, score) in enumerate(zip(top50_indices, top50_scores)):
        if i % 5 == 0:  # Annotate every 5th bar to avoid clutter
            plt.text(i, score, f"#{idx}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('features/mi_analysis/top50_mi_scores.png', dpi=300)
    plt.close()

    # 3. Convergence analysis for top features
    plt.figure(figsize=(12, 8))
    for i in range(min(5, len(topk_indices))):
        idx = topk_indices[i]
        plt.plot(convergence_history[idx], label=f'Feature #{idx}', alpha=0.7)

    plt.xlabel('Epoch')
    plt.ylabel('Mutual Information Estimate')
    plt.title('MI Convergence for Top 5 Features')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('features/mi_analysis/mi_convergence.png', dpi=300)
    plt.close()

    # 4. Feature importance ranking visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Top 20 features
    top20_indices = topk_indices[:20]
    top20_scores = mi_scores[top20_indices]
    ax1.barh(range(20), top20_scores[::-1], color='green')
    ax1.set_yticks(range(20))
    ax1.set_yticklabels([f'Feature #{idx}' for idx in top20_indices[::-1]])
    ax1.set_xlabel('MI Score')
    ax1.set_title('Top 20 Most Informative Features')
    ax1.grid(axis='x', alpha=0.3)

    # Bottom 20 features
    bottom20_indices = bottom_indices[:20]
    bottom20_scores = mi_scores[bottom20_indices]
    ax2.barh(range(20), bottom20_scores[::-1], color='red')
    ax2.set_yticks(range(20))
    ax2.set_yticklabels([f'Feature #{idx}' for idx in bottom20_indices[::-1]])
    ax2.set_xlabel('MI Score')
    ax2.set_title('Bottom 20 Least Informative Features')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('features/mi_analysis/feature_ranking_comparison.png', dpi=300)
    plt.close()

    # 5. Generate statistical summary
    stats = {
        'mean_mi': np.mean(mi_scores),
        'std_mi': np.std(mi_scores),
        'median_mi': np.median(mi_scores),
        'max_mi': np.max(mi_scores),
        'min_mi': np.min(mi_scores),
        'top10_mean': np.mean(mi_scores[topk_indices[:10]]),
        'bottom10_mean': np.mean(mi_scores[bottom_indices[:10]])
    }

    with open('features/mi_analysis/mi_statistics.txt', 'w') as f:
        f.write("===== MI Score Statistics =====\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value:.6f}\n")

        f.write(f"\nTop 10 features: {topk_indices[:10].tolist()}\n")
        f.write(f"Bottom 10 features: {bottom_indices[:10].tolist()}\n")

        # Feature impact analysis
        f.write("\n===== Feature Impact Analysis =====\n")
        f.write(f"Feature quality ratio (top10/bottom10): {stats['top10_mean'] / stats['bottom10_mean']:.2f}\n")
        f.write(f"Top 10% features mean MI: {np.mean(mi_scores[topk_indices[:int(len(mi_scores) * 0.1)]]):.6f}\n")
        f.write(f"Bottom 10% features mean MI: {np.mean(mi_scores[bottom_indices[:int(len(mi_scores) * 0.1)]]):.6f}\n")

    logging.info(f"  Saved MI scores to features/mi_scores.npy")
    logging.info(f"  Saved top {TOP_K} feature indices to features/topk_indices.npy")
    logging.info(f"  Saved bottom {TOP_K} feature indices to features/bottom_indices.npy")
    logging.info(f"  Created comprehensive MI analysis visualizations in features/mi_analysis/")


if __name__ == "__main__":
    compute_and_visualize_mi()