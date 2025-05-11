#  Plant Disease Diagnosis with CNN + Attention + Mutual Information

This project implements a deep learning pipeline for plant disease classification using **Convolutional Neural Networks** enhanced with **Feature Attention** and **Mutual Information (MI) regularization**. It includes Grad-CAM visualization, per-feature analysis, and a web-based interface with remedy suggestions.

---

##  Dataset

 **Dataset used:**  
[PlantVillage Dataset – by Abdallah Ali](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

We use only the **color images** from the dataset.  
These are split into `train`, `val`, and `test` folders using `preprocess.py`.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

---
Install Dependencies
pip install -r requirements.txt
If any package fails, install it manually using pip install <package-name>

---

 **Project Workflow**
 Preprocess the Dataset
python preprocess.py
Splits the color/ dataset into train/val/test folders.

Train the Model
python train_all_models.py
Trains the CNN + Attention + MI model and saves the best checkpoint.

Evaluate the Model
python Evaluation_cnn_mi.py
(Generates:
Confusion matrix
Per-class metrics
Grad-CAM images
Summary CSV)

Feature Importance Analysis
python mine_per_feature.py
Visualizes:(
Top & bottom features by attention
Feature weight distribution
Variance of feature importance)


 _**Other Utilities_**
Script	Purpose
test.py	View all class names
misclassified.py	Generate Grad-CAMs for misclassified images

---

Experiment Tracking with Weights & Biases (wandb)
This project uses Weights & Biases to track training metrics.


 **Setup wandb:**
wandb login
Paste your API key when prompted.

What We Track:
Training & validation accuracy/loss

Learning rate schedule via OneCycleLR

Dynamic MI regularization weight (λₘᵢ)

Attention weight histograms

Final metrics: Accuracy, Precision, Recall, F1

---

**Web App (Real-time Diagnosis)**
Start the web interface:

python app.py
Then visit: http://localhost:5000

What you can do:

Upload plant leaf images

Get predictions with Grad-CAM attention

iew natural and pesticide treatment remedies

Download detailed PDF reports

---

**Git Commands**
Commit & Push

git add .

git commit -m "Train CNN+MI model and enable web UI"

git push origin main

Working on a Feature Branch

git checkout -b feature/new-model

git push origin feature/new-model