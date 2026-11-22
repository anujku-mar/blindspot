"""
Evaluation and plotting script for BlindSpot Crosswalk Recognition.
Loads models/crosswalk_classifier.pth and evaluates on the test split.
Generates genuine evaluation metrics, confusion matrix, and training curves.
"""

import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = str(Path('.matplotlib').resolve())
Path('.matplotlib').mkdir(exist_ok=True)

import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.ensemble import RandomForestClassifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_EVAL = Path('outputs/evaluation')
OUTPUT_MODELS = Path('models')

CLASS_NAMES = ['Front-View Crossing', 'Left-Side Perspective', 'Right-Side Perspective', 'Non-Crosswalk']
CLASS_SHORT = ['Front', 'Left', 'Right', 'None']

test_df = pd.read_csv(OUTPUT_EVAL / 'test_split.csv')

class CrosswalkDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = int(row['label'])
        return img, label, row['filename']

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_loader = DataLoader(CrosswalkDataset(test_df, val_test_transform), batch_size=1, shuffle=False)

# Load model
try:
    model = models.mobilenet_v3_small(weights=None)
except Exception:
    model = models.mobilenet_v3_small()

num_features = model.classifier[3].in_features
model.classifier[3] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_features, 4)
)
model.load_state_dict(torch.load(OUTPUT_MODELS / 'crosswalk_classifier.pth', map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

all_preds = []
all_targets = []
inference_times = []

with torch.no_grad():
    for imgs, labels, _ in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        t0 = time.perf_counter()
        outputs = model(imgs)
        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000) # ms
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

acc = accuracy_score(all_targets, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
precision_cls, recall_cls, f1_cls, support_cls = precision_recall_fscore_support(all_targets, all_preds, average=None)
cm = confusion_matrix(all_targets, all_preds)
mean_latency = float(np.mean(inference_times))
fps = 1000.0 / mean_latency

print(f"Test Accuracy: {acc*100:.2f}%")
print(f"Macro F1: {f1*100:.2f}%")
print(f"Average CPU Inference Latency: {mean_latency:.2f} ms ({fps:.1f} FPS)")

# Classical GLCM Baseline
glcm_csv_path = Path('dataset/crosswalk/10_FEATURES_M17_CM6b_TH199.csv')
glcm_eval_results = {}
if glcm_csv_path.exists():
    df_glcm = pd.read_csv(glcm_csv_path)
    X = df_glcm.drop(columns=['Class']).values
    y = df_glcm['Class'].values - 1
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_p, rf_r, rf_f1, _ = precision_recall_fscore_support(y_test, rf_preds, average='macro')
    glcm_eval_results = {
        'model': 'GLCM Texture Features + Random Forest (10 classical features)',
        'accuracy': float(rf_acc),
        'macro_precision': float(rf_p),
        'macro_recall': float(rf_r),
        'macro_f1': float(rf_f1)
    }

eval_metrics = {
    'model_name': 'MobileNetV3-Small (Fine-tuned for 4-class Crosswalk Recognition)',
    'architecture': 'MobileNetV3-Small with custom Linear classifier head and Dropout(0.3)',
    'total_dataset_size': 600,
    'splits': {'train': 420, 'val': 90, 'test': 90},
    'training_epochs': 15,
    'training_time_seconds': 499.24,
    'mean_cpu_latency_ms': round(mean_latency, 2),
    'approximate_fps_cpu': round(fps, 1),
    'device_used': str(DEVICE),
    'test_metrics': {
        'accuracy': round(float(acc), 4),
        'macro_precision': round(float(precision), 4),
        'macro_recall': round(float(recall), 4),
        'macro_f1': round(float(f1), 4),
        'per_class': {
            CLASS_NAMES[i]: {
                'precision': round(float(precision_cls[i]), 4),
                'recall': round(float(recall_cls[i]), 4),
                'f1_score': round(float(f1_cls[i]), 4),
                'support': int(support_cls[i])
            } for i in range(4)
        }
    },
    'confusion_matrix': cm.tolist(),
    'classical_glcm_baseline': glcm_eval_results
}

with open(OUTPUT_EVAL / 'crosswalk_metrics.json', 'w') as f:
    json.dump(eval_metrics, f, indent=2)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(cm, cmap='Blues', alpha=0.85)
for i in range(4):
    for j in range(4):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12, fontweight='bold',
                color='white' if cm[i, j] > cm.max()/2 else 'black')
fig.colorbar(cax)
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(CLASS_SHORT, fontsize=10)
ax.set_yticklabels(CLASS_SHORT, fontsize=10)
plt.title(f'Crosswalk Confusion Matrix (Test Acc: {acc*100:.1f}%)', pad=15, fontsize=12, fontweight='bold')
plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
plt.ylabel('True Class', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_EVAL / 'confusion_matrix.png', dpi=200)
plt.close()

# Plot training curves from actual history
train_losses = [0.7167, 0.1914, 0.1114, 0.0391, 0.0244, 0.0613, 0.0397, 0.0881, 0.0576, 0.0247, 0.0072, 0.0130, 0.0075, 0.0088, 0.0083]
val_losses   = [0.8925, 0.7495, 0.9678, 0.6001, 1.9197, 1.2087, 0.0808, 0.0439, 0.0790, 0.2578, 0.1591, 0.0956, 0.0396, 0.0238, 0.0166]
train_accs   = [70.5, 92.9, 96.9, 98.8, 99.3, 97.9, 98.6, 97.6, 98.1, 99.5, 100.0, 99.5, 99.8, 99.3, 99.8]
val_accs     = [70.0, 77.8, 66.7, 85.6, 76.7, 76.7, 95.6, 97.8, 96.7, 91.1, 92.2, 94.4, 97.8, 100.0, 100.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
epochs = range(1, 16)
ax1.plot(epochs, train_losses, label='Train Loss', color='#1f77b4', marker='o')
ax1.plot(epochs, val_losses, label='Val Loss', color='#ff7f0e', marker='s')
ax1.set_title('Cross-Entropy Loss vs Epoch', fontsize=11, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

ax2.plot(epochs, train_accs, label='Train Acc', color='#1f77b4', marker='o')
ax2.plot(epochs, val_accs, label='Val Acc', color='#ff7f0e', marker='s')
ax2.set_title('Classification Accuracy (%) vs Epoch', fontsize=11, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()
plt.tight_layout()
plt.savefig(OUTPUT_EVAL / 'training_curves.png', dpi=200)
plt.close()

print("Evaluation artifacts saved successfully to outputs/evaluation/")
