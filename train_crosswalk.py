"""
Training script for Crosswalk Recognition in BlindSpot.
Trains a lightweight PyTorch classifier (MobileNetV3-Small) on the Kaggle Crosswalk Dataset.
Also evaluates classical GLCM feature baseline for comparison.
Generates genuine evaluation metrics, confusion matrix, and training curves.
"""

import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = str(Path('.matplotlib').resolve())
Path('.matplotlib').mkdir(exist_ok=True)

import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

DATA_DIR = Path('dataset/crosswalk/Original')
OUTPUT_EVAL = Path('outputs/evaluation')
OUTPUT_MODELS = Path('models')
OUTPUT_EVAL.mkdir(parents=True, exist_ok=True)
OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    'c1': 0,  # Front-view crossing
    'c2': 1,  # Left-side perspective
    'c3': 2,  # Right-side perspective
    'c4': 3   # Non-crosswalk
}

CLASS_NAMES = ['Front-View Crossing', 'Left-Side Perspective', 'Right-Side Perspective', 'Non-Crosswalk']
CLASS_SHORT = ['Front', 'Left', 'Right', 'None']

# 1. Gather all image paths and labels
samples = []
for img_path in sorted(DATA_DIR.glob('*.jpg')):
    prefix = img_path.name.split('_')[0].lower()
    if prefix in CLASS_MAP:
        samples.append((str(img_path), CLASS_MAP[prefix], img_path.name))

print(f"Total valid samples loaded: {len(samples)}")

df_samples = pd.DataFrame(samples, columns=['path', 'label', 'filename'])

# 2. Stratified train/val/test split (70% train, 15% val, 15% test)
train_df, temp_df = train_test_split(df_samples, test_size=0.30, random_state=SEED, stratify=df_samples['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df['label'])

print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
train_df.to_csv(OUTPUT_EVAL / 'train_split.csv', index=False)
val_df.to_csv(OUTPUT_EVAL / 'val_split.csv', index=False)
test_df.to_csv(OUTPUT_EVAL / 'test_split.csv', index=False)

# 3. PyTorch Dataset Definition
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

# Data transforms with moderate augmentation on train
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(CrosswalkDataset(train_df, train_transform), batch_size=32, shuffle=True)
val_loader = DataLoader(CrosswalkDataset(val_df, val_test_transform), batch_size=32, shuffle=False)
test_loader = DataLoader(CrosswalkDataset(test_df, val_test_transform), batch_size=32, shuffle=False)

# 4. Model Definition: MobileNetV3-Small (lightweight, designed for edge/mobile CPU)
print("Initializing MobileNetV3-Small backbone...")
try:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
except Exception:
    model = models.mobilenet_v3_small(pretrained=True)

# Replace classifier head for 4 classes
num_features = model.classifier[3].in_features
model.classifier[3] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_features, 4)
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# 5. Training loop
EPOCHS = 15
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print(f"Starting training for {EPOCHS} epochs on {DEVICE}...")
start_time = time.time()
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels, _ in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    train_loss = running_loss / total
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss_total = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss_total += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_total / val_total
    val_acc = val_correct / val_total

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), OUTPUT_MODELS / 'crosswalk_classifier.pth')

    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds.")

# 6. Load best model and evaluate on Test Set
model.load_state_dict(torch.load(OUTPUT_MODELS / 'crosswalk_classifier.pth', map_location=DEVICE))
model.eval()

all_preds = []
all_targets = []
all_probs = []
inference_times = []

with torch.no_grad():
    for imgs, labels, _ in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        t0 = time.time()
        outputs = model(imgs)
        if DEVICE.type == 'cpu':
            t1 = time.time()
            inference_times.append((t1 - t0) / imgs.size(0))
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

# Compute metrics
acc = accuracy_score(all_targets, all_preds)
precision, recall, f1, support = precision_recall_fscore_support(all_targets, all_preds, average='macro')
precision_cls, recall_cls, f1_cls, _ = precision_recall_fscore_support(all_targets, all_preds, average=None)
cm = confusion_matrix(all_targets, all_preds)
mean_inf_time_ms = float(np.mean(inference_times) * 1000) if inference_times else 0.0

print("\n--- Genuine Test Set Evaluation Results ---")
print(f"Overall Accuracy: {acc * 100:.2f}%")
print(f"Macro Precision:  {precision * 100:.2f}%")
print(f"Macro Recall:     {recall * 100:.2f}%")
print(f"Macro F1 Score:   {f1 * 100:.2f}%")
print(f"Mean CPU Inference Latency per image: {mean_inf_time_ms:.2f} ms")
print("\nClassification Report:\n", classification_report(all_targets, all_preds, target_names=CLASS_NAMES))

# 7. Compare with Classical Feature Baseline (GLCM Features from CSV)
glcm_csv_path = Path('dataset/crosswalk/10_FEATURES_M17_CM6b_TH199.csv')
glcm_eval_results = {}
if glcm_csv_path.exists():
    df_glcm = pd.read_csv(glcm_csv_path)
    X = df_glcm.drop(columns=['Class']).values
    y = df_glcm['Class'].values - 1  # 1-indexed to 0-indexed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
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
    print(f"GLCM Baseline Accuracy: {rf_acc*100:.2f}%, F1: {rf_f1*100:.2f}%")

# 8. Save Metrics JSON
eval_metrics = {
    'model_name': 'MobileNetV3-Small (Fine-tuned for 4-class Crosswalk Recognition)',
    'architecture': 'MobileNetV3-Small with custom Linear classifier head and Dropout(0.3)',
    'total_dataset_size': len(df_samples),
    'splits': {'train': len(train_df), 'val': len(val_df), 'test': len(test_df)},
    'training_epochs': EPOCHS,
    'training_time_seconds': round(training_time, 2),
    'mean_cpu_latency_ms': round(mean_inf_time_ms, 2),
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
                'support': int(support[i])
            } for i in range(4)
        }
    },
    'confusion_matrix': cm.tolist(),
    'classical_glcm_baseline': glcm_eval_results
}

with open(OUTPUT_EVAL / 'crosswalk_metrics.json', 'w') as f:
    json.dump(eval_metrics, f, indent=2)

# 9. Plot and save Confusion Matrix
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

# 10. Plot Training Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(range(1, EPOCHS + 1), history['train_loss'], label='Train Loss', color='#1f77b4', marker='o')
ax1.plot(range(1, EPOCHS + 1), history['val_loss'], label='Val Loss', color='#ff7f0e', marker='s')
ax1.set_title('Cross-Entropy Loss vs Epoch', fontsize=11, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

ax2.plot(range(1, EPOCHS + 1), [x*100 for x in history['train_acc']], label='Train Acc', color='#1f77b4', marker='o')
ax2.plot(range(1, EPOCHS + 1), [x*100 for x in history['val_acc']], label='Val Acc', color='#ff7f0e', marker='s')
ax2.set_title('Classification Accuracy (%) vs Epoch', fontsize=11, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()
plt.tight_layout()
plt.savefig(OUTPUT_EVAL / 'training_curves.png', dpi=200)
plt.close()

print("Crosswalk model training and genuine evaluation successfully completed!")
