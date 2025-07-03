# Full training & evaluation pipeline for ArcFace, AdaFace, MagFace, GhostFaceNet on YLFW
# Paper-based setup: ResNet-50 backbone + margin-based heads + SGD + cosine verification


print("Kod SLURM altında başladı", flush=True)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics import confusion_matrix
from torchvision.models import ResNet50_Weights
from backbones.ghostnetv2 import ghostnetv2
from torch.utils.data import random_split
from collections import defaultdict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------- ArcMarginProduct -------------------------
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - cosine ** 2)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)
        output = self.s * (one_hot * phi + (1 - one_hot) * cosine)
        return output

# ------------------------- AdaFace Margin Head -------------------------
class AdaFaceHead(nn.Module):
    def __init__(self, embedding_size, num_classes, m=0.4, h=0.333, s=64.0):
        super().__init__()
        self.s = s
        self.m = m
        self.h = h
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    #25.05.2025, 00.25

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-5, 1.0 - 1e-5))

    # BOYUT UYUMLU MASKE: [batch_size] -> [batch_size, 1] -> broadcast to theta shape
        labels_mask = (labels >= 0).unsqueeze(1).expand_as(theta)
        B = torch.where(labels_mask, torch.pow(torch.abs(theta - self.h), 2), torch.zeros_like(theta))

        margin_theta = theta + self.m + B
        logits = torch.cos(margin_theta)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = self.s * (one_hot * logits + (1.0 - one_hot) * cosine)
        return output


# ------------------------- MagFace Margin Head -------------------------
class MagFaceHead(nn.Module):
    def __init__(self, embedding_size, num_classes, s=64.0, m=0.5, a=10.0, b=110.0):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.a = a
        self.b = b

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        norms = torch.norm(embeddings, dim=1, keepdim=True).clamp(min=0.001, max=100)
        margin = self.m * (norms - self.a) / (self.b - self.a)
        margin = margin.clamp(min=0.0, max=self.m)
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-5, 1.0 - 1e-5))
        logits = torch.cos(theta + margin)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = self.s * (one_hot * logits + (1.0 - one_hot) * cosine)
        return output

# ------------------------- GhostFaceNet Margin Head -------------------------
class GhostFaceNetHead(nn.Module):
    def __init__(self, embedding_size, num_classes, s=64.0):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.s = s

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        logits = cosine
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = self.s * logits
        return output

# ------------------------- CosFace Margin Head -------------------------
class CosFaceHead(nn.Module):
    def __init__(self, embedding_size, num_classes, s=64.0, m=0.35):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = self.s * (one_hot * phi + (1.0 - one_hot) * cosine)
        return output

# ------------------------- SphereFace Margin Head -------------------------
class SphereFaceHead(nn.Module):
    def __init__(self, embedding_size, num_classes, s=64.0, m=1.35):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        cosine = torch.clamp(cosine, -1.0 + 1e-5, 1.0 - 1e-5)
        theta = torch.acos(cosine)
        phi = torch.cos(self.m * theta)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        output = self.s * (one_hot * phi + (1.0 - one_hot) * cosine)
        return output

# ------------------------- Backbone Wrapper -------------------------
def build_model(model_type, num_classes):
    embedding_size = 512

    if model_type == 'GhostFaceNetV2':
        backbone = ghostnetv2(
            pretrained=True,
            num_classes=embedding_size,
            width=1.0,
            dropout=0.2,
            args=None
        )
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Linear(in_features, embedding_size)
        head = GhostFaceNetHead(embedding_size, num_classes)
    else:
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, embedding_size)

        if model_type == 'ArcFace':
            head = ArcMarginProduct(embedding_size, num_classes)
        elif model_type == 'AdaFace':
            head = AdaFaceHead(embedding_size, num_classes)
        elif model_type == 'MagFace':
            head = MagFaceHead(embedding_size, num_classes)
        elif model_type == 'GhostFaceNet':
            head = GhostFaceNetHead(embedding_size, num_classes)
        elif model_type == 'CosFace':
            head = CosFaceHead(embedding_size, num_classes)
        elif model_type == 'SphereFace':
            head = SphereFaceHead(embedding_size, num_classes)
        else:
            raise ValueError("Unknown model type")

    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x, label):
            feats = self.backbone(x)
            feats = F.normalize(feats, p=2, dim=1)
            return self.head(feats, label)

    return FullModel()


# ------------------------- YLFW Dataset -------------------------
class YLFWTrainDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.class_to_idx = {}
        self.transform = transform
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            self.class_to_idx[class_name] = len(self.class_to_idx)
            for fname in os.listdir(class_path):
                if fname.endswith('.png'):
                    self.samples.append((os.path.join(class_path, fname), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# ------------------------- Pair Evaluation -------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def evaluate_model(model, device, transform, pair_file, image_root):
    pairs = []
    with open(pair_file, 'r') as f:
        for line in f:
            p1, p2, label = line.strip().split()
            race = p1.split('_')[0]  
            pairs.append((os.path.join(image_root, p1.strip('/')), 
                          os.path.join(image_root, p2.strip('/')), 
                          int(label), race))

    scores_by_race = defaultdict(list)
    labels_by_race = defaultdict(list)
    all_scores, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for p1, p2, label,race in tqdm(pairs):
            try:
                img1 = transform(Image.open(p1).convert("RGB")).unsqueeze(0).to(device)
                img2 = transform(Image.open(p2).convert("RGB")).unsqueeze(0).to(device)
                f1 = model.backbone(img1)
                f2 = model.backbone(img2)
                f1 = F.normalize(f1, p=2, dim=1).cpu().numpy().flatten()
                f2 = F.normalize(f2, p=2, dim=1).cpu().numpy().flatten()
                all_scores.append(cosine_similarity(f1, f2))
                all_labels.append(int(label))
                scores_by_race[race].append(cosine_similarity(f1, f2))
                labels_by_race[race].append(int(label))

            except:
                continue
    def compute_metrics(scores, labels):
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        fnmr1 = 1 - tpr[np.where(fpr <= 1e-1)[0][-1]] if np.any(fpr <= 1e-1) else 1.0
        fnmr2 = 1 - tpr[np.where(fpr <= 1e-2)[0][-1]] if np.any(fpr <= 1e-2) else 1.0
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.abs(fpr - fnr))]
        acc = accuracy_score(labels, np.array(scores) > 0.5)
        return {'FNMR@1e-1': fnmr1, 'FNMR@1e-2': fnmr2, 'EER': eer, 'AUC': roc_auc, 'Accuracy': acc}

    race_metrics = {
        race: compute_metrics(scores_by_race[race], labels_by_race[race])
        for race in scores_by_race
    }

    general_metrics = compute_metrics(all_scores, all_labels)
    general_metrics.update({
        'scores': all_scores,
        'labels': all_labels,
        'race_metrics': race_metrics
    })
    return general_metrics

# ------------------------- Utility: Save All Results -------------------------
def save_results_and_plots(model_name, results, train_losses, val_losses, model, train_accuracies, val_accuracies):

    global all_model_rocs
    # Create output directory
    os.makedirs("outputs_n7", exist_ok=True)

    scores = results['scores']
    labels = np.array(results['labels']).astype(int)


    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(f"outputs_n7/{model_name}_roc_curve.png")
    plt.close()


    # ROC bilgilerinin global listeye eklenmesi
    all_model_rocs.append({
        'model_name': model_name,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc
    })

     # Loss Plot
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {model_name}')
    plt.legend()
    plt.savefig(f"outputs_n7/{model_name}_loss_curve.png")
    plt.close()

    # Save confusion matrix (if binary classification like pair matching)
    pred_labels = np.array(scores) > 0.5
    cm = confusion_matrix(labels, pred_labels)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
     # Hücre içlerine sayıları yaz
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(f"outputs_n7/{model_name}_confusion_matrix.png")
    plt.close()

    # Save metrics to CSV
    fnmr1 = 1 - tpr[np.where(fpr <= 1e-1)[0][-1]] if np.any(fpr <= 1e-1) else 1.0
    fnmr2 = 1 - tpr[np.where(fpr <= 1e-2)[0][-1]] if np.any(fpr <= 1e-2) else 1.0
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]
    acc = accuracy_score(labels, pred_labels)

    results_path = "outputs_n7/results.csv"
    df = pd.DataFrame([[model_name, fnmr1, fnmr2, eer, roc_auc, acc]],
                      columns=["Model", "FNMR@1e-1", "FNMR@1e-2", "EER", "AUC", "Accuracy"])
    if os.path.exists(results_path):
        df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        df.to_csv(results_path, index=False)

        # Save race-based metrics if available
    if 'race_metrics' in results:
        race_df = pd.DataFrame([
            [model_name, race, m['FNMR@1e-1'], m['FNMR@1e-2'], m['EER'], m['AUC'], m['Accuracy']]
            for race, m in results['race_metrics'].items()
        ], columns=["Model", "Race", "FNMR@1e-1", "FNMR@1e-2", "EER", "AUC", "Accuracy"])

        race_results_path = "outputs_n7/results_by_race.csv"
        if os.path.exists(race_results_path):
            race_df.to_csv(race_results_path, mode='a', header=False, index=False)
        else:
            race_df.to_csv(race_results_path, index=False)
    
    # Save model
    torch.save(model.state_dict(), f"outputs_n7/{model_name}.pth")

    return {
    'FNMR@1e-1': fnmr1,
    'FNMR@1e-2': fnmr2,
    'EER': eer,
    'AUC': roc_auc,
    'Accuracy': acc,
    'scores': scores,
    'labels': labels }

all_model_rocs = []
# ------------------------- Main Function -------------------------
def train_and_evaluate_all():
    train_root1 = r"/arf/scratch/csozeri/YLFW_Dev/Train/data_aligned_112"
    train_root2 = r"/arf/scratch/csozeri/YLFW_Dev/Train/data_aligned_112_b_part"
    test_root = r"/arf/scratch/csozeri/YLFW_Dev/Test/data_aligned"
    pair_file = r"/arf/scratch/csozeri/YLFW_Dev/Test/metadata/YLFW_dev_test_pairs.txt"

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

     # Birleştirilen eğitim verisi
    dataset_original = YLFWTrainDataset(train_root1, transform)
    dataset_augmented = YLFWTrainDataset(train_root2, transform)

    # Sadece orijinal veriyi train/val olarak böl
    total_len = len(dataset_original)
    val_len = int(0.2 * total_len)
    train_len = total_len - val_len
    train_dataset, val_dataset = random_split(dataset_original, [train_len, val_len])

    # Augmente veriyi sadece train'e ekle
    combined_train_dataset = torch.utils.data.ConcatDataset([train_dataset, dataset_augmented])
    train_loader = DataLoader(combined_train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    print(f"Train: {len(train_dataset)}  |  Validation: {len(val_dataset)}\n")


    for model_name in ['GhostFaceNetV2','CosFace', 'SphereFace', 'ArcFace', 'AdaFace', 'MagFace', 'GhostFaceNet']:
        print(f"\nTraining {model_name}...")
        model = build_model(model_name, num_classes=len(dataset_original.class_to_idx)).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []
        num_epochs = 15
        for epoch in range(num_epochs):
            print(f"[{model_name}] Epoch {epoch+1}/{num_epochs}")
            model.train()
            lr = 0.01 - (epoch / num_epochs) * (0.01 - 0.00001)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            total_loss = 0
            correct = 0
            total = 0
            print(f"Model: {model_name}, Epoch {epoch+1}")
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images, labels)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            avg_train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_acc)

            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images, labels)
                    loss = F.cross_entropy(logits, labels)
                    val_loss += loss.item()
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)

            print(f"Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        print(f"\n Evaluating {model_name} on YLFW-Dev-Test...")
        results = evaluate_model(model, device, transform, pair_file, test_root)
        for k, v in results.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
        save_results_and_plots(model_name, results, train_losses, val_losses, model, train_accuracies, val_accuracies)

        # Extra: Plot accuracy curves
        plt.figure()
        plt.plot(train_accuracies, label="Train Accuracy")
        plt.plot(val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy - {model_name}")
        plt.legend()
        plt.savefig(f"outputs_n7/{model_name}_accuracy_curve.png")
        plt.close()

        # Extra: Plot val loss curve
        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss - {model_name}")
        plt.legend()
        plt.savefig(f"outputs_n7/{model_name}_train_val_loss_curve.png")
        plt.close()

    # Tüm modeller için tek ROC grafiği çiz
        
    plt.figure()
    for roc in all_model_rocs:
        plt.plot(roc['fpr'], roc['tpr'], label=f"{roc['model_name']} (AUC = {roc['auc']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc='lower right')
    plt.savefig("outputs_n7/all_models_roc_curve.png")
    plt.close()


if __name__ == "__main__":
    train_and_evaluate_all()
