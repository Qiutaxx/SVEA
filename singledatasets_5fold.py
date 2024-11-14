import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from torchvision import transforms
from PIL import Image

batch_size = 128
learning_rate = 0.001
num_epochs = 100
k_folds = 5
early_stopping_patience = 10

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CustomDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        transform = transforms.ToTensor()
        image = transform(image)
        return image, label

def load_datasets(dataset_path):
    image_paths = []
    labels = []
    class_to_idx = {name: i for i, name in enumerate(sorted(os.listdir(dataset_path)))}
    for class_name in class_to_idx.keys():
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image_paths.append(img_path)
            labels.append(class_to_idx[class_name])
    return CustomDataset(image_paths, labels)

# MHSA
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query_conv(x).view(batch_size, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2)
        key = self.key_conv(x).view(batch_size, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2)
        value = self.value_conv(x).view(batch_size, self.num_heads, C // self.num_heads, -1).permute(0, 1, 3, 2)

        attention_scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / (C // self.num_heads) ** 0.5
        attention_probs = self.softmax(attention_scores)
        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 1, 3, 2).contiguous().view(batch_size, C, H, W)

        out = self.out_conv(context_layer)
        return out + x

# Enhanced AlexNet with MultiHeadSelfAttention
class EnhancedAlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(EnhancedAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            MultiHeadSelfAttention(192, num_heads=8)
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3_4 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            MultiHeadSelfAttention(512, num_heads=8)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.LocalResponseNorm(5),
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_classes, num_classes)

    def forward(self, x):
        x = self.features(x)

        x1 = self.conv3_1(x)
        x2 = self.conv3_2(x)
        x3 = self.conv3_3(x)
        x4 = self.conv3_4(x)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.conv4(x)
        x = self.conv5(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 加载数据集
dataset_path = ''
dataset = load_datasets(dataset_path)

num_classes = len(set(dataset.labels))

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

fold_results = []

for fold_idx, (train_indices, val_indices) in enumerate(kf.split(range(len(dataset.image_paths)))):
    print(f"Fold {fold_idx + 1}")

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = EnhancedAlexNet(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_val_accuracy = 0.0
    best_val_f1 = 0.0
    best_val_recall = 0.0
    no_improvement_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct_train, total_train = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        train_loss = running_loss / len(train_loader)

        model.eval()
        correct_val, total_val = 0, 0
        val_labels_all, val_preds_all = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                val_labels_all.extend(labels.cpu().numpy())
                val_preds_all.extend(predicted.cpu().numpy())

        val_accuracy = 100 * correct_val / total_val
        val_f1 = f1_score(val_labels_all, val_preds_all, average='weighted')
        val_recall = recall_score(val_labels_all, val_preds_all, average='weighted')

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Accuracy: {val_accuracy:.2f}%, Val F1 Score: {val_f1:.2f}, Val Recall: {val_recall:.2f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_f1 = val_f1
            best_val_recall = val_recall
            no_improvement_counter = 0
            torch.save(model.state_dict(), f'HG00514_model_fold_{fold_idx + 1}.pth')
        else:
            no_improvement_counter += 1

        if no_improvement_counter >= early_stopping_patience:
            print("Early stopping due to no improvement.")
            break

    fold_results.append((best_val_accuracy, best_val_f1, best_val_recall, val_labels_all, val_preds_all))  # 保存Recall
    print(f"Best Validation Accuracy for fold {fold_idx + 1}: {best_val_accuracy:.2f}%")
    print(f"Best Validation F1 Score for fold {fold_idx + 1}: {best_val_f1:.2f}")
    print(f"Best Validation Recall for fold {fold_idx + 1}: {best_val_recall:.2f}")
    print("-" * 50)


for i, (val_accuracy, val_f1, val_recall, val_labels_all, val_preds_all) in enumerate(fold_results):
    print(f"Fold {i + 1} - Accuracy: {val_accuracy:.2f}%, F1 Score: {val_f1:.2f}, Recall: {val_recall:.2f}")
    for label in set(val_labels_all):
        precision = precision_score(val_labels_all, val_preds_all, labels=[label], average='weighted')
        recall = recall_score(val_labels_all, val_preds_all, labels=[label], average='weighted')
        f1 = f1_score(val_labels_all, val_preds_all, labels=[label], average='weighted')
        print(f"  Class {label} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")


all_labels = np.concatenate([result[3] for result in fold_results])
all_preds = np.concatenate([result[4] for result in fold_results])

total_accuracy = 100 * np.mean(all_labels == all_preds)
total_f1 = f1_score(all_labels, all_preds, average='weighted')
total_recall = recall_score(all_labels, all_preds, average='weighted')

print(f"Total Accuracy: {total_accuracy:.2f}%, Total F1 Score: {total_f1:.2f}, Total Recall: {total_recall:.2f}")


for label in set(all_labels):
    precision = precision_score(all_labels, all_preds, labels=[label], average='weighted')
    recall = recall_score(all_labels, all_preds, labels=[label], average='weighted')
    f1 = f1_score(all_labels, all_preds, labels=[label], average='weighted')
    print(f"  Class {label} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
