import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from sklearn.metrics import f1_score
from PIL import Image
import concurrent.futures
import multiprocessing


batch_size = 128
learning_rate = 0.001
num_epochs = 100
early_stopping_patience = 10
split_ratio = 0.9


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_datasets(dataset_paths, transform):
    image_paths = []
    labels = []
    for dataset_path in dataset_paths:
        class_names = sorted(os.listdir(dataset_path))
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        for class_name in class_to_idx.keys():
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if not img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    continue
                image_paths.append(img_path)
                labels.append(class_to_idx[class_name])
    return CustomDataset(image_paths, labels, transform=transform)


class EnhancedAlexNet(nn.Module):
    def __init__(self, num_classes=4):
        super(EnhancedAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

class LeNet(nn.Module):
    def __init__(self, num_classes=4):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(3, 6, kernel_size=5)
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.f6 = nn.Linear(120 * 49 * 49, 84)
        self.f7 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = self.s2(x)
        x = torch.relu(self.c3(x))
        x = self.s4(x)
        x = torch.relu(self.c5(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.f6(x))
        x = self.f7(x)
        return x

class CustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes=4):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.linear(out)
            return out

        def ResNet18(num_classes=4):
            return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

        def train_and_evaluate(model, train_loader, val_loader, device, num_epochs=100):
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            best_val_accuracy = 0.0
            best_val_f1 = 0.0
            no_improvement_counter = 0

            for epoch in range(num_epochs):
                model.train()
                running_loss, correct_train, total_train = 0.0, 0, 0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
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
                        _, predicted = torch.max(outputs, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).sum().item()
                        val_labels_all.extend(labels.cpu().numpy())
                        val_preds_all.extend(predicted.cpu().numpy())

                val_accuracy = 100 * correct_val / total_val
                val_f1 = f1_score(val_labels_all, val_preds_all, average='weighted')

                print(f"Epoch [{epoch + 1}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%, Val F1: {val_f1:.2f}")

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_val_f1 = val_f1
                    no_improvement_counter = 0
                    torch.save(model.state_dict(), f'{model.__class__.__name__}_best.pth')
                else:
                    no_improvement_counter += 1

                if no_improvement_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            return best_val_accuracy, best_val_f1

        def train_model(model_name, model, train_loader, val_loader, device, num_epochs):
            print(f"\nTraining {model_name}...")
            model = model.to(device)
            best_val_accuracy, best_val_f1 = train_and_evaluate(model, train_loader, val_loader, device, num_epochs)
            print(
                f"{model_name} Training Completed! Best Val Accuracy: {best_val_accuracy:.2f}%, Best Val F1: {best_val_f1:.2f}")
            return best_val_accuracy

        def parallel_training(models_dict, train_loader, val_loader, device, num_epochs=100):
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                for model_name, model in models_dict.items():
                    futures.append(
                        executor.submit(train_model, model_name, model, train_loader, val_loader, device, num_epochs))

                for future in concurrent.futures.as_completed(futures):
                    future.result()

        if __name__ == '__main__':
            train_dataset_paths = [
                '',
                '',
                ''
            ]
            dataset = load_datasets(train_dataset_paths, transform)
            train_size = int(split_ratio * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            models_dict = {
                "EnhancedAlexNet": EnhancedAlexNet(num_classes=4),
                "LeNet": LeNet(num_classes=4),
                "CustomCNN": CustomCNN(num_classes=4),
                "AlexNet": AlexNet(num_classes=4),
            }

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            parallel_training(models_dict, train_loader, val_loader, device, num_epochs)

