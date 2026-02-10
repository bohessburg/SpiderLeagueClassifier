# To download the dataset, install the kaggle CLI (`pip install kaggle`),
# configure your API credentials (~/.kaggle/kaggle.json), then run:
# kaggle datasets download -d gpiosenka/yikes-spiders-15-species -p data --unzip

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from huggingface_hub import hf_hub_download, HfApi

# ─── Constants ───────────────────────────────────────────────────────────────

DATA_DIR = 'data'
MODEL_PATH = 'best_spider_model.pth'
IMAGE_SIZE = 224
NUM_CLASSES = 15

# Replace with your Hugging Face username
HF_REPO_ID = "YOUR_HF_USERNAME/SpiderLeague"
HF_MODEL_FILENAME = "best_spider_model.pth"

# ImageNet normalization — required when using ImageNet-pretrained weights.
# The pretrained backbone learned features assuming inputs are normalized this way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# The 15 spider species in the dataset (alphabetical, matching ImageFolder order)
CLASS_NAMES = [
    'Black Widow', 'Blue Tarantula', 'Bold Jumper', 'Brown Grass Spider',
    'Brown Recluse Spider', 'Deinopis Spider', 'Golden Orb Weaver',
    'Hobo Spider', 'Huntsman Spider', 'Ladybird Mimic Spider',
    'Peacock Spider', 'Red Knee Tarantula', 'Spiny Backed Orb Weaver',
    'White Knee Spider', 'Yellow Garden Spider',
]

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Transforms ──────────────────────────────────────────────────────────────

# Training: augmentation forces the model to learn shape/texture, not memorize poses
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Validation/test/inference: deterministic preprocessing only, no augmentation
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ─── Model ───────────────────────────────────────────────────────────────────

class SpiderClassificationHead(nn.Module):
    """
    Multi-layer classification head for fine-grained spider species classification.

    Architecture:
      Linear(768 -> 512) -> BatchNorm -> ReLU -> Dropout(0.4)
      Linear(512 -> 256) -> BatchNorm -> ReLU -> Dropout(0.3)
      Linear(256 -> num_classes)

    Why this design:
    - Two hidden layers let the network learn non-linear feature combinations
      needed to distinguish similar species (e.g. Brown Recluse vs Hobo Spider).
    - Decreasing width (768 -> 512 -> 256) creates an information bottleneck that
      forces the network to distill the most discriminative features.
    - Dropout is aggressive (0.4) because the dataset is small (~2K training images).
    - BatchNorm stabilizes training, especially when the backbone is frozen.
    """
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class SpiderClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # ConvNeXt-Tiny: a modernized ResNet that uses ideas from Vision Transformers
        # (7x7 depthwise convs, LayerNorm, GELU, inverted bottleneck).
        # num_classes=0 removes the original head and returns pooled 768-dim features.
        self.backbone = timm.create_model('convnext_tiny', pretrained=True, num_classes=0)
        self.classifier = SpiderClassificationHead(
            in_features=self.backbone.num_features,  # 768 for convnext_tiny
            num_classes=num_classes,
        )

    def forward(self, x):
        features = self.backbone(x)       # (batch, 768)
        return self.classifier(features)


# ─── Helper functions ────────────────────────────────────────────────────────

def evaluate_model(model, dataloader, class_names, device):
    """Compute overall accuracy, per-class accuracy, and confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    overall_acc = (all_preds == all_labels).mean()
    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)

    return overall_acc, report, cm


def plot_confusion_matrix(cm, class_names):
    """Visualize confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Spider Species Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def preprocess_image(image_path, transform):
    """Load a single image for inference. Returns the original PIL image and the tensor."""
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)


def predict(model, image_tensor, device):
    """Run inference on a single image tensor. Returns class probabilities."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


def visualize_predictions(original_image, probabilities, class_names):
    """Show the image alongside a horizontal bar chart of class probabilities."""
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))
    axarr[0].imshow(original_image)
    axarr[0].axis("off")
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0, 1)
    plt.tight_layout()
    plt.show()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, dataloader, criterion, device):
    """Validate for one epoch. Returns average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def download_model():
    """Download the pretrained model from Hugging Face Hub."""
    print(f"Downloading pretrained model from {HF_REPO_ID}...")
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
    return path


def upload_model():
    """Upload the trained model to Hugging Face Hub."""
    if not os.path.exists(MODEL_PATH):
        print("No model file found. Train the model first.")
        return
    print(f"Uploading model to {HF_REPO_ID}...")
    api = HfApi()
    api.create_repo(repo_id=HF_REPO_ID, exist_ok=True)
    api.upload_file(
        path_or_fileobj=MODEL_PATH,
        path_in_repo=HF_MODEL_FILENAME,
        repo_id=HF_REPO_ID,
    )
    print(f"Model uploaded to https://huggingface.co/{HF_REPO_ID}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpiderLeague: Spider Species Classifier")
    parser.add_argument("--infer", type=str, metavar="IMAGE_PATH",
                        help="Skip training. Download pretrained model and classify the given image.")
    parser.add_argument("--upload", action="store_true",
                        help="Upload best_spider_model.pth to Hugging Face Hub after training.")
    args = parser.parse_args()

    # ── Inference-only mode ──────────────────────────────────────────────
    if args.infer:
        model = SpiderClassifier(num_classes=NUM_CLASSES).to(device)

        # Use local model if available, otherwise download from HF Hub
        if os.path.exists(MODEL_PATH):
            print(f"Loading local model from {MODEL_PATH}")
            weights_path = MODEL_PATH
        else:
            weights_path = download_model()

        model.load_state_dict(torch.load(weights_path, map_location=device))

        original_image, image_tensor = preprocess_image(args.infer, eval_transform)
        probabilities = predict(model, image_tensor, device)
        visualize_predictions(original_image, probabilities, CLASS_NAMES)
        exit()

    # ── Training mode ────────────────────────────────────────────────────

    # Load datasets (requires the data/ folder)
    train_ds = ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    valid_ds = ImageFolder(os.path.join(DATA_DIR, 'valid'), transform=eval_transform)
    test_ds = ImageFolder(os.path.join(DATA_DIR, 'test'), transform=eval_transform)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=32, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

    print(f"Classes ({len(train_ds.classes)}): {train_ds.classes}")
    print(f"Training samples: {len(train_ds)}")

    model = SpiderClassifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    # ── Phase 1: Train head only (backbone frozen) ───────────────────
    # The backbone has good ImageNet features. The head is randomly initialized.
    # If we train everything together, the random gradients from the head would
    # destroy the pretrained features. So we freeze the backbone first.

    print("\n=== Phase 1: Training classification head (backbone frozen) ===")

    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer_p1 = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    num_epochs_p1 = 10

    for epoch in range(num_epochs_p1):
        if device.type == "cuda":
            torch.cuda.empty_cache()

        train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer_p1, device)
        val_loss, val_acc = validate(model, valid_dl, criterion, device)

        print(f"  Epoch {epoch+1}/{num_epochs_p1} — "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"    -> Saved best model (val_acc={val_acc:.4f})")

    # ── Phase 2: Fine-tune everything with discriminative learning rates ──
    # Now the head is trained, so gradients flowing into the backbone will be meaningful.
    # We use a much smaller LR for the backbone (pretrained features need gentle updates)
    # and a moderately smaller LR for the head (already partially converged).

    print("\n=== Phase 2: Fine-tuning full model (backbone unfrozen) ===")

    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer_p2 = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-5},    # backbone: very small LR
        {'params': model.classifier.parameters(), 'lr': 1e-4},  # head: 10x backbone
    ], weight_decay=1e-4)

    num_epochs_p2 = 20
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=num_epochs_p2, eta_min=1e-7)

    for epoch in range(num_epochs_p2):
        if device.type == "cuda":
            torch.cuda.empty_cache()

        train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer_p2, device)
        val_loss, val_acc = validate(model, valid_dl, criterion, device)
        scheduler.step()

        print(f"  Epoch {epoch+1}/{num_epochs_p2} — "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"    -> Saved best model (val_acc={val_acc:.4f})")

    # ── Final evaluation on test set ─────────────────────────────────
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    print("\n=== Test Set Evaluation ===")
    class_names = train_ds.classes
    overall_acc, report, cm = evaluate_model(model, test_dl, class_names, device)
    print(f"Overall test accuracy: {overall_acc:.4f}")
    print(report)
    plot_confusion_matrix(cm, class_names)

    # ── Upload to Hugging Face Hub if requested ──────────────────────
    if args.upload:
        upload_model()

    # ── Single image inference example ───────────────────────────────
    test_image_path = os.path.join(DATA_DIR, "test", "Huntsman Spider", "3.jpg")
    original_image, image_tensor = preprocess_image(test_image_path, eval_transform)
    probabilities = predict(model, image_tensor, device)
    visualize_predictions(original_image, probabilities, class_names)
