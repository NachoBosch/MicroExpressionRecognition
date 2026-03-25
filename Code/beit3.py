import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import BeitForImageClassification, BeitImageProcessor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# CONFIG
MODEL_NAME = "microsoft/beit-base-patch16-224"
BATCH_SIZE = 16
num_epochs = 5
LEARNING_RATE = 2e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = "../CASME3/dataset-splitted"

# Procesador (similar a tokenizer)
processor = BeitImageProcessor.from_pretrained(MODEL_NAME)

# Transformaciones de imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)
test_dataset  = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader, test_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE),DataLoader(test_dataset, batch_size=BATCH_SIZE)

num_classes = len(train_dataset.classes)
print(num_classes)

# MODEL
model = BeitForImageClassification.from_pretrained(MODEL_NAME,
    num_labels=num_classes,
    ignore_mismatched_sizes=True)
model.to(device)

# OPTIMIZER
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

save_dir = f"CASME3-beit-microexp-{num_epochs}-v2"
log_dir = f"runs/{save_dir}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    total_loss, correct_train, total_train = 0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(pixel_values=images, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(-1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    avg_train_loss = total_loss / len(train_loader)
    train_acc = correct_train / total_train

    writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
    writer.add_scalar("Accuracy/Train", train_acc, epoch + 1)

    model.eval()
    val_loss, correct_val, total_val = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images, labels=labels)
            val_loss += outputs.loss.item()
            preds = outputs.logits.argmax(-1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct_val / total_val

    writer.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)
    writer.add_scalar("Accuracy/Validation", val_acc, epoch + 1)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint_path = os.path.join(log_dir, "best_model.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Modelo mejorado guardado en {checkpoint_path}")

model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluando en Test"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(pixel_values=images)
        preds = outputs.logits.argmax(-1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

test_acc = test_correct / test_total
writer.add_scalar("Accuracy/Test", test_acc, num_epochs)
print(f"🎯 Precisión en Test: {test_acc:.4f}")

# ====================
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
processor.save_pretrained(save_dir)
writer.close()
print(f"Modelo y procesador guardados en: {save_dir}")