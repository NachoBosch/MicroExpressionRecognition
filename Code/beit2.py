from datasets import load_dataset
from transformers import BeitImageProcessor, BeitForImageClassification
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

batch_size = 16
num_epochs = 50

dataset = load_dataset(
    "imagefolder",
    data_dir="../CASMEII/CASME-II-Binary-splitted",
    split=None
)

train_ds = dataset["train"]
val_ds = dataset["validation"]
test_ds = dataset["test"]

model_name = "microsoft/beit-base-patch16-224"
image_processor = BeitImageProcessor.from_pretrained(model_name)

def transform(example):
    image = example["image"]
    inputs = image_processor(images=image, return_tensors="pt")
    inputs["labels"] = example["label"]
    return inputs

train_ds.set_transform(transform)
val_ds.set_transform(transform)
test_ds.set_transform(transform)

def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    return {'pixel_values': pixel_values, 'labels': labels}

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

model = BeitForImageClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "ME", 1: "NE"},
    label2id={"ME": 0, "NE": 1},
    ignore_mismatched_sizes=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)

log_dir = f"runs/CASMEII-beit-microexp-{num_epochs}"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/Train"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        predictions = outputs.logits.argmax(-1)
        correct_train += (predictions == labels).sum().item()
        total_train += labels.size(0)

    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = correct_train / total_train

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
    writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
    writer.add_scalar("Accuracy/Train", train_accuracy, epoch + 1)

    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            val_loss += outputs.loss.item()

            predictions = outputs.logits.argmax(-1)
            correct_val += (predictions == labels).sum().item()
            total_val += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct_val / total_val

    print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    writer.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)
    writer.add_scalar("Accuracy/Validation", val_accuracy, epoch + 1)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint_path = os.path.join(log_dir, "best_model.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")


model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for batch in test_loader:
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(pixel_values=pixel_values)
        predictions = outputs.logits.argmax(-1)
        test_correct += (predictions == labels).sum().item()
        test_total += labels.size(0)

test_accuracy = test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.4f}")
writer.add_scalar("Accuracy/Test", test_accuracy, num_epochs)

writer.close()
model.save_pretrained(f"CASMEII-beit-microexp-{num_epochs}")
image_processor.save_pretrained(f"CASMEII-beit-microexp-{num_epochs}")