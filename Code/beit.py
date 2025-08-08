from datasets import load_dataset
from transformers import BeitImageProcessor
from torch.utils.data import DataLoader
import torch
from transformers import BeitForImageClassification
import torch
from torch.optim import AdamW
from tqdm import tqdm

# Cargar cada split desde sus carpetas
dataset = load_dataset(
    "imagefolder",
    data_dir="../CASMEII/CASME-II-Binary-splitted",
    split=None
)

# Accedemos a cada uno
train_ds = dataset["train"]
val_ds = dataset["validation"]
test_ds = dataset["test"]

model_name = "microsoft/beit-base-patch16-224"
image_processor = BeitImageProcessor.from_pretrained(model_name)

# Transformación con procesador
def transform(example):
    image = example["image"]
    inputs = image_processor(images=image, return_tensors="pt")
    inputs["labels"] = example["label"]
    return inputs

# Aplicar transformación a cada conjunto
train_ds.set_transform(transform)
val_ds.set_transform(transform)
test_ds.set_transform(transform)

def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    return {'pixel_values': pixel_values, 'labels': labels}

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False, collate_fn=collate_fn)


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

# Entrenamiento
model.train()
for epoch in range(5):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")

    # Validación
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            val_loss += outputs.loss.item()
            predictions = outputs.logits.argmax(-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(f"Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {correct/total:.4f}")

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

print(f"Test Accuracy: {test_correct / test_total:.4f}")

model.save_pretrained("CASMEII-beit-microexp")
image_processor.save_pretrained("CASMEII-beit-microexp")