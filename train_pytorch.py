import os
import shutil
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 0️⃣ Hilfsfunktion: Organisiere Bilder in Unterordner
def organize_dataset(base_path):
    print("Step 0️⃣: Organizing dataset...")
    for split in ['train', 'val']:
        split_path = os.path.join(base_path, split)
        if not os.path.exists(split_path):
            print(f"  {split_path} does not exist, skipping.")
            continue
        cats_path = os.path.join(split_path, 'cats')
        dogs_path = os.path.join(split_path, 'dogs')
        os.makedirs(cats_path, exist_ok=True)
        os.makedirs(dogs_path, exist_ok=True)

        for filename in os.listdir(split_path):
            filepath = os.path.join(split_path, filename)
            if os.path.isfile(filepath):
                if filename.startswith('cat'):
                    shutil.move(filepath, os.path.join(cats_path, filename))
                elif filename.startswith('dog'):
                    shutil.move(filepath, os.path.join(dogs_path, filename))
    print("Dataset organized.")

# 1️⃣ Organisiere Dataset
organize_dataset('data')

# 2️⃣ Gerät wählen: GPU oder CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Step 2️⃣: Device chosen: {device}")

# 3️⃣ Transformationen für Bilder
print("Step 3️⃣: Setting up image transformations...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),      # ### data augmentations generalization
    transforms.RandomRotation(10),               # ### data augmentations generalization
    transforms.ColorJitter(brightness=0.2),      # ### data augmentations generalization
    transforms.ToTensor(),
])
print("Transformations ready.")

# 4️⃣ Datasets
print("Step 4️⃣: Loading datasets...")
train_dataset = datasets.ImageFolder('data/train', transform=transform)
val_dataset = datasets.ImageFolder('data/val', transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
]))
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# 5️⃣ DataLoader
print("Step 5️⃣: Creating DataLoaders...")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print("DataLoaders ready.")

# 6️⃣ Modell
print("Step 6️⃣: Loading pretrained ResNet18 model...")
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Kopf anpassen für 2 Klassen (Cat/Dog)
model.fc = nn.Linear(model.fc.in_features, 2)

# ### one output between 0..1. activation function softmax
# Für CrossEntropyLoss wird *kein* Softmax im Modell verwendet, da CE intern softmax(logits) ausführt.
# Wenn man am Ende Softmax sehen will:
softmax = nn.Softmax(dim=1)

model = model.to(device)
print("Model loaded and moved to device.")

# 7️⃣ Loss und Optimizer
print("Step 7️⃣: Setting up loss function and optimizer...")

# DEFAULT: CrossEntropy
criterion = nn.CrossEntropyLoss()

# ### hyperparameter tuning / different learning size / different optimizer
LEARNING_RATE = 1e-4  # ### different learning size (ändert LR)
USE_ADAM = True       # ### different optimizer

if USE_ADAM:
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
else:
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

# ### binary crossentropy loss (Alternative)
# BCE würde ein 1-Output-Modell erfordern:
# model.fc = nn.Linear(model.fc.in_features, 1)
# criterion = nn.BCEWithLogitsLoss()
# -> Dann Labels als float und Sigmoid statt Softmax.
print("Loss and optimizer ready.")

# 8️⃣ Training
### plot loss over epochs
num_epochs = 3
epoch_losses = []

print("Step 8️⃣: Starting training...")
for epoch in range(num_epochs):
    print(f"  Epoch {epoch+1}/{num_epochs}...")
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader, 1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"    Batch {batch_idx}/{len(train_loader)}, Current Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    epoch_losses.append(avg_loss)
    print(f"  Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

# --- Plot Loss ---
plt.figure()
plt.plot(range(1, num_epochs + 1), epoch_losses)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_plot.png")
print("Loss plot saved as loss_plot.png")

# 9️⃣ Validation Accuracy
print("Step 9️⃣: Evaluating on validation set...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(val_loader, 1):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 10 == 0:
            print(f"    Evaluated {batch_idx * val_loader.batch_size} samples...")

print(f"Validation Accuracy: {100 * correct / total:.2f}%")
print("All steps completed.")

