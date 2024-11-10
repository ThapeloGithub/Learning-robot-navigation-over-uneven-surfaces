import os
from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Set dataset paths
data_dir = r"C:/Users/thapelo/Downloads/Quadruped_Robot/QUADRUPED_ROBOT/CAPTURED" 

# Check if dataset exists
if not os.path.exists(data_dir):
    raise ValueError("Dataset directory does not exist!")

# Define transformations with data augmentation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_data = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transforms)
val_data = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transforms)

# DataLoaders
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

# Check class-to-index mapping
class_names = train_data.classes
print(f"Classes: {class_names}")

# Load pretrained model
model = models.resnet18(pretrained=True)

# Freeze the model's earlier layers
for param in model.parameters():
    param.requires_grad = False

num_classes = len(class_names)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_classes),
    nn.LogSoftmax(dim=1)
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    train_loss_history, val_loss_history = [], []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_loss_history.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_loss = val_loss / len(val_loader)
        val_loss_history.append(val_loss)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return train_loss_history, val_loss_history

# Train the model
epochs = 15
train_loss, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)

# Save the trained model
model_save_path = r"C:/Users/thapelo/Downloads/Quadruped_Robot/Quadruped_robot/classifier/terrain_classifier.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved at {model_save_path}")

# Plot training and validation loss
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend()
plt.show()
