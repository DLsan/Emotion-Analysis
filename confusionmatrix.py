import torch
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src.model import get_model
from src.dataset_loader import make_dataloaders

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
train_loader, val_loader, test_loader, class_names, _ = make_dataloaders(
    root="FER-2013",
    batch_size=32,
    num_workers=2,
)

# Load model
model = get_model(num_classes=len(class_names))
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — FER-2013")
plt.show()

print(classification_report(all_labels, all_preds, target_names=class_names))
