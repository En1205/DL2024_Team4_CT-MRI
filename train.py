import os
import cv2
import numpy as np
import albumentations as A
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
import matplotlib.pyplot as plt


train_transform = A.Compose([
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

val_transform = A.Compose([
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=0),  # 同樣修正
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# Directories for dataset
train_img_dir = "show_output/brain_resized/train"
train_mask_dir = "show_output/tumor_resized/train"
val_img_dir   = "show_output/brain_resized/val"
val_mask_dir   = "show_output/tumor_resized/val"

# Create output directories if not exist
os.makedirs("checkpoints1", exist_ok=True)
os.makedirs("show_output1", exist_ok=True)

class BrainTumorDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        # List all image files
        self.image_files = sorted([f for f in os.listdir(images_dir) if not f.startswith('.')])
        # Ensure non-empty
        assert len(self.image_files) > 0, f"No images found in {images_dir}"
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        # Read image and mask
        # Use cv2 to read. For image: color, for mask: grayscale.
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Failed to read image or mask for index {idx}: {img_path}, {mask_path}")
        # Convert BGR (opencv) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # If mask is not binary (e.g., 255 and 0), convert to 0 and 1
        mask = (mask > 127).astype(np.uint8)  # assuming tumor mask is white (255)
        # Apply augmentation if specified
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        # Convert to torch tensors
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()  # from HWC to CHW
        mask = torch.from_numpy(mask).float()  # shape HxW, binary (0.,1.)
        return image, mask

# Initialize datasets and dataloaders
train_dataset = BrainTumorDataset(train_img_dir, train_mask_dir, transform=train_transform)
val_dataset   = BrainTumorDataset(val_img_dir, val_mask_dir, transform=val_transform)
# You can adjust batch_size based on your hardware capabilities
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=False)

# Define model – using segmentation_models.pytorch for UNet with ResNet34 encoder
# If not installed, you can install via pip: pip install segmentation-models-pytorch

model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation=None)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # binary cross-entropy with logits
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Metrics tracking
best_val_iou = 0.0
train_history = {"loss": [], "iou": [], "f1": []}
val_history   = {"loss": [], "iou": [], "f1": []}

# Training loop
num_epochs = 300  # you may adjust this or implement early stopping
for epoch in range(1, num_epochs+1):
    model.train()
    epoch_train_loss = 0.0
    # Metrics accumulators for training
    tp_train = fp_train = fn_train = 0  # true pos, false pos, false neg for F1/IoU
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # outputs shape [B, 1, H, W]
        # Compute loss
        loss = criterion(outputs, masks.unsqueeze(1))  # make mask shape [B,1,H,W] to match outputs
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() * images.size(0)
        # Compute train metrics on the fly (using threshold 0.5 on outputs)
        pred = (outputs.detach() > 0).float()  # sigmoid > 0.5 threshold; comparing logits > 0 is equivalent
        # Update confusion counts
        tp_train += torch.sum((pred == 1) & (masks.unsqueeze(1) == 1)).item()
        fp_train += torch.sum((pred == 1) & (masks.unsqueeze(1) == 0)).item()
        fn_train += torch.sum((pred == 0) & (masks.unsqueeze(1) == 1)).item()
    # Finish computing train loss and metrics
    epoch_train_loss /= len(train_dataset)  # average loss
    # Calculate IoU and F1 for training
    train_iou = tp_train / (tp_train + fp_train + fn_train + 1e-7)
    train_f1  = (2 * tp_train) / (2 * tp_train + fp_train + fn_train + 1e-7)
    train_history["loss"].append(epoch_train_loss)
    train_history["iou"].append(train_iou)
    train_history["f1"].append(train_f1)

    # Validation phase
    model.eval()
    epoch_val_loss = 0.0
    tp_val = fp_val = fn_val = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            epoch_val_loss += loss.item() * images.size(0)
            # Metrics: threshold at 0.5
            pred = (outputs > 0).float()
            tp_val += torch.sum((pred == 1) & (masks.unsqueeze(1) == 1)).item()
            fp_val += torch.sum((pred == 1) & (masks.unsqueeze(1) == 0)).item()
            fn_val += torch.sum((pred == 0) & (masks.unsqueeze(1) == 1)).item()
    epoch_val_loss /= len(val_dataset)
    val_iou = tp_val / (tp_val + fp_val + fn_val + 1e-7)
    val_f1  = (2 * tp_val) / (2 * tp_val + fp_val + fn_val + 1e-7)
    val_history["loss"].append(epoch_val_loss)
    val_history["iou"].append(val_iou)
    val_history["f1"].append(val_f1)

    # Print epoch summary
    print(f"Epoch {epoch}/{num_epochs} - "
          f"Train Loss: {epoch_train_loss:.4f}, IoU: {train_iou:.4f}, F1: {train_f1:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f}, IoU: {val_iou:.4f}, F1: {val_f1:.4f}")

    # Save best model if current val IoU is the highest
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(), "checkpoints/best_model.pt")
        print(f"  [*] Saved new best model (Val IoU = {val_iou:.4f})")

epochs = range(1, len(train_history["loss"]) + 1)
plt.figure(figsize=(12,4))
# Loss subplot
plt.subplot(1,3,1)
plt.plot(epochs, train_history["loss"], label="Train Loss")
plt.plot(epochs, val_history["loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch"); plt.legend()
# IoU subplot
plt.subplot(1,3,2)
plt.plot(epochs, train_history["iou"], label="Train IoU")
plt.plot(epochs, val_history["iou"], label="Val IoU")
plt.xlabel("Epoch"); plt.ylabel("IoU"); plt.title("IoU vs Epoch"); plt.legend()
# F1 subplot
plt.subplot(1,3,3)
plt.plot(epochs, train_history["f1"], label="Train F1")
plt.plot(epochs, val_history["f1"], label="Val F1")
plt.xlabel("Epoch"); plt.ylabel("F1-score"); plt.title("F1 vs Epoch"); plt.legend()

plt.tight_layout()
plt.savefig("show_output/training_metrics.png")
plt.close()
print("Training complete. Metrics plots saved to show_output/training_metrics.png")
