import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

# Directories for test images and masks
test_img_dir = "show_output/brain_resized/test"
test_mask_dir = "show_output/tumor_resized/test"
output_vis_dir = "show_output1/predictions"
os.makedirs(output_vis_dir, exist_ok=True)

# Define the same Dataset class (as above) and transformation for test
class BrainTumorDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if not f.startswith('.')])
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask > 127).astype(np.uint8)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask).float()
        return img_name, image, mask

# Normalization transform for test (same as val)
test_transform = A.Compose([
    A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=0),  # 這是關鍵
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
test_dataset = BrainTumorDataset(test_img_dir, test_mask_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the model (make sure the model architecture matches the saved one)
import segmentation_models_pytorch as smp
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
# Note: encoder_weights=None means we aren't loading ImageNet weights now, not needed since we'll load our trained weights.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# Load saved best model weights
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
model.eval()

# Metrics counters
TP = FP = FN = TN = 0  # true positive, false positive, false negative, true negative
# Loop over test set and make predictions
for img_name, image, mask in test_loader:
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        output = model(image)  # output shape [1,1,H,W]
        pred = (output > 0).float()  # binary prediction mask (logit > 0 -> class 1)
    # Update confusion matrix counts
    # Flatten to 1D for simplicity
    pred_flat = pred.view(-1)
    mask_flat = mask.view(-1)
    TP += torch.sum((pred_flat == 1) & (mask_flat == 1)).item()
    FP += torch.sum((pred_flat == 1) & (mask_flat == 0)).item()
    FN += torch.sum((pred_flat == 0) & (mask_flat == 1)).item()
    TN += torch.sum((pred_flat == 0) & (mask_flat == 0)).item()
    # Create visualization image (side by side)
    # Detach and convert tensors to numpy and then to proper format for saving
    image_np = image.cpu().numpy()[0].transpose(1, 2, 0)  # CHW to HWC
    # Denormalize the image for visualization (inverse of mean/std normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_vis = (image_np * std + mean) * 255.0  # back to [0,255] range
    image_vis = np.clip(image_vis, 0, 255).astype(np.uint8)
    # Convert to BGR for saving with cv2
    image_vis_bgr = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
    mask_gt = mask.cpu().numpy()[0]  # shape HxW, values 0/1
    mask_pred = pred.cpu().numpy()[0,0]  # shape HxW, values 0/1
    # We will color the masks in red for visualization. Create 3-channel masks:
    red_mask_gt = np.zeros_like(image_vis_bgr)
    red_mask_pred = np.zeros_like(image_vis_bgr)
    # Color the tumor region in red (BGR: (0,0,255))
    red_mask_gt[mask_gt == 1] = (0, 0, 255)
    red_mask_pred[mask_pred == 1] = (0, 0, 255)
    # Overlay the red mask on the original image with some transparency
    overlay_gt = cv2.addWeighted(image_vis_bgr, 0.7, red_mask_gt, 0.3, 0)
    overlay_pred = cv2.addWeighted(image_vis_bgr, 0.7, red_mask_pred, 0.3, 0)
    # Compose side-by-side: original vs ground truth vs prediction
    # We can also put text labels on top of images if desired
    combined = np.concatenate([image_vis_bgr, overlay_gt, overlay_pred], axis=1)
    save_path = os.path.join(output_vis_dir, f"{img_name[0]}_prediction.png")
    cv2.imwrite(save_path, combined)
    print(f"Saved prediction visualization: {save_path}")

# Calculate final metrics
iou = TP / (TP + FP + FN + 1e-7)
f1  = (2 * TP) / (2 * TP + FP + FN + 1e-7)
print("Test set evaluation:")
print(f" Confusion Matrix: TP={TP}, FP={FP}, FN={FN}, TN={TN}")
print(f" IoU = {iou:.4f}")
print(f" F1-score (Dice) = {f1:.4f}")
