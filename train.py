import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import BrainTumorDataset
from unet import UNet
from utils import hybrid_loss, iou_score
import os
import matplotlib.pyplot as plt
import numpy as np

# 設定超參數
EPOCHS = 100
BATCH_SIZE = 8
LR = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 10  # EarlyStopping 忍耐次數


# EarlyStopping 模組
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(f"🛑 EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def visualize_validation_results(model, val_loader, epoch, num_samples=3):
    """在訓練過程中視覺化驗證結果"""
    model.eval()

    # 創建保存資料夾
    os.makedirs("training_validation_viz", exist_ok=True)

    with torch.no_grad():
        # 只取前幾個樣本進行視覺化
        sample_count = 0
        for imgs, masks in val_loader:
            if sample_count >= num_samples:
                break

            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            # 模型預測
            preds = model(imgs)
            pred_masks = torch.sigmoid(preds)  # 轉換為機率
            pred_binary = (pred_masks > 0.5).float()  # 二值化

            # 轉換為 numpy 陣列
            img_np = imgs[0].cpu().squeeze().numpy()
            true_mask_np = masks[0].cpu().squeeze().numpy()
            pred_mask_np = pred_binary[0].cpu().squeeze().numpy()

            # 計算 IoU
            intersection = np.logical_and(true_mask_np, pred_mask_np).sum()
            union = np.logical_or(true_mask_np, pred_mask_np).sum()
            iou = intersection / union if union > 0 else 1.0

            # 創建視覺化圖像
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 原始 MRI 影像
            axes[0].imshow(img_np, cmap='gray')
            axes[0].set_title('MRI Image', fontsize=12, fontweight='bold')
            axes[0].axis('off')

            # 真實遮罩
            axes[1].imshow(true_mask_np, cmap='gray')
            axes[1].set_title('True Mask', fontsize=12, fontweight='bold')
            axes[1].axis('off')

            # 預測遮罩
            axes[2].imshow(pred_mask_np, cmap='gray')
            axes[2].set_title(f'Predicted Mask\nIoU: {iou:.3f}', fontsize=12, fontweight='bold')
            axes[2].axis('off')

            plt.suptitle(f'Epoch {epoch + 1} - Validation Sample {sample_count + 1}', fontsize=14, fontweight='bold')
            plt.tight_layout()

            # 保存圖像
            plt.savefig(f"training_validation_viz/epoch_{epoch + 1}_sample_{sample_count + 1}.png",
                        dpi=100, bbox_inches='tight')
            plt.close()  # 關閉圖像以節省記憶體

            sample_count += 1

    model.train()  # 切回訓練模式


def create_combined_visualization(model, val_loader, epoch, num_samples=6):
    """創建組合視覺化圖像（多個樣本在一張圖上）"""
    model.eval()

    samples_data = []
    sample_count = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            if sample_count >= num_samples:
                break

            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            pred_masks = torch.sigmoid(preds)
            pred_binary = (pred_masks > 0.5).float()

            # 儲存資料
            img_np = imgs[0].cpu().squeeze().numpy()
            true_mask_np = masks[0].cpu().squeeze().numpy()
            pred_mask_np = pred_binary[0].cpu().squeeze().numpy()

            # 計算 IoU
            intersection = np.logical_and(true_mask_np, pred_mask_np).sum()
            union = np.logical_or(true_mask_np, pred_mask_np).sum()
            iou = intersection / union if union > 0 else 1.0

            samples_data.append({
                'img': img_np,
                'true_mask': true_mask_np,
                'pred_mask': pred_mask_np,
                'iou': iou
            })

            sample_count += 1

    # 創建大圖
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i, data in enumerate(samples_data):
        # MRI 影像
        axes[i, 0].imshow(data['img'], cmap='gray')
        axes[i, 0].set_title('MRI Image' if i == 0 else '', fontsize=10)
        axes[i, 0].axis('off')

        # 真實遮罩
        axes[i, 1].imshow(data['true_mask'], cmap='gray')
        axes[i, 1].set_title('True Mask' if i == 0 else '', fontsize=10)
        axes[i, 1].axis('off')

        # 預測遮罩
        axes[i, 2].imshow(data['pred_mask'], cmap='gray')
        axes[i, 2].set_title(f'Predicted Mask' if i == 0 else f'IoU: {data["iou"]:.3f}', fontsize=10)
        axes[i, 2].axis('off')

        # 左側加上樣本編號
        axes[i, 0].text(-0.1, 0.5, f'Sample {i + 1}', transform=axes[i, 0].transAxes,
                        rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')

    plt.suptitle(f'Epoch {epoch + 1} - Validation Results Overview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"training_validation_viz/epoch_{epoch + 1}_overview.png",
                dpi=150, bbox_inches='tight')
    plt.close()  # 關閉圖像以節省記憶體

    model.train()


# 載入資料
train_dataset = BrainTumorDataset("unet_dataset/train/brain", "unet_dataset/train/tumor")
val_dataset = BrainTumorDataset("unet_dataset/validation/brain", "unet_dataset/validation/tumor")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 建立模型
model = UNet(in_channels=1, out_channels=1).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
bce_loss = nn.BCEWithLogitsLoss()
early_stopper = EarlyStopping(patience=PATIENCE)

# 創建保存資料夾
os.makedirs("training_validation_viz", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# 顯示第一張影像與遮罩
print("保存訓練資料樣本...")
for img, mask in train_loader:
    img_np = img[0].squeeze().numpy()
    mask_np = mask[0].squeeze().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("MRI Image")
    plt.imshow(img_np, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Tumor Mask")
    plt.imshow(mask_np, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("training_validation_viz/train_sample.png", dpi=150, bbox_inches='tight')
    plt.close()
    break

# 訓練與驗證流程
train_losses, val_losses, val_ious = [], [], []
best_iou = 0
EPOCHS = 50

print(f"開始訓練... (共 {EPOCHS} epochs)")
print(f"驗證視覺化將每 5 epochs 保存一次")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    # 訓練階段
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        preds = model(imgs)
        loss = hybrid_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 驗證階段
    model.eval()
    val_loss, val_iou = 0, 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            val_loss += hybrid_loss(preds, masks).item()
            val_iou += iou_score(preds, masks)

    train_losses.append(total_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    val_ious.append(val_iou / len(val_loader))

    print(
        f"[Epoch {epoch + 1}/{EPOCHS}] Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, IoU: {val_ious[-1]:.4f}")

    # 保存最佳模型
    if val_ious[-1] > best_iou:
        best_iou = val_ious[-1]
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Best model saved!")

    # 每5個epoch或最後一個epoch保存驗證視覺化
    if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
        print(f"\n保存 Epoch {epoch + 1} 驗證結果...")

        # 個別樣本視覺化
        visualize_validation_results(model, val_loader, epoch, num_samples=2)

        # 組合視覺化
        create_combined_visualization(model, val_loader, epoch, num_samples=3)

    # Early Stopping
    early_stopper(val_ious[-1])
    if early_stopper.early_stop:
        print("Early stopping triggered.")
        break

print("🎉 Training finished!")

# 畫出學習曲線
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_losses, label='Val Loss', linewidth=2)
plt.title('Loss over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(val_ious, label='Val IoU', linewidth=2, color='green')
plt.title('IoU over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/learning_curve.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"最佳 IoU: {best_iou:.4f}")
print(f"訓練過程的驗證視覺化已保存在 'training_validation_viz' 資料夾中")
print(f"學習曲線已保存在 'plots/learning_curve.png'")