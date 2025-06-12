import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms

class BrainTumorDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.mask_list = sorted(os.listdir(mask_dir))
        self.use_transform = transform
        self.size = (256, 256)  # 設定 resize 的大小

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        # 使用 OpenCV 讀取為灰階 (0 表示單通道)
        image = cv2.imread(img_path, 0)  # shape: (H, W)
        mask = cv2.imread(mask_path, 0)  # shape: (H, W)

        if self.use_transform:
            # 確保圖像與 mask 的尺寸一致
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)

        # 轉換成 Tensor
        image = torch.from_numpy(image).float().unsqueeze(0) / 255.0   # (1, H, W)
        mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0
        mask = (mask > 0.5).float()  # 二值化 mask

        return image, mask
