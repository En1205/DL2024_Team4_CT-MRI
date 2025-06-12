import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from unet import UNet

# 設定裝置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型初始化與載入
model = UNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

# 圖像前處理
transform = transforms.Compose([
    transforms.ToTensor()
])

def predict_image(path):
    # 用 OpenCV 讀取圖像，保留原始色彩
    img_bgr = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # resize + to tensor
    img_gray_resized = cv2.resize(img_gray, (256, 256))
    input_tensor = transform(img_gray_resized).unsqueeze(0).to(DEVICE)  # [1, 1, 256, 256]


    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (pred > 0.5).astype(np.uint8) * 255

    # resize 原圖為 256x256
    img_rgb_resized = cv2.resize(img_rgb, (256, 256))

    return img_rgb_resized, mask

# 顯示圖片
def show_prediction():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    original_rgb, mask = predict_image(file_path)

    # 建立紅色遮罩 overlay
    overlay = original_rgb.copy()
    red_mask = np.zeros_like(original_rgb)
    red_mask[:, :, 0] = mask  # R channel

    overlay = cv2.addWeighted(original_rgb, 0.5, red_mask, 0.5, 0)

    # 轉成 PIL 格式給 Tkinter 顯示
    original_pil = Image.fromarray(original_rgb)
    overlay_pil = Image.fromarray(overlay)

    original_tk = ImageTk.PhotoImage(original_pil)
    overlay_tk = ImageTk.PhotoImage(overlay_pil)

    original_label.config(image=original_tk)
    original_label.image = original_tk

    result_label.config(image=overlay_tk)
    result_label.image = overlay_tk

# 建立 GUI 介面
window = tk.Tk()
window.title("腦腫瘤預測 GUI")
window.geometry("600x350")

btn = tk.Button(window, text="選擇 MRI 影像進行預測", command=show_prediction, font=("Arial", 14))
btn.pack(pady=10)

frame = tk.Frame(window)
frame.pack()

original_label = tk.Label(frame)
original_label.grid(row=0, column=0, padx=10)
tk.Label(frame, text="原始 MRI", font=("Arial", 12)).grid(row=1, column=0)

result_label = tk.Label(frame)
result_label.grid(row=0, column=1, padx=10)
tk.Label(frame, text="預測結果", font=("Arial", 12)).grid(row=1, column=1)

window.mainloop()
