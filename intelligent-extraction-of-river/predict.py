import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import rasterio
from skimage.transform import resize
import cv2

# 输入路径
tiff_path = "/home/zmw/sam2/image6/input/test.tif"
input_image_path = "/home/zmw/sam2/image6/input/test.jpg"
sam2_checkpoint = "/home/zmw/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# 输出目录（只输出tif）
binary_mask_tiff_dir = '/home/zmw/sam2/image6/output/binary_mask_tiff'
os.makedirs(binary_mask_tiff_dir, exist_ok=True)

# ------------- 读取图片并打点 ----------------
image = Image.open(input_image_path)
image = np.array(image.convert("RGB"))
input_points = []
input_labels = []

def onclick(event):
    if event.inaxes is not None:
        x, y = int(event.xdata), int(event.ydata)
        if event.button == 1:
            label = 1
            color = 'green'
        elif event.button == 3:
            label = 0
            color = 'red'
        else:
            return
        print(f"Clicked at: ({x}, {y}) with label {label}")
        input_points.append([x, y])
        input_labels.append(label)
        plt.scatter(x, y, color=color, marker='*', s=50, edgecolor='white', linewidth=1.25)
        plt.draw()

def onscroll(event):
    ax = event.inaxes
    if ax is None: return
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    xdata = event.xdata if event.xdata is not None else (cur_xlim[0] + cur_xlim[1]) / 2
    ydata = event.ydata if event.ydata is not None else (cur_ylim[0] + cur_ylim[1]) / 2

    if event.button == 'up': scale_factor = 1 / 1.2
    elif event.button == 'down': scale_factor = 1.2
    else: scale_factor = 1

    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
    relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
    rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

    ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * relx])
    ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * rely])
    ax.figure.canvas.draw_idle()

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)
ax.set_title("左键前景，右键背景，点完关窗口")
ax.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('scroll_event', onscroll)
plt.show()

input_point = np.array(input_points)
input_label = np.array(input_labels)

# ------------ SAM2 segmentation -------------
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")
predictor = SAM2ImagePredictor(sam2_model)
predictor.set_image(image)
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

# 只保留分数最高的mask
best_mask = masks[0]

# ---------- 导出带地理信息的tif ----------
with rasterio.open(tiff_path) as src:
    tif_profile = src.profile.copy()
    tif_shape = src.read(1).shape

binary_mask = (best_mask >= 0.5).astype(np.uint8) * 255
if binary_mask.shape != tif_shape:
    print("掩码和TIF分辨率不一致，正在缩放……")
    binary_mask = resize(binary_mask, tif_shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

geo_mask_path = os.path.join(binary_mask_tiff_dir, 'geo_mask.tif')
tif_profile.update(driver='GTiff', dtype=np.uint8, count=1, nodata=0)
with rasterio.open(geo_mask_path, 'w', **tif_profile) as dst:
    dst.write(binary_mask, 1)
print(f"分数最高的掩码（GeoTIFF）已保存: {geo_mask_path}")