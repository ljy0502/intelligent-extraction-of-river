#此代码是读取jpg/png等非tiff图片
import torch
import numpy as np


import matplotlib
matplotlib.use('TkAgg')  # 强制使用 Tkinter 交互式后端
import matplotlib.pyplot as plt

from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os
import cv2


torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 显示并保存分割掩码
def save_mask(mask, image, mask_index, save_dir, random_color=False, borders=True):
    mask_file_path = os.path.join(save_dir, f'mask_{mask_index+1}.png')

    # 创建输出目录
    os.makedirs(save_dir, exist_ok=True)

    # 绘制并保存掩码图像
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    show_mask(mask, ax, random_color=random_color, borders=borders)
    ax.axis('off')  # 不显示坐标轴
    plt.savefig(mask_file_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # 关闭图像，释放内存

def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.3])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    # 显示提示点：前景点为绿色，背景为红色
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    # 显示坐标框
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, input_labels=None, borders=True, save_dir=None):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        if save_dir:
            save_mask(mask, image, i, save_dir, random_color=False, borders=borders)  # 保存分割掩码
            
            # 将分割掩码转换为二值图并保存
            binary_mask = (mask >= 0.5).astype(np.uint8) * 255  # 转换为二值图，阈值为0.5
            binary_mask_file_path = os.path.join(save_dir, f'binary_mask_{i+1}.png')
            cv2.imwrite(binary_mask_file_path, binary_mask)  # 使用cv2保存二值图
        else:
            # 如果没有指定保存路径，则显示图像
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(mask, plt.gca(), borders=borders)
            if point_coords is not None:
                assert input_labels is not None
                show_points(point_coords, input_labels, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()



# 加载图像并进行预处理
image=Image.open("/home/zmw/sam2/image2/heliu.jpg")
image=np.array(image.convert("RGB"))


# # 交互式获取前景点
# def onclick(event):
#     if event.inaxes is not None:
#         x, y = int(event.xdata), int(event.ydata)
#         print(f"Clicked at: ({x}, {y})")
#         input_points.append([x, y])
#         input_labels.append(1)  # 假设所有点击点都是前景点
#         plt.scatter(x, y, color='green', marker='*', s=375, edgecolor='white', linewidth=1.25)
#         plt.draw()

# input_points = []
# input_labels = []

# fig, ax = plt.subplots(figsize=(10, 10))
# ax.imshow(image)
# ax.set_title("Click on the image to select foreground points. Close the window when done.")
# fig.canvas.mpl_connect('button_press_event', onclick)
# plt.show()

# input_point = np.array(input_points)
# input_label = np.array(input_labels)

# 交互式获取前景点和后景点
def onclick(event):
    if event.inaxes is not None:
        x, y = int(event.xdata), int(event.ydata)
        if event.button == 1:  # 左键点击为前景点
            label = 1
            color = 'green'
        elif event.button == 3:  # 右键点击为后景点
            label = 0
            color = 'red'
        else:
            return  # 忽略其他按钮
        
        print(f"Clicked at: ({x}, {y}) with label {label}")
        input_points.append([x, y])
        input_labels.append(label)
        plt.scatter(x, y, color=color, marker='*', s=375, edgecolor='white', linewidth=1.25)
        plt.draw()

def onwheel(event):
    if event.inaxes is not None:
        ax = event.inaxes
        x, y = event.xdata, event.ydata
        
        # 获取当前的缩放级别
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # 计算新的缩放级别
        scale_factor = 1.1 if event.button == 'up' else 0.9
        
        # 更新缩放级别
        new_xlim = (xlim[0] * scale_factor, xlim[1] * scale_factor)
        new_ylim = (ylim[0] * scale_factor, ylim[1] * scale_factor)
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        
        # 重新绘制图像
        ax.figure.canvas.draw_idle()

input_points = []
input_labels = []

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image)
ax.set_title("Left click for foreground points, right click for background points. Close the window when done.")
ax.axis('off')  # 关闭坐标轴
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('scroll_event', onwheel)
plt.show()

input_point = np.array(input_points)
input_label = np.array(input_labels)

# 加载sam2模型
sam2_checkpoint="/home/zmw/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"
sam2_model=build_sam2(model_cfg,sam2_checkpoint,device="cuda")
predictor=SAM2ImagePredictor(sam2_model)
predictor.set_image(image)

#使用交互式获取的前景点进行分割
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]


# masks.shape  # (number_of_masks) x H x W

# (3, 1200, 1800)

# 显示分割结果
save_dir = '/home/zmw/sam2/image2/masks'
show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True, save_dir=save_dir)
print("完成了")





