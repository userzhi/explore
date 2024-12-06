import os
import sys
import warnings
import numpy as np
sys.path.append("/hy-tmp/hengyuanyun/dinov2")
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from matplotlib.colors import Normalize
from IPython.display import display_png 

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from dinov2.models.vision_transformer import vit_small, vit_base, vit_large

# prototype = torch.load('/hy-tmp/dior_N10/prototypes.pth')
# print(prototype['prototype'])

if 1:
    # # These are settings for ensuring input images to DinoV2 are properly sized
    model = vit_base(
            patch_size=14,
            img_size=526,
            init_values=1.0,
            num_register_tokens=4,
            block_chunks=0
            )

    class ResizeAndPad:
        def __init__(self, target_size, multiple):
            self.target_size = target_size
            self.multiple = multiple

        def __call__(self, img):
            # Resize the image
            img = transforms.Resize(self.target_size)(img)

            # Calculate padding
            pad_width = (self.multiple - img.width % self.multiple) % self.multiple
            pad_height = (self.multiple - img.height % self.multiple) % self.multiple

            # Apply padding
            img = transforms.Pad((pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2))(img)
            
            return img

    image_dimension = 602
        
    # This is what DinoV2 sees
    target_size = (image_dimension, image_dimension)

    # During inference / testing / deployment, we want to remove data augmentations from the input transform:
    data_transforms = transforms.Compose([ ResizeAndPad(target_size, 14),
                                        transforms.CenterCrop(image_dimension),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        ])

    patch_size = 14
    device = "cuda"

    model.load_state_dict(torch.load('/hy-tmp/dinov2_vitb14_reg4_pretrain.pth'))
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()

    # path = "/root/autodl-tmp/simd/train"
    # for file in os.listdir(path):
    # for head in range(12):

    file = '00005.jpg'
    original_image = Image.open(os.path.join('/hy-tmp/hengyuanyun/dior_train/', file))
    (original_w, original_h) = original_image.size
    img = data_transforms(original_image)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h]

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    img = img.unsqueeze(0)
    img = img.to(device)


    attention = model.get_last_self_attention(img.to(device))
                                                          
    # q = attention[0]
    print(attention[0].shape, attention[1].shape, attention[2].shape)
    attention = attention[2]
    print(attention.shape)
    
    q = torch.sum(attention[0], dim=0)
    print(q.shape)
    q_head = q.cpu().numpy().reshape(1854, -1)      # 选择第一个头的所有序列位置的64维向量
    


    """画出热力图（可以选择其中一个维度的特征）"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(q_head, cmap='plasma', cbar=True)
    # plt.imshow(q_head.cpu().numpy(), cmap='gray')
    plt.title("Visualization of Query for Head 1")
    plt.xlabel("Dimensionality")
    plt.ylabel("Sequence Length")

    # # 将图像保存到文件
    plt.savefig("query_visualization.png", dpi=300) 
    plt.close() 

    # number_of_heads = attention.shape[1]
    # n_register_tokens = 4

    """  rewight from object location
    att = attention[:, :, 0, 1 + n_register_tokens:].reshape(1, 12, -1)
    att = att.reshape(1, 12, w_featmap, h_featmap)
    threshold = torch.mean(att.reshape(12, -1), dim=1)
    print(threshold)
    Q = torch.sum(
        att.reshape(1, 12, w_featmap * h_featmap) > threshold[:, None, None], axis=2
    ) / (w_featmap * h_featmap)
    beta = torch.log(torch.sum(Q + 1e-10, dim=1)[:, None] / (Q + 1e-10))
    att = att.reshape(1, 12, w_featmap, h_featmap) * beta[:, :, None, None]
    """

    # # attention tokens are packed in after the first token; the spatial tokens follow
    # attention = attention[0, :, 0, 1 + n_register_tokens:].reshape(number_of_heads, -1)
    # """
    #     从多维张量中提取特定的注意力权重，并调整成适合处理的形状，跳过某些特定注册tokens
    #     attention shape: torch.Size([12, 4624])
    # """
    # # print(attention.shape)

    # attention = attention.reshape(number_of_heads, w_featmap, h_featmap)
    # """
    #     attention shape: torch.Size([12, 68, 68]) 
    #     952 / 14 = 68   image_size / patch_size
    #     68 * 68 = 4624
    # """
    # # attention = att[0]                 #######re1


    # # attention = nn.functional.interpolate(attention.unsqueeze(0), scale_factor=patch_size, mode = "nearest")[0].cpu()       #####不需要此操作
    # """
    #     interpolate: 上采样对矩阵注意力操作
    #     attention shape: torch.Size([12, 952, 952])
    # """

    attention = torch.sum(attention, dim=0)
    """
        attention shape: torch.Size([952, 952])
    """
    plt.figure(figsize=(20, 16))
    sns.heatmap(attention.cpu().numpy(), cmap='viridis', cbar=True)
    # plt.imshow(q_head.cpu().numpy(), cmap='gray')
    plt.title("Visualization of Query for Head 1")
    plt.xlabel("Dimensionality")
    plt.ylabel("Sequence Length")

    # 将图像保存到文件
    plt.savefig("query_visualization00032.png", dpi=300)  # 保存为 PNG 文件
    plt.close()  # 关闭图形
    # # plt.show()


    # attention_of_image = nn.functional.interpolate(attention.unsqueeze(0).unsqueeze(0), size=(original_h, original_w), mode='bilinear', align_corners=False)
    # attention_of_image = attention_of_image.squeeze()
    # """
    #     对注意力矩阵进行双线性插值上采样，并将其调整为原始图像大小
    #     attention shape: torch.Size([1195, 800])
    # """


    # """以下代码主要是将注意力矩阵转换为图像并于原始图像叠加"""
    # attention_of_image = attention_of_image.to("cpu")
    # image_metric = attention_of_image.numpy()
    # normalized_metric = Normalize(vmin=image_metric.min(), vmax=image_metric.max())(image_metric)
    # # 对image_metric进行归一化处理，将其值域映射到[0, 1]之间
    # # Normalize是一个可以按最大最小值进行归一化的类


    # reds = plt.cm.Reds(normalized_metric)
    # # 通过plt.cm.Reds将归一化后的image_metric应用红色调色板
    # # Reds是一个包含RGB和alpha通道的数组，表示按Reds调色板渲染后的颜色


    # alpha_max_value = 1.00  
    # # 设置alpha通道的最大值（完全不透明）

    # # Adjust this value as needed to enhance lower values visibility
    # gamma = 0.5  
    # # 应用伽马校正来增强较低值的可视性。值小于 1 的伽马可以增强较低亮度区域的可见度。

    # # Apply gamma transformation to enhance lower values
    # enhanced_metric = np.power(normalized_metric, gamma)
    # # 通过伽马变换增强较低数值区域的可视性。

    # # Create the alpha channel with enhanced visibility for lower values
    # alpha_channel = enhanced_metric * alpha_max_value
    # # 创建 alpha 通道，增强后的值将乘以最大 alpha 值，决定每个像素的透明度。


    # # Add the alpha channel to the RGB data
    # rgba_mask = np.zeros((image_metric.shape[0], image_metric.shape[1], 4))
    # # 创建一个零初始化的 rgba_mask，其形状为 (height, width, 4)，表示每个像素有 RGBA 四个通道。
    # rgba_mask[..., :3] = reds[..., :3]  # RGB
    # # 将应用了 Reds 调色板的 RGB 数据放入 RGBA 图像中的 RGB 部分。
    # rgba_mask[..., 3] = alpha_channel  # Alpha
    # # 将计算得到的 alpha 通道放入 RGBA 图像中的 Alpha 部分。


    # # Convert the numpy array to PIL Image
    # rgba_image = Image.fromarray((rgba_mask * 255).astype(np.uint8))
    # # 将 NumPy 数组转换为 PIL 图像对象，并将其数值范围从 [0, 1] 转换为 [0, 255]，以便生成有效的图像

    # # Save the image
    # mask_file = os.path.join('/hy-tmp/hengyuanyun/reweight_attention/mask_image', file.split(".")[0]+".png")
    # state = rgba_image.save(mask_file)
    # # 保存生成的RGBA图像
    # attention_mask_image = Image.open(mask_file)
    # # 读取注意力图像

    # # Ensure both images are in the same mode
    # if original_image.mode != 'RGBA':
    #     original_image = original_image.convert('RGBA')
    # # 检查原始图像是不是RGBA

    # # Overlay the second image onto the first image
    # # The second image must be the same size as the first image
    # original_image.paste(attention_mask_image, (0, 0), attention_mask_image)
    # """使用paste()方法将注意力图像叠加到原始图像"""

    # # Save or show the combined image
    # original_image.save(os.path.join('/hy-tmp/hengyuanyun/reweight_attention/attention_map_image', file.split(".")[0]+".png"))