import os
import sys
import warnings
import numpy as np
sys.path.append("/hy-tmp/explore_attention_vit/dinov2")
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


__all__ = ['00005.jpg', '00008.jpg',  '00026.jpg', '00180.jpg', '00884.jpg', '01067.jpg']

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

    file = '01067.jpg'
    original_image = Image.open(os.path.join('/hy-tmp/explore_attention_vit/dior_train/', file))
    (original_w, original_h) = original_image.size
    img = data_transforms(original_image)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h]

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    img = img.unsqueeze(0)
    img = img.to(device)

    q_k_attn = model.get_last_self_attention(img.to(device))
                                                          
    q = q_k_attn[0]
    k = q_k_attn[1]
    print(q_k_attn[0].shape, q_k_attn[1].shape, q_k_attn[2].shape)

    number_of_heads = q.shape[1]
    n_register_tokens = 4
    
    # q[:, :, 0:1, :]=0
    k[:, :, 0:1, :]=0

    # q_without_cls = q[:, :, 1:, :]         ####
    # k_without_cls = k[:, :, 1:, :]         ####
    
    attn_without_cls = q @ k.transpose(-2, -1)
    attn_without_cls = attn_without_cls.softmax(dim=-1)
    attn_without_cls = nn.Dropout(0.0)(attn_without_cls)

    attn_without_cls = attn_without_cls[0, :, 0, 5:].reshape(number_of_heads, -1)
    attn_without_cls = attn_without_cls.reshape(number_of_heads, w_featmap, h_featmap)
    attn_without_cls = torch.sum(attn_without_cls, dim=0)


    """画出热力图（可以选择其中一个维度的特征）"""
    plt.figure(figsize=(40, 32))
    sns.heatmap(attn_without_cls.cpu().numpy(), cmap='viridis', cbar=True)
    plt.title("Visualization of heatmap_without_cls_"+file.split('.')[0])
    plt.xlabel("Dimensionality")
    plt.ylabel("Sequence Length")


    """将图像保存到文件"""
    print(os.path.join("/hy-tmp/explore_attention_vit/attention_with_cls_token_as_zero/", "heatmap_with_k_cls_token_zero_"+file.split('.')[0]+'.png'))
    plt.savefig(os.path.join("/hy-tmp/explore_attention_vit/attention_with_cls_token_as_zero/", "heatmap_with_k_cls_token_zero_"+file.split('.')[0]+'.png'), dpi=300)
    plt.close() 

