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

    file = '00044.jpg'
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

    q_k_attn = model.get_last_self_attention(img.to(device))
    print(q_k_attn[0].shape, q_k_attn[1].shape, q_k_attn[2].shape)
                                                          
    q = q_k_attn[0]
    k = q_k_attn[1]
    
    q_cls_token = q[:, :, 0:1, :]
    print(q_cls_token.shape)

    k_cls_token = k[:, :, 0:1, :]

    number_of_heads = q.shape[1]
    n_register_tokens = 4
    
    q_cls_token =torch.sum(q_cls_token[0], dim=0).cpu().numpy().reshape(1, -1)
    print(q_cls_token.shape)

    # kmeans = KMeans(n_clusters=2, random_state=42)
    # labels = kmeans.fit_predict(q_cls_token)

            # tsne = TSNE(n_components=2, random_state=42)    # 使用t-SNE进行降维到2D
    # q_tsne = tsne.fit_transform(q_cls_token)

    
    """绘制t-SNE降维后的2D散点图 """
    # plt.figure(figsize=(10, 8))
    # plt.scatter(q_tsne[:, 0], q_tsne[:, 1], s=2, c=q_tsne[:, 0], cmap='viridis', alpha=0.5)
    # # scatter = plt.scatter(q_tsne[:, 0], q_tsne[:, 1], c=labels, cmap='viridis', s=20, alpha=0.7)
    # plt.title("t-SNE Visualization of 00044 cls token")
    # plt.xlabel("t-SNE Component 1")
    # plt.ylabel("t-SNE Component 2")
  
    plt.scatter(q_cls_token[0], q_cls_token[1], color='red', label='Sample')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Feature Analysis of 1 Sample")
  
    """将图像保存到文件"""
    plt.savefig("/hy-tmp/hengyuanyun/attention_without_class_token/t-sne/cls_token_sne_00044.png", dpi=300) 
    plt.close() 

   

    