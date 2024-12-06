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
import torch.nn.functional as F
from torchvision import datasets, transforms

from dinov2.models.vision_transformer import vit_small, vit_base, vit_large


__all__ = ['00005.jpg', '00008.jpg',  '00026.jpg', '00180.jpg', '00884.jpg', '01067.jpg']

def count_similarity(nm1, nm2):
    nm1 = nm1.cpu().numpy()
    nm2 = nm2.cpu().numpy()
    dot_product = np.dot(nm1, nm2)
    norm_nm1 = np.linalg.norm(nm1)
    norm_nm2 = np.linalg.norm(nm2)
    consine_sim = dot_product
    return consine_sim

np.set_printoptions(threshold=np.inf)
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
    
    fg_prototype = torch.load('/hy-tmp/dior_N10/prototypes.pth')
    bg_prototype = torch.load('/hy-tmp/dior_N10/bg_prototypes_dinov2.pt')
    # for file in ['00005.jpg', '00008.jpg',  '00026.jpg', '00180.jpg', '00884.jpg', '01067.jpg']:
    for file in ['00008.jpg']:

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
                                                            
        # k = q_k_attn[1]
        # k = torch.sum(k, dim=1) / 12.0
        # print(k[0, 0, 8, :])

        with open(f'/hy-tmp/explore_attention_vit/similarity/sim{file}.txt', 'a') as f:
            for p in range(5, 1854):
                k = q_k_attn[1]
                pa = k[0, 0, p, :].repeat(1024//64)[:1024]
                
                fg_sim = []
                bg_sim = []
                for i in range(len(fg_prototype['prototypes'])):
                    fg_sim.append(count_similarity(pa, fg_prototype['prototypes'][i]))
                
                for j in range(len(bg_prototype['prototypes'])):
                    bg_sim.append(count_similarity(pa, bg_prototype['prototypes'][j]))
                
                # f.write(f'The sim between {p}th patch and fg_pro is {fg_sim}\n')
                # f.write(f'The sim between {p}th patch and bg_pro is {bg_sim}\n')

                x1 = np.arange(len(fg_sim))
                x2 = np.arange(len(bg_sim))
                
                fig, ax = plt.subplots()

                ax.scatter(x1, fg_sim, color='blue', label='fg_sim')

                # 绘制第二个数据集的散点图
                ax.scatter(x2, bg_sim, color='red', label='bg_sim')

                # 添加标题和标签
                ax.set_title('Scatter Plot with Different Length Arrays')
                ax.set_xlabel('Index')
                ax.set_ylabel('Value')
                
                ax.legend()
                # plt.scatter(x, fg_sim, color='blue', label='fg_sim')
                # plt.scatter(x, bg_sim, color='red', label='bg_sim')
                
                # plt.title('Scatter Plot of sim between patch and proto')
                # plt.xlabel('Index')
                # plt.ylabel('sim Value')
                plt.savefig(f'/hy-tmp/explore_attention_vit/similarity/sim between patch{p} and proto.png')
                plt.close()
                
