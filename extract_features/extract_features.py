import os
import sys
sys.path.append("/hy-tmp/explore_attention_vit/dinov2")
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob
import os.path as osp
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torchvision import transforms
from torchvision import transforms
from argparse import ArgumentParser

from dinov2.models.vision_transformer import vit_small, vit_base, vit_large

def preprocess(image, mask=None, backbone_type='dinov2', target_size=(602, 602), patch_size=14):
    '''
    Preprocess an image and its mask to fed the image to the backbone and mask the extracted patches.

    Args:
        image (PIL.Image): Input image
        mask (PIL.Image): Input mask
        backbone_type (str): Backbone type
        target_size (tuple): Target size of the image
        patch_size (int): Patch size of the backbone
    '''

    if 'clip' in backbone_type:
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    # Transform images to tensors and normalize
    transform = transforms.Compose([
        transforms.Resize(target_size),  # Resize the images to a size larger than the window size
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  # Normalize the images
    ])
    
    if mask is not None:
        m_w, m_h = target_size
        mask = transforms.Resize((m_w//patch_size, m_h//patch_size), interpolation=Image.NEAREST)(mask)

    image = transform(image).unsqueeze(0)
    # print(image.shape)

    return image, mask


def extract_backbone_features(images, model, backbone_type, scale_factor=1):
    '''
    Extract features from a pre-trained backbone for any of the supported backbones.

    Args:
        images (torch.Tensor): Input tensor with shape (B, C, H, W)
        model (torch.nn.Module): Backbone model
        backbone_type (str): Backbone type
        scale_factor (int): Scale factor for the input images. Set to 1 for no scaling.
    '''
    # images = F.interpolate(images, scale_factor=scale_factor, mode='bicubic')   ## zhouzhi

    # if 'dinov2' in backbone_type:
    with torch.no_grad():
        feats = model.forward_features(images)['x_prenorm'][:, 1:]
    
    return feats

def main():
    img_file = '/hy-tmp/explore_attention_vit/dior_train/00005.jpg'
    image = Image.open(img_file)

    model = torch.hub.load('/hy-tmp/explore_attention_vit/dinov2','dinov2_vitl14',source='local')
    image, _ = preprocess(image, backbone_type='dinov2', target_size=(602, 602), patch_size=14)
    # model = vit_large(
    #         patch_size=14,
    #         img_size=526,
    #         init_values=1.0,
    #         num_register_tokens=4,
    #         block_chunks=0
    #         )
    
    device = "cuda"

    # model.load_state_dict(torch.load('/hy-tmp/dinov2_vitl14_pretrain.pth'))
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        feats = model.forward_features(image.to(device))['x_prenorm'][:, 1:]
        mid_feats = model.get_intermediate_layers(image.to(device), [3, 5, 7, 9], reshape=True)

    print(feats.shape)
    for i in range(len(mid_feats)):
        print(mid_feats[i].shape)
    
if __name__ == '__main__':
    main()