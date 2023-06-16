import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os.path
from pathlib import Path
import glob
import sys

import pdb

from modules.unet import UNet
from modules.midas.dpt_depth import DPTDepthModel
from data.transforms import get_transform


def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


def save_outputs(img_path, output_file_name, img_size, output_direct, trans_topil, trans_totensor, trans_rgb, device, model, type):
    with torch.no_grad():
        save_direct = os.path.join(args.output_path, output_direct)
        if not os.path.exists(save_direct):
            os.mkdir(save_direct)
        output_file_name = output_file_name.replace("_rgb", "")
        save_path = os.path.join(args.output_path, output_direct, f'{output_file_name}_{type}_omnidata.png')
        save_path_npy = os.path.join(args.output_path, output_direct, f'{output_file_name}_{type}_omnidata.npy')

        print(f'Reading input {img_path} ...')
        img = Image.open(img_path)

        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)

        # Uncomment if also rgb_images should be saved again
        #rgb_path = os.path.join(args.output_path, f'{output_file_name}_rgb.png')
        #trans_rgb(img).save(rgb_path)

        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3, 1)

        output = model(img_tensor).clamp(min=0, max=1)

        if type == 'depth':
            output = F.interpolate(output.unsqueeze(0), (img_size, img_size), mode='bicubic').squeeze(0)
            output = output.clamp(0, 1)
            output = 1 - output
            #             output = standardize_depth_map(output)
            plt.imsave(save_path, output.detach().cpu().squeeze(), cmap='viridis')
            np.save(save_path_npy, output.detach().cpu().squeeze())

        else:
            trans_topil(output[0]).save(save_path)
            np.save(save_path_npy, output.detach().cpu().squeeze())

        print(f'Writing output {save_path} ...')


def input_checker(img_path, output_path):
    if args.task != 'depth' and args.task != 'normal' and args.task != 'both':
        print("Task should be one of the following: normal, depth or both")
        sys.exit()
    if not img_path.is_dir() and not img_path.is_file():
        print("Input image path is not a valid path")
        sys.exit()
    if output_path.is_file():
        print("Output path is not a valid path")
        sys.exit()
    elif not output_path.is_dir():
        os.system(f"mkdir -p {args.output_path}")


def create_normal_maps(img_path, map_location, pretrained_weights_path, trans_topil, device, image_size):
    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)  # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                         transforms.CenterCrop(image_size),
                                         get_transform('rgb', image_size=None)])

    trans_rgb = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                    transforms.CenterCrop(image_size)])

    if img_path.is_file():
        save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0], image_size, 'single', trans_topil, trans_totensor, trans_rgb, device, model, 'normal')
    elif img_path.is_dir():
        for f in glob.glob(args.img_path + '/train/*rgb.png'):
            save_outputs(f, os.path.splitext(os.path.basename(f))[0], image_size, 'train', trans_topil, trans_totensor, trans_rgb, device, model, 'normal')
        for f in glob.glob(args.img_path + '/test/*rgb.png'):
            save_outputs(f, os.path.splitext(os.path.basename(f))[0], image_size, 'test', trans_topil, trans_totensor, trans_rgb, device, model, 'normal')
        for f in glob.glob(args.img_path + '/val/*rgb.png'):
            save_outputs(f, os.path.splitext(os.path.basename(f))[0], image_size, 'val', trans_topil, trans_totensor, trans_rgb, device, model, 'normal')


def create_depth_maps(img_path, map_location, pretrained_weights_path, trans_topil, device, image_size):
    # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
    model = DPTDepthModel(backbone='vitb_rn50_384')  # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                         transforms.CenterCrop(image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=0.5, std=0.5)])
    trans_rgb = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                    transforms.CenterCrop(image_size)])

    if img_path.is_file():
        save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0], image_size, 'single', trans_topil, trans_totensor, trans_rgb, device, model, 'depth')
    elif img_path.is_dir():
        for f in glob.glob(args.img_path + '/train/*rgb.png'):
            save_outputs(f, os.path.splitext(os.path.basename(f))[0], image_size, 'train', trans_topil, trans_totensor, trans_rgb, device, model, 'depth')
        for f in glob.glob(args.img_path + '/test/*rgb.png'):
            save_outputs(f, os.path.splitext(os.path.basename(f))[0], image_size, 'test', trans_topil, trans_totensor, trans_rgb, device, model, 'depth')
        for f in glob.glob(args.img_path + '/val/*rgb.png'):
            save_outputs(f, os.path.splitext(os.path.basename(f))[0], image_size, 'val', trans_topil, trans_totensor, trans_rgb, device, model, 'depth')



parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals or both')

parser.add_argument('--task', dest='task', help="normal or depth or both", required=True)

parser.add_argument('--output_size', dest='output_size', help="output image size e.g. 512x512", default=512)

parser.add_argument('--img_path', dest='img_path', help="path to rgb image", required=True)

parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored", required=True)

args = parser.parse_args()


def main():
    img_path = Path(args.img_path)
    output_path = Path(args.output_path)
    input_checker(img_path, output_path)

    root_dir = './pretrained_models/'
    trans_topil = transforms.ToPILImage()
    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image_size = int(args.output_size)

    # normal maps
    if args.task == 'normal' or args.task == 'both':
        pretrained_weights_path = root_dir + 'omnidata_dpt_normal_v2.ckpt'
        create_normal_maps(img_path, map_location, pretrained_weights_path, trans_topil, device, image_size)

    # depth maps
    if args.task == 'depth' or args.task == 'both':
        pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
        create_depth_maps(img_path, map_location, pretrained_weights_path, trans_topil, device, image_size)


if __name__ == '__main__':
    main()




