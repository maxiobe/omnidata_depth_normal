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


class NormalDepthGenerator():
    def __init__(self, args):
        self.task = args.task
        self.trans_topil = transforms.ToPILImage()
        self.img_path = args.img_path
        self.img_path_posix = Path(args.img_path)
        self.output_path = Path(args.output_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.image_size = int(args.output_size)
        self.root_dir = './pretrained_models/'
        self.map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
        self.normal_pretrained_weights_path = self.root_dir + 'omnidata_dpt_normal_v2.ckpt'
        self.depth_pretrained_weights_path = self.root_dir + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
        self.trans_rgb = transforms.Compose([transforms.Resize(self.image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.CenterCrop(self.image_size)])

    def input_checker(self):
        if self.task != 'depth' and self.task != 'normal' and self.task != 'both':
            print("Task should be one of the following: normal, depth or both")
            sys.exit()
        if not self.img_path_posix.is_dir() and not self.img_path_posix.is_file():
            print("Input image path is not a valid path")
            sys.exit()
        if self.output_path.is_file():
            print("Output path is not a valid path")
            sys.exit()
        elif not self.output_path.is_dir():
            os.system(f"mkdir -p {self.output_path}")

    def standardize_depth_map(self, img, mask_valid=None, trunc_value=0.1):
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

    def save_outputs(self, image_path, output_file_name, output_direct, trans_totensor, model, type):
        with torch.no_grad():
            save_direct = os.path.join(self.output_path, output_direct)
            if not os.path.exists(save_direct):
                os.mkdir(save_direct)
            output_file_name = output_file_name.replace("_rgb", "")
            save_path = os.path.join(self.output_path, output_direct, f'{output_file_name}_{type}_omnidata.png')
            save_path_npy = os.path.join(self.output_path, output_direct, f'{output_file_name}_{type}_omnidata.npy')

            print(f'Reading input {image_path} ...')
            img = Image.open(image_path)

            img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(self.device)

            # Uncomment if also rgb_images should be saved again
            # rgb_path = os.path.join(self.output_path, f'{output_file_name}_rgb.png')
            # self.trans_rgb(img).save(rgb_path)

            if img_tensor.shape[1] == 1:
                img_tensor = img_tensor.repeat_interleave(3, 1)

            output = model(img_tensor).clamp(min=0, max=1)

            if type == 'depth':
                output = F.interpolate(output.unsqueeze(0), (self.image_size, self.image_size), mode='bicubic').squeeze(0)
                output = output.clamp(0, 1)
                output = 1 - output
                #             output = standardize_depth_map(output)
                plt.imsave(save_path, output.detach().cpu().squeeze(), cmap='viridis')
                np.save(save_path_npy, output.detach().cpu().squeeze())

            else:
                self.trans_topil(output[0]).save(save_path)
                np.save(save_path_npy, output.detach().cpu().squeeze())

            print(f'Writing output {save_path} ...')

    def create_normal_maps(self):
        model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)  # DPT Hybrid
        checkpoint = torch.load(self.normal_pretrained_weights_path, map_location=self.map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(self.device)
        trans_totensor = transforms.Compose([transforms.Resize(self.image_size, interpolation=PIL.Image.BILINEAR),
                                             transforms.CenterCrop(self.image_size),
                                             get_transform('rgb', image_size=None)])

        if self.img_path_posix.is_file():
            self.save_outputs(os.path.splitext(os.path.basename(self.img_path))[0], 'single', trans_totensor, model, 'normal')
        elif self.img_path_posix.is_dir():
            for folder in ['train', 'test', 'val' ]:
                for f in glob.glob(self.img_path + '/' + folder + '/*rgb.png'):
                    self.save_outputs(f, os.path.splitext(os.path.basename(f))[0], folder, trans_totensor, model, 'normal')

    def create_depth_maps(self):
        # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
        model = DPTDepthModel(backbone='vitb_rn50_384')  # DPT Hybrid
        checkpoint = torch.load(self.depth_pretrained_weights_path, map_location=self.map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(self.device)
        trans_totensor = transforms.Compose([transforms.Resize(self.image_size, interpolation=PIL.Image.BILINEAR),
                                             transforms.CenterCrop(self.image_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=0.5, std=0.5)])

        if self.img_path_posix.is_file():
            self.save_outputs(os.path.splitext(os.path.basename(self.img_path))[0], 'single', trans_totensor, model, 'depth')
        elif self.img_path_posix.is_dir():
            for folder in ['train', 'test', 'val']:
                for f in glob.glob(self.img_path + '/' + folder + '/*rgb.png'):
                    self.save_outputs(f, os.path.splitext(os.path.basename(f))[0], folder, trans_totensor, model, 'depth')

def main():
    parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals or both')
    parser.add_argument('--task', dest='task', help="normal or depth or both", required=True)
    parser.add_argument('--output_size', dest='output_size', help="output image size e.g. 512x512", default=512)
    parser.add_argument('--img_path', dest='img_path', help="path to rgb image", required=True)
    parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored", required=True)
    args = parser.parse_args()

    NormalDepth = NormalDepthGenerator(args)
    NormalDepth.input_checker()

    # normal maps
    if args.task == 'normal' or args.task == 'both':
        NormalDepth.create_normal_maps()
    # depth maps
    if args.task == 'depth' or args.task == 'both':
        NormalDepth.create_depth_maps()


if __name__ == '__main__':
    main()




