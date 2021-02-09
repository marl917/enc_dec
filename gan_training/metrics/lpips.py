from __future__ import absolute_import
import numpy as np
from os import path
import os
import sys
sys.path.append('gan_training')
import utils
from torchvision import transforms, datasets
import torch
from tqdm import tqdm
import lpips

N=10 #nb of fake images used to find the closest image from the dataset

def get_transform(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def compute_lpips_from_npz(path_fake_img, results_dir, device):
    print(path_fake_img)
    with np.load(path_fake_img) as data:
        fake_imgs = data['fake']

        name_dat = None
        for name in ['imagenet', 'cifar', 'places', 'lsun']:
            if name in path_fake_img:
                name_dat = name
                break
        print('Inferred name', name_dat)

    fake_imgs_dir = os.path.join(results_dir, "fake_imgs")
    if not path.exists(fake_imgs_dir):
        os.makedirs(fake_imgs_dir)


    list_tensor_fake_img = []
    result_dict = {}
    for i in range(N):
        img = np.transpose(fake_imgs[i], (2, 0, 1))
        img = torch.from_numpy(img)
        img *= (1 / 255)
        t = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        img = t(img)
        list_tensor_fake_img.append(img)
        utils.save_images(img, path.join(fake_imgs_dir, "%d.png"%(i)))
        result_dict[i] = float('inf')

    tensor_fake_img = torch.stack(list_tensor_fake_img, dim = 0)
    tensor_fake_img = torch.unsqueeze(tensor_fake_img, dim = 1)
    print(tensor_fake_img.size(), tensor_fake_img[0].size())


    real_data = datasets.LSUN(root='data/lsun/train',
                         classes=['church_outdoor_train'],
                         transform=get_transform(64))   #hardcoded : size of dataset

    loss_fn = lpips.LPIPS(net='alex', version=0.1)
    loss_fn.cuda()

    closest_img = path.join(results_dir,"closest_real")
    if not path.exists(closest_img):
        os.makedirs(closest_img)

    for x, y in tqdm(real_data):
        x = torch.unsqueeze(x, dim = 0)
        for i in range(N):
            dist01 = loss_fn.forward(x.cuda(), tensor_fake_img[i].cuda())
            if dist01<result_dict[i]:
                result_dict[i]=dist01
                utils.save_images(x[0], path.join(closest_img, "%d.png"%(i)))
    return result_dict

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser('compute LPIPS similarity')
    parser.add_argument('--samples', help='path to samples')
    parser.add_argument('--results_dir', help='path to results_dir')
    parser.add_argument('--device', help='gpu device')
    args = parser.parse_args()


    results_dir = args.results_dir
    result_dict = compute_lpips_from_npz(args.samples, results_dir, args.device)
    print(result_dict)