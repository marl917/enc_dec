import os
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms, datasets
import numpy as np
import random


def get_images(root, N):
    if False and os.path.exists(root + '.txt'):
        with open(os.path.exists(root + '.txt')) as f:
            files = f.readlines()
            random.shuffle(files)
            return files
    else:
        all_files = []
        for i, (dp, dn, fn) in enumerate(os.walk(os.path.expanduser(root))):
            print(dp, dn, fn)
            for j, f in enumerate(fn):
                if j >= 1000:
                    break     # don't get whole dataset, just get enough images per class
                if f.endswith(('.png', '.webp', 'jpg', '.JPEG')):
                    all_files.append(os.path.join(dp, f))
        random.shuffle(all_files)
        return all_files


def pt_to_np(imgs):
    '''normalizes pytorch image in [-1, 1] to [0, 255]'''
    return (imgs.permute(0, 2, 3, 1).mul_(0.5).add_(0.5).mul_(255)).clamp_(0, 255).numpy()


def get_transform(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_gt_samples(dataset, nimgs=50000):
    if dataset not in ['cifar','lsun', 'lsun_bedroom']:
        transform = get_transform(sizes[dataset])
        all_images = get_images(paths[dataset], nimgs)
        print(paths[dataset], dataset)
        images = []
        for file_path in tqdm(all_images[:nimgs]):
            images.append(transform(Image.open(file_path).convert('RGB')))
        return pt_to_np(torch.stack(images))
    elif dataset == 'lsun':
        data = datasets.LSUN(root='data/lsun/train',
                                   classes=['church_outdoor_train'],
                                   transform=get_transform(sizes[dataset]))
        images = []
        i=0
        for x, y in tqdm(data):
            i+=1
            images.append(x)
            if i>=50000:
                break
        return pt_to_np(torch.stack(images))
    elif dataset == 'lsun_bedroom':
        data = datasets.LSUN(root='data/lsun_bedroom/train',
                                   classes=['bedroom_train'],
                                   transform=get_transform(sizes[dataset]))
        images = []
        i=0
        for x, y in tqdm(data):
            i+=1
            images.append(x)
            if i>=50000:
                break
        return pt_to_np(torch.stack(images))

    else:
        data = datasets.CIFAR10(paths[dataset], transform=get_transform(sizes[dataset]))
        images = []

        for x, y in tqdm(data):
            images.append(x)
        return pt_to_np(torch.stack(images))


paths = {
    'imagenet': 'data/ImageNet',
    'places': 'data/Places365',
    'cifar': 'data/CIFAR',
    'lsun': 'data/lsun',
    'lsun_bedroom':'data/lsun_bedroom'
}

sizes = {'imagenet': 128, 'places': 128, 'cifar': 32, 'lsun': 64,'lsun_bedroom': 64}

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Save a batch of ground truth train set images for evaluation')
    parser.add_argument('--cifar', action='store_true')
    parser.add_argument('--imagenet', action='store_true')
    parser.add_argument('--places', action='store_true')
    parser.add_argument('--lsun', action='store_true')
    parser.add_argument('--lsun_bedroom', action='store_true')
    args = parser.parse_args()

    os.makedirs('output', exist_ok=True)

    if args.cifar:
        cifar_samples = get_gt_samples('cifar', nimgs=50000)  #was 50000, changed to 500
        np.savez('output/cifar_gt_imgs.npz', fake=cifar_samples, real=cifar_samples)
    if args.imagenet:
        imagenet_samples = get_gt_samples('imagenet', nimgs=50000)
        np.savez('output/imagenet_gt_imgs.npz', fake=imagenet_samples, real=imagenet_samples)
    if args.places:
        places_samples = get_gt_samples('places', nimgs=50000)
        np.savez('output/places_gt_imgs.npz', fake=places_samples, real=places_samples)
    if args.lsun:
        lsun_samples = get_gt_samples('lsun', nimgs=50000)
        np.savez('output/lsun_gt_imgs.npz', fake=lsun_samples, real=lsun_samples)
    if args.lsun_bedroom:
        lsun_samples = get_gt_samples('lsun_bedroom', nimgs=50000)
        np.savez('output/lsun_bedroom_gt_imgs.npz', fake=lsun_samples, real=lsun_samples)
