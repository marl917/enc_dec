import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

import os
import torch.utils.data as data
from torchvision.datasets.folder import default_loader
from PIL import Image
import random


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_dataset(name,
                data_dir,
                size=64,
                lsun_categories=None,
                deterministic=False,
                transform=None):
                
    transform = transforms.Compose([
        t for t in [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            (not deterministic) and transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            (not deterministic) and
            transforms.Lambda(lambda x: x + 1. / 128 * torch.rand(x.size())),
        ] if t is not False
    ]) if transform == None else transform

    dataset_test=None
    if name == 'image':
        print('Using image labels')
        dataset = datasets.ImageFolder(data_dir, transform)
        nlabels = len(dataset.classes)

    elif name == 'imagenet100':
        parent_dir = os.path.join('data/imagenet100', 'symlinkToImgNet100')
        if not os.path.isdir(parent_dir):
            os.makedirs(parent_dir)
            mini_keys = ['n02110341', 'n01930112', 'n04509417', 'n04067472', 'n04515003', 'n02120079', 'n03924679',
                          'n02687172', 'n03075370', 'n07747607', 'n09246464', 'n02457408', 'n04418357', 'n03535780',
                          'n04435653', 'n03207743', 'n04251144', 'n03062245', 'n02174001', 'n07613480', 'n03998194',
                          'n02074367', 'n04146614', 'n04243546', 'n03854065', 'n03838899', 'n02871525', 'n03544143',
                          'n02108089', 'n13133613', 'n03676483', 'n03337140', 'n03272010', 'n01770081', 'n09256479',
                          'n02091244', 'n02116738', 'n04275548', 'n03773504', 'n02606052', 'n03146219', 'n04149813',
                          'n07697537', 'n02823428', 'n02089867', 'n03017168', 'n01704323', 'n01532829', 'n03047690',
                          'n03775546', 'n01843383', 'n02971356', 'n13054560', 'n02108551', 'n02101006', 'n03417042',
                          'n04612504', 'n01558993', 'n04522168', 'n02795169', 'n06794110', 'n01855672', 'n04258138',
                          'n02110063', 'n07584110', 'n02091831', 'n03584254', 'n03888605', 'n02113712', 'n03980874',
                          'n02219486', 'n02138441', 'n02165456', 'n02108915', 'n03770439', 'n01981276', 'n03220513',
                          'n02099601', 'n02747177', 'n01749939', 'n03476684', 'n02105505', 'n02950826', 'n04389033',
                          'n03347037', 'n02966193', 'n03127925', 'n03400231', 'n04296562', 'n03527444', 'n04443257',
                          'n02443484', 'n02114548', 'n04604644', 'n01910747', 'n04596742', 'n02111277', 'n03908618',
                          'n02129165', 'n02981792']
            print('Using image labels')
            for i in range(len(mini_keys)):
                os.symlink(os.path.join(data_dir, mini_keys[i]), os.path.join(parent_dir, mini_keys[i]))
            print("Created symbolic link containing the 100 classes of mini-ImageNet to dir : ", parent_dir)
        dataset = datasets.ImageFolder(parent_dir, transform)
        nlabels = len(dataset.classes)

    elif name == 'webp':
        print('Using no labels from webp')
        dataset = CachedImageFolder(data_dir, transform)
        nlabels = len(dataset.classes)
    elif name == 'npy':
        # Only support normalization for now
        dataset = datasets.DatasetFolder(data_dir, npy_loader, ['npy'])
        nlabels = len(dataset.classes)
    elif name == 'lsun':
        dataset = datasets.LSUN(root='data/lsun/train',
                                   classes=['church_outdoor_train'],
                                   transform=transform)
        nlabels = 1

    elif name == 'cifar10':
        dataset = datasets.CIFAR10(root=data_dir,
                                   train=True,
                                   download=True,
                                   transform=transform)
        dataset_test = datasets.CIFAR10(root=data_dir,
                                   train=False,
                                   download=True,
                                   transform=transform)

        nlabels = 10
    elif name == 'stacked_mnist':
        dataset = StackedMNIST(data_dir,
                               transform=transforms.Compose([
                                   transforms.Resize(size),
                                   transforms.CenterCrop(size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, ), (0.5, ))
                               ]))
        nlabels = 1000

    elif name == 'colored_3_mnist':
        dataset = Col3MNIST(data_dir,
                               transform=transforms.Compose([
                                   transforms.Resize(size//3),
                                   transforms.CenterCrop(size//3),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, ), (0.5, ))
                               ]))
        nlabels = 1000
    # elif name == 'lsun':
    #     if lsun_categories is None:
    #         lsun_categories = 'train'
    #     dataset = datasets.LSUN(data_dir, lsun_categories, transform)
    #     nlabels = len(dataset.classes)
    elif name == 'lsun_class':
        dataset = datasets.LSUNClass(data_dir,
                                     transform,
                                     target_transform=(lambda t: 0))
        nlabels = 1
    else:
        raise NotImplemented
    return dataset, nlabels, dataset_test

class CachedImageFolder(data.Dataset):
    """
    A version of torchvision.dataset.ImageFolder that takes advantage
    of cached filename lists.
    photo/park/004234.jpg
    photo/park/004236.jpg
    photo/park/004237.jpg
    """

    def __init__(self, root, transform=None, loader=default_loader):
        classes, class_to_idx = find_classes(root)
        self.imgs = make_class_dataset(root, class_to_idx)
        if len(self.imgs) == 0:
            raise RuntimeError("Found 0 images within: %s" % root)
        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, classidx = self.imgs[index]
        source = self.loader(path)
        if self.transform is not None:
            source = self.transform(source)
        return source, classidx

    def __len__(self):
        return len(self.imgs)


class Col3MNIST(data.Dataset):
    def __init__(self, data_dir, transform, batch_size=100000):
        super().__init__()
        self.channel1 = datasets.MNIST(data_dir,
                                       transform=transform,
                                       train=True,
                                       download=True)
        self.channel2 = datasets.MNIST(data_dir,
                                       transform=transform,
                                       train=True,
                                       download=True)
        self.channel3 = datasets.MNIST(data_dir,
                                       transform=transform,
                                       train=True,
                                       download=True)
        self.indices = {
            k: (random.randint(0,
                               len(self.channel1) - 1),
                random.randint(0,
                               len(self.channel1) - 1),
                random.randint(0,
                               len(self.channel1) - 1))
            for k in range(batch_size)
        }
        # print("Col3MNIST", self.channel1[1][0])
        # print(len(self.indices))

    def __getitem__(self, index):
        index1, index2, index3 = self.indices[index]
        x1, y1 = self.channel1[index1]
        x2, y2 = self.channel2[index2]
        x3, y3 = self.channel3[index3]
        x=-torch.ones(3,64,64)
        xVal = []
        yVal = []
        for k, e in enumerate([x1,x2,x3]):
            if k==0:
                a,b=random.randint(0,x.size(1) - x1.size(1)), random.randint(0, x.size(2)-x1.size(2))
                acoord=[i for i in range(max(a-10,0), a+x1.size(1)-10)]
                bcoord = [i for i in range(max(b-10,0), b+ x1.size(2)-10)]
            elif k==1:
                a,b = random.choice([i for i in range(x.size(1) - x1.size(1)) if i not in acoord]), random.choice([i for i in range(x.size(2)-x1.size(2)) if i not in bcoord])
                acoord += [i for i in range(max(a-10,0), a+x1.size(1)-10)]
                bcoord += [i for i in range(max(b - 10, 0), b + x1.size(2) - 10)]
            else:
                a, b = random.choice([i for i in range(x.size(1) - x1.size(1)) if i not in acoord]), random.choice(
                    [i for i in range(x.size(2) - x1.size(2)) if i not in bcoord])
            x[k,a:a+x1.size(1),b:b+x1.size(2)]=e
            xVal.append(a)
            yVal.append(b)
        return x, [y1 * 100 + y2 * 10 + y3] + xVal + yVal

    def __len__(self):
        return len(self.indices)

class StackedMNIST(data.Dataset):
    def __init__(self, data_dir, transform, batch_size=100000):
        super().__init__()
        self.channel1 = datasets.MNIST(data_dir,
                                       transform=transform,
                                       train=True,
                                       download=True)
        self.channel2 = datasets.MNIST(data_dir,
                                       transform=transform,
                                       train=True,
                                       download=True)
        self.channel3 = datasets.MNIST(data_dir,
                                       transform=transform,
                                       train=True,
                                       download=True)
        self.indices = {
            k: (random.randint(0,
                               len(self.channel1) - 1),
                random.randint(0,
                               len(self.channel1) - 1),
                random.randint(0,
                               len(self.channel1) - 1))
            for k in range(batch_size)
        }


    def __getitem__(self, index):
        index1, index2, index3 = self.indices[index]
        x1, y1 = self.channel1[index1]
        x2, y2 = self.channel2[index2]
        x3, y3 = self.channel3[index3]
        return torch.cat([x1, x2, x3], dim=0), y1 * 100 + y2 * 10 + y3

    def __len__(self):
        return len(self.indices)
        

def is_npy_file(path):
    return path.endswith('.npy') or path.endswith('.NPY')


def walk_image_files(rootdir):
    print(rootdir)
    if os.path.isfile('%s.txt' % rootdir):
        print('Loading file list from %s.txt instead of scanning dir' %
              rootdir)
        basedir = os.path.dirname(rootdir)
        with open('%s.txt' % rootdir) as f:
            result = sorted([
                os.path.join(basedir, line.strip()) for line in f.readlines()
            ])
            import random
            random.Random(1).shuffle(result)
            return result
    result = []

    IMG_EXTENSIONS = [
        '.jpg',
        '.JPG',
        '.jpeg',
        '.JPEG',
        '.png',
        '.PNG',
        '.ppm',
        '.PPM',
        '.bmp',
        '.BMP',
    ]

    for dirname, _, fnames in sorted(os.walk(rootdir)):
        for fname in sorted(fnames):
            if any(fname.endswith(extension)
                   for extension in IMG_EXTENSIONS) or is_npy_file(fname):
                result.append(os.path.join(dirname, fname))
    return result


def find_classes(dir):
    classes = [
        d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))
    ]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_class_dataset(source_root, class_to_idx):
    """
    Returns (source, classnum, feature)
    """
    imagepairs = []
    source_root = os.path.expanduser(source_root)
    for path in walk_image_files(source_root):
        classname = os.path.basename(os.path.dirname(path))
        imagepairs.append((path, 0))
    return imagepairs


def npy_loader(path):
    img = np.load(path)

    if img.dtype == np.uint8:
        img = img.astype(np.float32)
        img = img / 127.5 - 1.
    elif img.dtype == np.float32:
        img = img * 2 - 1.
    else:
        raise NotImplementedError

    img = torch.Tensor(img)
    if len(img.size()) == 4:
        img.squeeze_(0)

    return img
