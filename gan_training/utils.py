import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import numpy as np
import os
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=1)


def gumble_softmax( logits, n_locallabels, is_cuda=True, temperature = 1.0):
    FloatTensor = torch.cuda.FloatTensor if is_cuda \
        else torch.FloatTensor
    y = gumbel_softmax_sample(logits, temperature=temperature)
    x = torch.argmax(y, dim=1)
    x = torch.unsqueeze(x, dim=1)
    bs, _, h, w = x.size()
    input_label = FloatTensor(bs, n_locallabels, h, w).zero_()
    y_hard = input_label.scatter_(1, x.long().cuda(), 1.0)
    return (y_hard - y).detach() + y

def save_images(imgs, outfile, nrow=8):
    # if len(list(imgs.size()))==3:
    #     print("size = 3")
    #     imgs=torch.unsqueeze(imgs,1)
    #     torchvision.utils.save_image(imgs, outfile, nrow=nrow, normalize = True)
    # else:
    imgs = imgs / 2 + 0.5  # unnormalize
    torchvision.utils.save_image(imgs, outfile, nrow=nrow)


def save_imageAlt(image_numpy, image_path, create_dir=False):
    # print(image_numpy.shape)
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)

    # save to png
    image_pil.save(image_path.replace('.jpg', '.png'))

# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8, tile=False):
    if label_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(label_tensor.size(0)):
            one_image = label_tensor[b]
            one_image_np = tensor2label(one_image, n_label, imtype)
            # print("in tensor 2 label: ", one_image_np.shape, one_image_np.reshape(1, *one_image_np.shape).shape)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            # images_np=torch.Tensor(images_np).permute(0,3,1,2)
            # images_np = images_np[0]
            return images_np

    if label_tensor.dim() == 1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70),
                         (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153),
                         (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
                         (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32),
                         (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b

        if N == 182:  # COCO
            important_colors = {
                'sea': (54, 62, 167),
                'sky-other': (95, 219, 255),
                'tree': (140, 104, 47),
                'clouds': (170, 170, 170),
                'grass': (29, 195, 49)
            }
            for i in range(N):
                name = util.coco.id2label(i)
                if name in important_colors:
                    color = important_colors[name]
                    cmap[i] = np.array(list(color))

    return cmap

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)

def tile_images(imgs, picturesPerRow=4):
    """ Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled


def get_nsamples(data_loader, N):
    x = []
    y = []
    n = 0
    for x_next, y_next in data_loader:
        x.append(x_next)
        y.append(y_next)
        n += x_next.size(0)
        if n > N:
            break
    x = torch.cat(x, dim=0)[:N]
    if isinstance(y_next, list):
        y1=torch.cat((y[0][0], y[1][0]), dim=0)[:N]
        y2 = torch.cat((y[0][1], y[1][1]), dim=0)[:N]
        y3 = torch.cat((y[0][2], y[1][2]), dim=0)[:N]
        y4 = torch.cat((y[0][3], y[1][3]), dim=0)[:N]
        y5 = torch.cat((y[0][4], y[1][4]), dim=0)[:N]
        y6 = torch.cat((y[0][5], y[1][5]), dim=0)[:N]
        y7 = torch.cat((y[0][6], y[1][6]), dim=0)[:N]
        y=torch.stack([y1,y2,y3,y4,y5,y6,y7], dim=0)
    else:
        y = torch.cat(y, dim=0)[:N]
    return x, y


def update_average(model_tgt, model_src, beta):
    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)


def get_most_recent(d, ext):
    if not os.path.exists(d):
        print('Directory', d, 'does not exist')
        return -1 
    its = []
    for f in os.listdir(d):
        try:
            it = int(f.split(ext + "_")[1].split('.pt')[0])
            its.append(it)
        except Exception as e:
            pass
    if len(its) == 0:
        print('Found no files with extension \"%s\" under %s' % (ext, d))
        return -1
    return max(its)
