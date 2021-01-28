import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torchvision import datasets
from torch.nn import functional as F
from torchvision import transforms
from pytorch_playground.utee import selector
import torchvision
import os
from torch.autograd import Variable
import sys

MEAN = torch.tensor([0.1307])
MEAN = MEAN[None, :, None, None]
VAR = torch.tensor([0.3081])
VAR = VAR[None, :, None, None]

model_raw, ds_fetcher, is_imagenet = selector.select('mnist')

def compute_accuracy(x,y):
    s=0
    count=x.size(0)*3


    # save_images(x[0], os.path.join('.', 'real_img.png'))



    for i in range(x.size(0)):

        # save_images(x[i], os.path.join('.', 'cluster.png'))

        d=x[i]
        nb,a1,a2,a3,b1,b2,b3=y[0,i], y[1,i],y[2,i],y[3,i],y[4,i],y[5,i],y[6,i]
        # print("position first digits", a1,b1,a2,b2,a3,b3)

        cont = torch.unsqueeze(torch.stack((d[0, a1:a1 + 21, b1:b1 + 21], d[1, a2:a2 + 21, b2:b2 + 21], d[2, a3:a3 + 21, b3:b3 + 21]), dim=0), dim=1)

        cont = F.interpolate(cont, size=(28,28))
        cont = ((cont + 1) / 2 - MEAN.cuda()) / VAR.cuda()
        # save_images(cont, os.path.join('.', 'cluster.png'))

        res=model_raw(cont)
        nb_fake=torch.argmax(res, dim=1)

        nb1 = nb//100
        nb2 = (nb - (nb1*100))//10
        nb3 = (nb - (nb1*100)) - nb2*10

        nb_real=torch.stack((nb1,nb2,nb3))

        res_fin = torch.eq(nb_real, nb_fake).sum()
        s += res_fin.item()

    print("Accuracy", s/count)
        # print(F.interpolate(torch.unsqueeze(d[1, a1:a1+21, b1:b1+21], dim=0), size=[28,28]).size())



def save_images(imgs, outfile, nrow=8):
    # if len(list(imgs.size()))==3:
    #     print("size = 3")
    #     imgs=torch.unsqueeze(imgs,1)
    #     torchvision.utils.save_image(imgs, outfile, nrow=nrow, normalize = True)
    # else:
    imgs = imgs / 2 + 0.5  # unnormalize
    torchvision.utils.save_image(imgs, outfile, nrow=nrow)