import numpy as np
import torch
from torch.nn import functional as F
from gan_training import utils
import torchvision
import os
import sys
from tqdm import tqdm

from gan_training.metrics import inception_score

class Evaluator(object):
    def __init__(self,
                 decoder,
                 encoder,
                 zdist,
                 zdist_lab,
                 train_loader,
                 batch_size=64,
                 inception_nsamples=10000,
                 device=None,
                 label_generator = None,
                 n_locallabels=0):

        self.decoder = decoder
        self.encoder = encoder
        self.train_loader = train_loader
        self.zdist = zdist
        self.zdist_lab = zdist_lab
        self.inception_nsamples = inception_nsamples
        self.batch_size = batch_size
        self.device = device
        self.label_generator = label_generator
        self.n_locallabels = n_locallabels

        self.cmap = self.labelcolormap(self.n_locallabels)
        self.cmap = torch.from_numpy(self.cmap[:self.n_locallabels])

    def uint82bin(self, n, count=8):
        """returns the binary of integer n, count refers to amount of bits"""
        return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

    def labelcolormap(self,n_labels):
        cmap = np.zeros((n_labels, 3), dtype=np.uint8)
        for i in range(n_labels):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = self.uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        return cmap

    def sample_z(self, batch_size):
        return self.zdist.sample((batch_size, )).to(self.device)

    def sample_z_lab(self, batch_size):
        return self.zdist_lab.sample((batch_size, )).to(self.device)

    def get_y(self, x, y):
        return self.clusterer.get_Globallabels(x, y).to(self.device)

    def get_local(self, x):
        return self.clusterer.convertLoc_Feat2One_Hot(x)

    def get_local_global(self,x,y):
        return self.clusterer.get_Global_LocalOneHotEnc_labels(x,y)

    def get_fake_real_samples(self, N, labelgen=False):
        ''' returns N fake images and N real images in pytorch form'''
        with torch.no_grad():
            self.decoder.eval()
            self.encoder.eval()
            fake_imgs = []
            real_imgs = []
            while len(fake_imgs) < N:
                for x_real, y_gt in self.train_loader:
                    x_real = x_real.cuda()
                    z = self.sample_z(x_real.size(0))
                    if not labelgen:
                        _,label_map = self.encoder(x_real)
                    else:
                        z_lab = self.sample_z_lab(x_real.size(0))
                        _,label_map = self.label_generator(z_lab)
                        # z_fake = torch.randn(x_real.size(0), 256, 1, 1, device='cuda')
                    # samples = self.decoder(z_fake)
                    samples = self.decoder(seg = label_map, input = z)

                    samples = [s.data.cpu() for s in samples]
                    fake_imgs.extend(samples)
                    real_batch = [img.data.cpu() for img in x_real]
                    real_imgs.extend(real_batch)
                    assert (len(real_imgs) == len(fake_imgs))
                    if len(fake_imgs) >= N:
                        fake_imgs = fake_imgs[:N]
                        real_imgs = real_imgs[:N]
                        return fake_imgs, real_imgs

    def compute_inception_score(self, labelgen=False):
        imgs, _ = self.get_fake_real_samples(self.inception_nsamples,  labelgen=labelgen)
        imgs = [img.numpy() for img in imgs]
        score, score_std = inception_score(imgs,
                                           device=self.device,
                                           resize=True,
                                           splits=1)

        return score, score_std

    def get_fake_samples(self,BS=100):
        self.decoder.eval()
        self.label_generator.eval()
        with torch.no_grad():
            z_lab = self.sample_z_lab(BS)
            z = self.sample_z(BS)
            _, label_map = self.label_generator(z_lab)
            x_fake = self.decoder(seg=label_map, input=z)
        return x_fake

    def get_dataset_from_path(self,path):
        datasets = ['imagenet', 'cifar', 'stacked_mnist', 'places','lsun']
        for name in datasets:
            if name in path:
                print('Inferred dataset:', name)
                return name

    def compute_fid_score(self, outdir, it):
        samples = []
        N = 50000
        BS = 100
        print("Starting to create samples for fid")
        for _ in tqdm(range(N // BS + 1)):
            x_fake =self.get_fake_samples(BS).detach().cpu()
            x_fake = [x.detach().cpu() for x in x_fake]
            samples.extend(x_fake)
        samples = torch.stack(samples[:N], dim=0)
        samples = (samples.permute(0, 2, 3, 1).mul_(0.5).add_(0.5).mul_(255)).clamp_(0, 255).numpy()

        samples_path = os.path.join(outdir,'samples.npz')
        dataset_name = self.get_dataset_from_path(outdir)
        print("Saving samples in : ", samples_path)
        np.savez(samples_path, fake=samples, real=dataset_name)

        arguments = f'--samples {samples_path} --it {it} --results_dir {outdir}'
        print("Computing fid score")
        print(self.device)
        os.system(f'CUDA_VISIBLE_DEVICES={0} python gan_training/metrics/fid.py {arguments}')


    def create_samples_labelGen(self,z, z_lab, out_dir=None):
        self.decoder.eval()
        self.label_generator.eval()
        with torch.no_grad():
            _,label_map = self.label_generator(z_lab)

            # print("generator labels : ", torch.argmax(label_map, dim=1).long())
            
            # save_lab = torch.unsqueeze(torch.argmax(label_map, dim=1), dim=1).float()
            # torchvision.utils.save_image(save_lab, os.path.join(out_dir, 'label_maps.png'))

            x_fake1 = self.decoder(seg = label_map, input=z)

            count = torch.bincount(torch.argmax(label_map, dim=1).view(-1), minlength=self.n_locallabels)
            print("occurence on label maps for gen label maps for each classes", count)

            lab_up = F.interpolate(label_map.float(), size=x_fake1.size()[2:], mode='bilinear', align_corners=True)
            color_lab_map = self.create_colormap(lab_up)

            # z_Bis = self.sample_z(z.size(0))
            # x_fake2 = self.decoder(seg = label_map, input = z_Bis)
            # print('range in labelGen sample', torch.min(x_fake), torch.max(x_fake), x_fake.size())
            # z_fake = torch.randn(50, 256, 1, 1, device = 'cuda')
            # x_fake = self.decoder(z_fake)

        # return torch.cat((x_fake1, x_fake2), dim=0)
        return x_fake1, color_lab_map
    def create_samples(self, x_real,z):
        self.decoder.eval()
        self.encoder.eval()
        # batch_size = z.size(0)
        # Parse y
        # if y is None:
        #     raise NotImplementedError()
        # elif isinstance(y, int):
        #     y = torch.full((batch_size, ),
        #                    y,
        #                    device=self.device,
        #                    dtype=torch.int64)
        # Sample x
        with torch.no_grad():
            _,label_map = self.encoder(x_real)
            print("encoder labels : " ,torch.argmax(label_map, dim = 1)[0])

            x_fake = self.decoder(seg = label_map, input = z)
            print("min max of fake img : ", torch.min(x_fake), torch.max(x_fake))
            lab_up = F.interpolate(label_map.float(), size=x_real.size()[2:], mode='bilinear', align_corners=True)
            color_lab_map = self.create_colormap(lab_up)

            count = torch.bincount(torch.argmax(label_map, dim = 1).view(-1), minlength=self.n_locallabels)
            print("occurence on label maps from dataset img for each classes", count)
            # print("color_label map", color_lab_map)
            # z_real = self.encoder(x_real)
            # z_real = z_real.view(x_real.size(0), -1)
            # mu, log_sigma = z_real[:, :latent_size], z_real[:, latent_size:]
            # sigma = torch.exp(log_sigma)
            #
            # epsilon = torch.randn(x_real.size(0), latent_size, device='cuda')
            # output_z = mu + epsilon * sigma
            # output_z = output_z[:,:,None,None]
            # # print("output_z size", output_z.size())
            # x_fake = self.decoder(output_z)
        return x_fake, color_lab_map

    def create_colormap(self, oneHot):
        gray_img = torch.argmax(oneHot, dim = 1, keepdim=True)
        # print("gray img size :", gray_img.size())
        size = gray_img.size()
        color_image = torch.ByteTensor(size[0],3, size[2], size[3]).fill_(0)
        # print("size of color img", color_image.size())
        for i in range(size[0]):
            one_img = gray_img[i].cpu()
            for label in range(0, len(self.cmap)):
                mask = (label == one_img[0]).cpu()
                color_image[i][0][mask] = self.cmap[label][0]
                color_image[i][1][mask] = self.cmap[label][1]
                color_image[i][2][mask] = self.cmap[label][2]
            #     if i==0:
            #         print("label map cmap", self.cmap[label][0], self.cmap[label][1], self.cmap[label][2])
            # if i==0:
            #
            #     print("color_img[0]", color_image[0])
        return color_image


