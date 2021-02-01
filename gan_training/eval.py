import numpy as np
import torch
from torch.nn import functional as F
from gan_training import utils

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

    def create_samples_labelGen(self,z, z_lab):
        self.decoder.eval()
        self.label_generator.eval()
        with torch.no_grad():
            _,label_map = self.label_generator(z_lab)
            print("generator labels : ", torch.argmax(label_map, dim=1)[:3])
            x_fake1 = self.decoder(seg = label_map, input=z)

            z_Bis = self.sample_z(z.size(0))
            x_fake2 = self.decoder(seg = label_map, input = z_Bis)
            # print('range in labelGen sample', torch.min(x_fake), torch.max(x_fake), x_fake.size())
            # z_fake = torch.randn(50, 256, 1, 1, device = 'cuda')
            # x_fake = self.decoder(z_fake)

        return torch.cat((x_fake1, x_fake2), dim=0)

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
            print("encoder labels : " ,torch.argmax(label_map, dim = 1)[:3])
            x_fake = self.decoder(seg = label_map, input = z)

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
        return x_fake


