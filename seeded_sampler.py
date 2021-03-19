''' Samples from a (class-conditional) GAN, so that the samples can be reproduced '''

import os
import pickle
import random
import copy

import torch
from torch import nn

from gan_training.checkpoints import CheckpointIO
from gan_training.config import (load_config, build_models)
from seeing.yz_dataset import YZDataset
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training import utils


def get_most_recent(models):
    model_numbers = [
        int(model.split("model.pt")[0]) if model != "model.pt" else 0
        for model in models
    ]
    return str(max(model_numbers)) + "model.pt"


class SeededSampler():
    def __init__(
            self,
            config_name,        # name of experiment's config file
            model_path="",      # path to the model. empty string infers the most recent checkpoint
            pretrained={},      # urls to the pretrained models
            rootdir='./',
            device='cuda:0',
            useLabelGen=False,
            iteration_label_gen=None):
        self.config = load_config(os.path.join(rootdir, config_name), 'configs/default.yaml')

        self.model_path = model_path
        self.rootdir = rootdir
        self.device = device
        self.pretrained = pretrained

        self.useLabelGen = useLabelGen

        self.decoder, self.encoderOrLabGen = self.get_decoderencoder(useLabelGen=useLabelGen, iteration_label_gen=iteration_label_gen)
        self.decoder.eval()
        self.encoderOrLabGen.eval()
        #self.yz_dist = self.get_yz_dist()

        train_dataset, _, _ = get_dataset(
            name=self.config['data']['type'],
            data_dir=self.config['data']['train_dir'],
            size=self.config['data']['img_size'],
            deterministic=self.config['data']['deterministic'])

        self.batch_size = 100  #hardcoded
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.config['training']['nworkers'],
            shuffle=True,
            pin_memory=True,
            sampler=None,
            drop_last=True)

        if 'sean' in self.config['decoder']['name']:
            self.zdist = get_zdist(self.config['z_dist']['type'],
                              (self.config['label_generator']['n_locallabels'], self.config['decoder']['zdim']))
        else:
            self.zdist = get_zdist(self.config['z_dist']['type'], self.config['decoder']['zdim'])

        self.zdist_lab = get_zdist(self.config['z_dist']['type'], self.config['label_generator']['zdim'])
    def sample(self, nimgs):
        '''
        samples an image using the generator, with z drawn from isotropic gaussian, and y drawn from self.yz_dist.
        For baseline methods, y doesn't matter because y is ignored in the input
        yz_dist is the empirical label distribution for the clustered gans.

        returns the image, and the integer seed used to generate it. generated sample is in [-1, 1]
        '''
        self.generator.eval()
        with torch.no_grad():
            seeds = [random.randint(0, 1e8) for _ in range(nimgs)]
            z, y = self.yz_dist(seeds)
            return self.generator(z, y), seeds

    def sample_fromDataset(self, nimgs):
        self.decoder.eval()
        self.encoderOrLabGen.eval()
        with torch.no_grad():
            x_real,y=next(iter(self.train_loader))
            z = self.zdist.sample((self.batch_size,))
            x_real= x_real.to('cuda')
            _, label_map = self.encoderOrLabGen(x_real)
            return self.decoder(seg= label_map, input = z)[:nimgs,:,:,:], None

    def sample_fromLabelGenerator(self,nimgs): #to use if useLabelGen=True
        self.decoder.eval()
        self.encoderOrLabGen.eval()
        with torch.no_grad():
            z_lab = self.zdist_lab.sample((self.batch_size,))
            z = self.zdist.sample((self.batch_size,))
            _,input_semantics = self.encoderOrLabGen(z_lab)

        return self.decoder(seg= input_semantics, input = z)[:nimgs, :, :, :], None

    def conditional_sample(self, yi, seed=None):
        ''' returns a generated sample, which is in [-1, 1], seed is an int'''
        self.generator.eval()
        with torch.no_grad():
            if seed is None:
                seed = [random.randint(0, 1e8)]
            else:
                seed = [seed]
            z, _ = self.yz_dist(seed)
            y = torch.LongTensor([yi]).to(self.device)
            return self.generator(z, y)

    def sample_with_seed(self, seeds):
        ''' returns a generated sample, which is in [-1, 1] '''
        self.generator.eval()
        z, y = self.yz_dist(seeds)
        return self.generator(z, y)

    def get_zy(self, seeds):
        '''returns the batch of z, y corresponding to the seeds'''
        return self.yz_dist(seeds)

    def sample_with_zy(self, z, y):
        ''' returns a generated sample given z and y, which is in [-1, 1].'''
        self.generator.eval()
        return self.generator(z, y)

    def get_decoderencoder(self, useLabelGen=False, iteration_label_gen = None):
        ''' loads a decoder/encoder according to self.model_path '''

        exp_out_dir = os.path.join(self.rootdir, self.config['training']['out_dir'])
        # infer checkpoint if neeeded
        checkpoint_dir = os.path.join(exp_out_dir, 'chkpts') if self.model_path == "" or 'model' in self.pretrained else "./"
        model_name = get_most_recent(os.listdir(checkpoint_dir)) if self.model_path == "" else self.model_path

        checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)
        print("checkpoint dir : ", checkpoint_dir, model_name)
        self.checkpoint_io = checkpoint_io

        decoder, encoder, discriminator, label_generator, qhead_discriminator = build_models(self.config)
        if not useLabelGen:
            decoder = decoder.to(self.device)
            decoder = nn.DataParallel(decoder)

            encoder = encoder.to(self.device)
            encoder = nn.DataParallel(encoder)

            if self.config['training']['take_model_average']:
                decoder_test = copy.deepcopy(decoder)
                checkpoint_io.register_modules(decoder_test=decoder_test)
            else:
                decoder_test = decoder

            checkpoint_io.register_modules(decoder=decoder, encoder=encoder)

            try:
                it = checkpoint_io.load(model_name, pretrained=self.pretrained)
                assert (it != -1)
            except Exception as e:
                # try again without data parallel
                print(e)
                checkpoint_io.register_modules(decoder=decoder.module)
                checkpoint_io.register_modules(encoder=encoder.module)
                checkpoint_io.register_modules(decoder_test=decoder_test.module)
                it = checkpoint_io.load(model_name, pretrained=self.pretrained)
                assert (it != -1)

            print('Loaded iteration:', it['it'])
            return decoder_test, encoder
        else:
            print("size img config" , self.config['data']['img_size'])
            label_generator = label_generator.to(self.device)
            label_generator = nn.DataParallel(label_generator)

            decoder = decoder.to(self.device)
            decoder = nn.DataParallel(decoder)

            checkpoint_io.register_modules(label_generator = label_generator, decoder = decoder)

            if iteration_label_gen!=None:
                it_lab = iteration_label_gen
            else:
                it_lab = utils.get_most_recent(os.path.join(exp_out_dir, 'chkpts'), 'model')
            print("Loading iteration from label gen pretrained model : ", it_lab)

            try:
                it= checkpoint_io.load(os.path.join(exp_out_dir, 'chkpts','model_%08d.pt' % it_lab))
                assert (it != -1)
            except Exception as e:
                print(e)
                checkpoint_io.register_modules(decoder=decoder.module)
                checkpoint_io.register_modules(label_generator=label_generator.module)
                it = checkpoint_io.load(os.path.join(exp_out_dir, 'chkpts', 'model_%08d.pt' % it_lab))
                assert (it != -1)




            return decoder, label_generator

    def get_yz_dist(self):
        '''loads the z and y dists used to sample from the generator.'''

        if self.config['clusterer']['name'] != 'supervised':
            print(self.clusterer_path)
            if 'clusterer' in self.pretrained:
                clusterer = self.checkpoint_io.load_clusterer('pretrained', load_samples=False, pretrained=self.pretrained)
            elif os.path.exists(self.clusterer_path):
                with open(self.clusterer_path, 'rb') as f:
                    clusterer = pickle.load(f)

            if isinstance(clusterer.discriminator, nn.DataParallel):
                clusterer.discriminator = clusterer.discriminator.module

            if clusterer.kmeansG is not None:
                # use clusterer empirical distribution as sampling
                print('Using k-means empirical distribution')
                distribution = clusterer.get_label_distribution()
                probs = [f / sum(distribution) for f in distribution]
            else:
                # otherwise, use a uniform distribution. this is not desired, unless it's a random label or unconditional GAN
                print("Sampling with uniform distribution over", clusterer.k, "labels")
                probs = [1. / clusterer.k for _ in range(clusterer.k)]
        else:
            # if it's supervised, then sample uniformly over all classes.
            # this might not be the right thing to do, since datasets are usually imbalanced.
            print("Sampling with uniform distribution over", self.nlabels,
                  "labels")
            probs = [1. / self.nlabels for _ in range(self.nlabels)]
        self.clusterer=clusterer
        return YZDataset(zdim=self.config['z_dist']['dim'],
                         nlabels=len(probs),
                         distribution=probs,
                         device=self.device)
