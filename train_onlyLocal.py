import argparse
import os
import copy
import pprint
from os import path
import sys
import shutil

import torch
import numpy as np
from torch import nn
from torch.nn import init

from gan_training import utils

from gan_training.train_BiGAN import Trainer
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (load_config, build_models, build_optimizers)
from seeing.pidfile import exit_if_job_done, mark_job_done

import torchvision.transforms as transforms

torch.backends.cudnn.benchmark = True

# Arguments
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--outdir', type=str, help='used to override outdir (useful for multiple runs)')
parser.add_argument('--nepochs', type=int, default=250, help='number of epochs to run before terminating')
parser.add_argument('--model_it', type=int, default=-1, help='which model iteration to load from, -1 loads the most recent model')
parser.add_argument('--devices', nargs='+', type=str, default=['0','1'], help='devices to use')

args = parser.parse_args()
config = load_config(args.config, 'configs/default.yaml')
out_dir = config['training']['out_dir'] if args.outdir is None else args.outdir


def see_cluster_frequency(train_loader, encoder):
    x_test, y_test = utils.get_nsamples(train_loader, 50000)
    batch_size = 100
    tot_count = torch.zeros(50, device = 'cuda')
    tot_sum = x_test.size(0)

    for batch in range(x_test.size(0) // batch_size):
        x_batch = x_test[batch * batch_size:(batch + 1) * batch_size].cuda()
        _,label_maps = encoder(x_batch)
        index = torch.argmax(label_maps, dim = 1)
        print(index[:5])
        count = torch.bincount(index[:,0,0].view(-1), minlength=config['encoder']['n_locallabels'])
        tot_count = tot_count + count
    if (x_test.size(0) % batch_size != 0):
        x_batch = x_test[x_test.size(0) // batch_size * batch_size:].cuda()
        _, label_maps = encoder(x_batch)
        index = torch.argmax(label_maps, dim=1)
        count = torch.bincount(index[:,0,0].view(-1), minlength=config['encoder']['n_locallabels'])
        tot_count = tot_count + count
    torch.set_printoptions(precision=6, sci_mode=False)
    print(torch.sum(tot_count), tot_sum)
    print((tot_count/tot_sum*100))
    sys.exit()

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

def main():
    pp = pprint.PrettyPrinter(indent=1)
    pp.pprint({
        'data': config['data'],
        'decoder': config['decoder'],
        'encoder': config['encoder'],
        'training': config['training'],
        'devices': args.devices
    })
    is_cuda = torch.cuda.is_available()

    # Short hands
    batch_size = config['training']['batch_size']
    log_every = config['training']['log_every']
    inception_every = config['training']['inception_every']
    backup_every = config['training']['backup_every']
    nlabels = config['data']['nlabels']

    checkpoint_dir = path.join(out_dir, 'chkpts')

    # Create missing directories
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    shutil.copyfile(args.config, os.path.join(out_dir, "config.yaml"))
    # Logger
    checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)


    devices = [i for i in range(len(args.devices))]
    device = torch.device(devices[0] if is_cuda else "cpu")

    train_dataset, _, test_dataset = get_dataset(
        name=config['data']['type'],
        data_dir=config['data']['train_dir'],
        size=config['data']['img_size'],
        deterministic=config['data']['deterministic'])

    test_loader=None
    if test_dataset!=None:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=config['training']['nworkers'],
            shuffle=False,
            pin_memory=True,
            sampler=None,
            drop_last=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=config['training']['nworkers'],
        shuffle=True,
        pin_memory=True,
        sampler=None,
        drop_last=True)

    # Create models
    decoder, encoder, discriminator, label_generator, qhead_discriminator = build_models(config)

    # Put models on gpu if needed
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    discriminator = discriminator.to(device)
    label_generator = label_generator.to(device)

    print("USING QHEAD DISCRIMINATOR")
    qhead_discriminator = qhead_discriminator.to(device)

    for name, module in encoder.named_modules():
        if isinstance(module, nn.Sigmoid):
            print('Found sigmoid layer in encoder; not compatible with BCE with logits')
            exit()

    dec_optimizer, enc_optimizer, disc_optimizer, label_gen_optimizer = build_optimizers(decoder, encoder, discriminator, label_generator, config, qhead_disc = qhead_discriminator)

    decoder = nn.DataParallel(decoder, device_ids=devices)
    encoder = nn.DataParallel(encoder, device_ids=devices)
    discriminator = nn.DataParallel(discriminator, device_ids=devices)
    label_generator = nn.DataParallel(label_generator, device_ids=devices)
    qhead_discriminator = nn.DataParallel(qhead_discriminator, device_ids=devices)

    # Register modules to checkpoint
    checkpoint_io.register_modules(decoder=decoder,
                                   encoder=encoder,
                                   label_generator = label_generator,

                                   discriminator= discriminator,
                                   qhead_discriminator=qhead_discriminator,
                                   disc_optimizer=disc_optimizer,
                                   enc_optimizer=enc_optimizer,
                                   dec_optimizer=dec_optimizer,
                                   label_gen_optimizer = label_gen_optimizer
                                   )


    # Logger
    logger = Logger(log_dir=path.join(out_dir, 'logs'),
                    img_dir=path.join(out_dir, 'imgs'),
                    monitoring=config['training']['monitoring'],
                    monitoring_dir=path.join(out_dir, 'monitoring'))

    # Distributions
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'], device=device)
    zdist_lab = get_zdist(config['z_dist']['type'], config['label_generator']['zdim'], device=device)

    ntest = config['training']['ntest']
    if test_loader!=None:
        x_test, y_test = utils.get_nsamples(test_loader, ntest)
    else:
        x_test, y_test = utils.get_nsamples(train_loader, ntest)

    x_test, y_test = x_test.to(device), y_test.to(device)
    z_test = zdist.sample((ntest, ))
    utils.save_images(x_test, path.join(out_dir, 'real.png'))
    logger.add_imgs(x_test, 'gt', 0)




    # Test decoder
    if config['training']['take_model_average']:
        print('Taking model average')
        bad_modules = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
        for model in [decoder, encoder]:
            for name, module in model.named_modules():
                for bad_module in bad_modules:
                    if isinstance(module, bad_module):
                        print('Batch norm in encoder not compatible with exponential moving average')
                        exit()
        decoder_test = copy.deepcopy(decoder)
        checkpoint_io.register_modules(decoder_test=decoder_test)
    else:
        decoder_test = decoder


    decoder.apply(init_weights)
    encoder.apply(init_weights)
    discriminator.apply(init_weights)
    label_generator.apply(init_weights)
    qhead_discriminator.apply(init_weights)

    # Load checkpoint if it exists
    it = utils.get_most_recent(checkpoint_dir, 'model') if args.model_it == -1 else args.model_it
    it, epoch_idx = checkpoint_io.load_models(it=it)

    # Evaluator
    evaluator = Evaluator(
        decoder_test,
        encoder,
        zdist,
        zdist_lab,
        label_generator = label_generator,
        train_loader=train_loader,
        batch_size=batch_size,
        device=device,
        inception_nsamples=config['training']['inception_nsamples'])

    # Trainer
    # print(label_discriminator)
    print("GAN TYPE : ", config['training']['gan_type'])
    trainer = Trainer(decoder,
                      encoder,
                      discriminator=discriminator,
                      label_generator = label_generator,
                      qhead_discriminator = qhead_discriminator,
                      disc_optimizer= disc_optimizer,
                      enc_optimizer=enc_optimizer,
                      dec_optimizer=dec_optimizer,
                      label_gen_optimizer=label_gen_optimizer,
                      gan_type=config['training']['gan_type'],
                      con_loss = config['training']['con_loss'] if 'con_loss' in config['training'] else False,
                      decDeterministic = config['decoder']['deterministicOnSeg'])


    # Training loop
    # see_cluster_frequency(train_loader, encoder)
    print('Start training...')
    while it < args.nepochs * len(train_loader):
        epoch_idx += 1
        for x_real, y in train_loader:
            it += 1

            x_real, y = x_real.to(device), y
            z = zdist.sample((batch_size, ))
            z_lab = zdist_lab.sample((batch_size,))
            zbis = zdist.sample((batch_size,))

            gloss = trainer.encoderdecoder_trainstep(x_real, z, z_lab=z_lab, check_norm = (it%200 ==0))

            dloss = trainer.discriminator_trainstep(x_real, z, z_lab)

            for key, value in gloss.items():
                logger.add('losses', key, value, it=it)
            for key, value in dloss.items():
                logger.add('losses', key, value, it=it)

            # Print stats
            if it % log_every == 0:
                print('[epoch %0d, it %4d]' % (epoch_idx, it))
                print(gloss)
                print(dloss)

            # (i) Sample if necessary
            if it % config['training']['sample_every'] == 0:
                print("it", it)
                print('Creating samples...')

                x = evaluator.create_samples(x_test, z_test)
                logger.add_imgs(x, 'all', it)

                z_lab = zdist_lab.sample((ntest,))
                x = evaluator.create_samples_labelGen(z_test, z_lab, out_dir=out_dir)
                logger.add_imgs(x, 'all', it+1)


            # (ii) Compute inception if necessary
            if (it -1) % inception_every == 0 and it > 1 or it == 5001:
                print('PyTorch Inception score...')
                inception_mean_label, inception_std_label = evaluator.compute_inception_score(labelgen=True)
                logger.add('metrics', 'pt_inception_mean', inception_mean_label, it=it)
                logger.add('metrics', 'pt_inception_stddev', inception_std_label, it=it)
                print(
                    f'[epoch {epoch_idx}, it {it}] for label gen pt_inception_mean: {inception_mean_label}, pt_inception_stddev: {inception_std_label}')

            # (iii) Backup if necessary
            if it % backup_every == 0 or it ==1000:
                print('Saving backup...')
                checkpoint_io.save('model_%08d.pt' % it, it=it)

                logger.save_stats('stats_%08d.p' % it)

                if it > 0:
                    checkpoint_io.save('model.pt', it=it)


if __name__ == '__main__':
    exit_if_job_done(out_dir)
    main()
    mark_job_done(out_dir)
