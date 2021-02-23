import argparse
import json
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

from gan_training.train_BiGAN import Trainer, update_average
from gan_training.logger import Logger
from gan_training.checkpoints import CheckpointIO
from gan_training.inputs import get_dataset
from gan_training.distributions import get_ydist, get_zdist
from gan_training.eval import Evaluator
from gan_training.config import (load_config, build_models, build_optimizers)
from seeing.pidfile import exit_if_job_done, mark_job_done

import torchvision.transforms as transforms
from torch.nn import functional as F

torch.backends.cudnn.benchmark = True

# Arguments
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--outdir', type=str, help='used to override outdir (useful for multiple runs)')
parser.add_argument('--nepochs', type=int, default=250, help='number of epochs to run before terminating')
parser.add_argument('--model_it', type=int, default=-1, help='which model iteration to load from, -1 loads the most recent model')
parser.add_argument('--devices', nargs='+', type=str, default=['0','1'], help='devices to use')
parser.add_argument('--eval_mode', action='store_true', help='save sample images')
parser.add_argument('--niterBeforeLRDecay', type=int, default=100, help='number of epochs to run before lr decay')
parser.add_argument('--niter_decay', type=int, default=100, help='number of epochs to run before lr decay')

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
    fid_every = config['training']['fid_every']
    backup_every = config['training']['backup_every']
    nlabels = config['data']['nlabels']

    checkpoint_dir = path.join(out_dir, 'chkpts')

    # Create missing directories
    if not path.exists(out_dir):
        os.makedirs(out_dir)
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    #to compute online fid
    results_online_fid = path.join(out_dir, "fid")
    if not path.exists(results_online_fid):
        os.makedirs(results_online_fid)
    results_file_fid = os.path.join(results_online_fid, 'fid_results.json')
    if not os.path.exists(results_file_fid):
        with open(results_file_fid, 'w') as f:
            f.write(json.dumps({}))


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

    encdec_optimizer, disc_optimizer = build_optimizers(decoder, encoder, discriminator, label_generator, config, qhead_disc = qhead_discriminator)

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
                                   encdec_optimizer=encdec_optimizer
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

    decoder.apply(init_weights)
    encoder.apply(init_weights)
    discriminator.apply(init_weights)
    label_generator.apply(init_weights)
    qhead_discriminator.apply(init_weights)

    # Test decoder
    if config['training']['take_model_average']:
        print('Taking model average')
        bad_modules = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
        for model in [decoder, label_generator]:
            for name, module in model.named_modules():
                for bad_module in bad_modules:
                    if isinstance(module, bad_module):
                        print(f'Batch norm in {model.module.__class__.__name__} not compatible with exponential moving average')
                        #exit()

        decoder_test = copy.deepcopy(decoder)
        checkpoint_io.register_modules(decoder_test=decoder_test)
        # decoder_test = decoder

        label_generator_test = copy.deepcopy(label_generator)
        checkpoint_io.register_modules(label_generator_test=label_generator_test)
        # label_generator_test = label_generator
    else:
        decoder_test = decoder
        label_generator_test = label_generator




    # Load checkpoint if it exists
    it = utils.get_most_recent(checkpoint_dir, 'model') if args.model_it == -1 else args.model_it
    it, epoch_idx = checkpoint_io.load_models(it=it)

    # Evaluator
    evaluator = Evaluator(
        decoder_test,
        encoder,
        zdist,
        zdist_lab,
        label_generator = label_generator_test,
        train_loader=train_loader,
        batch_size=batch_size,
        device=device,
        inception_nsamples=config['training']['inception_nsamples'],
        n_locallabels=config['encoder']['n_locallabels'])

    # Trainer
    # print(label_discriminator)
    print("GAN TYPE : ", config['training']['gan_type'])
    trainer = Trainer(decoder,
                      encoder,
                      discriminator=discriminator,
                      label_generator = label_generator,
                      qhead_discriminator = qhead_discriminator,
                      disc_optimizer= disc_optimizer,
                      encdec_optimizer=encdec_optimizer,
                      gan_type=config['training']['gan_type'],
                      con_loss = config['training']['con_loss'] if 'con_loss' in config['training'] else False,
                      entropy_loss=config['training']['entropy_loss'] if 'entropy_loss' in config['training'] else False,
                      decDeterministic = config['decoder']['deterministicOnSeg'],
                      lambda_LabConLoss = config['training']['lambda_LabConLoss'] if 'lambda_LabConLoss' in config['training'] else 1,
                      n_locallabels=config['encoder']['n_locallabels'],
                      reg_type=config['training']['reg_type'],
                      reg_param=config['training']['reg_param']
                      )

    if args.eval_mode:
        # test with same z_img, different segmentations
        n_samples = 16
        for i in range(20):
            z_lab = zdist_lab.sample((n_samples,))
            z_test_same = zdist.sample((1,))
            z_test_same = torch.cat((z_test_same,) * n_samples, dim=0)
            print(z_test_same.size())
            x,_ = evaluator.create_samples_labelGen(z_test_same, z_lab, out_dir=out_dir)
            logger.add_imgs(x, 'sameZImg', i)

        # test with different z_img, same seg
        for i in range(20):
            z_test = zdist.sample((n_samples,))
            z_lab_same = zdist_lab.sample((1,))
            z_lab_same = torch.cat((z_lab_same,) * n_samples, dim=0)
            print(z_lab_same.size())
            x,x_c = evaluator.create_samples_labelGen(z_test, z_lab_same, out_dir=out_dir)
            x = torch.cat((torch.unsqueeze(x_c[0].float().cuda(),dim=0), x), dim = 0)
            logger.add_imgs(x, 'sameZLab', i)
        sys.exit()

    # see_cluster_frequency(train_loader, encoder)
    print('Start training...')
    # for param_group in encdec_optimizer.param_groups:
    #     lr = param_group['lr']
    #     if epoch_idx >= args.niterBeforeLRDecay:
    #         lr = lr*2

    # Training loop
    while it < args.nepochs * len(train_loader):
        epoch_idx += 1
        # lr = update_learning_rate(epoch_idx, lr, encdec_optimizer, disc_optimizer)

        for x_real, y in train_loader:
            it += 1

            x_real, y = x_real.to(device), y
            z = zdist.sample((batch_size, ))
            z_lab = zdist_lab.sample((batch_size,))
            zbis = zdist.sample((batch_size,))

            gloss = trainer.encoderdecoder_trainstep(x_real, z, z_lab=z_lab, check_norm = (it%200 ==0))

            dloss = trainer.discriminator_trainstep(x_real, z, z_lab)

            if config['training']['take_model_average']:
                update_average(decoder_test, decoder, beta=config['training']['model_average_beta'])
                update_average(label_generator_test, label_generator, beta=config['training']['model_average_beta'])

            for key, value in gloss.items():
                logger.add('losses', key, value, it=it)
            for key, value in dloss.items():
                logger.add('losses', key, value, it=it)

            # Print stats
            if it % log_every == 0:
                print('[epoch %0d, it %4d]' % (epoch_idx, it), gloss, dloss)

            # (i) Sample if necessary
            if it % config['training']['sample_every'] == 0:
                print("it", it)
                print('Creating samples...')

                z_test = zdist.sample((ntest,))
                x, lab_color = evaluator.create_samples(x_test, z_test)
                logger.add_imgs(x, 'all', it)
                logger.add_imgs(lab_color, 'all', it+2)

                z_lab = zdist_lab.sample((ntest,))
                x,_ = evaluator.create_samples_labelGen(z_test, z_lab, out_dir=out_dir)
                logger.add_imgs(x, 'all', it+1)

                with torch.no_grad():
                    _, label_map = label_generator(z_lab)
                    x_fake = decoder(seg=label_map, input=z_test)
                    logger.add_imgs(x_fake, 'all', it + 3)

            # (ii) Compute inception if necessary
            if (it - 1) % inception_every == 0 and it > 1 and False or it == 5001:
                print('PyTorch Inception score...')
                inception_mean_label, inception_std_label = evaluator.compute_inception_score(labelgen=True)
                logger.add('metrics', 'pt_inception_mean', inception_mean_label, it=it)
                logger.add('metrics', 'pt_inception_stddev', inception_std_label, it=it)
                print(
                    f'[epoch {epoch_idx}, it {it}] for label gen pt_inception_mean: {inception_mean_label}, pt_inception_stddev: {inception_std_label}')

            if (it - 1) % fid_every == 0 and it > 1 and False:
                print('Tensorflow FID score...')
                evaluator.compute_fid_score(results_online_fid, it=it)


            # (iii) Backup if necessary
            if it % backup_every == 0 or it ==500 or it==1000:
                print('Saving backup...')
                checkpoint_io.save('model_%08d.pt' % it, it=it, epoch_idx = epoch_idx)
                logger.save_stats('stats_%08d.p' % it)
                if it > 0:
                    checkpoint_io.save('model.pt', it=it)


def update_learning_rate(epoch, old_lr, encdec_optim, disc_optim):
    if epoch >= args.niterBeforeLRDecay:
        lrd = config['training']['lr_g'] / (4 * args.niter_decay)
        new_lr = old_lr - lrd
    else:
        new_lr = old_lr

    if new_lr != old_lr:
        new_lr_G = new_lr / 2
        new_lr_D = new_lr * 2

        for param_group in encdec_optim.param_groups:
            param_group['lr'] = new_lr_G
        for param_group in disc_optim.param_groups:
            param_group['lr'] = new_lr_D
        print('update learning rate: %f -> %f' % (old_lr, new_lr))
    return new_lr

if __name__ == '__main__':
    exit_if_job_done(out_dir)
    main()
    mark_job_done(out_dir)
