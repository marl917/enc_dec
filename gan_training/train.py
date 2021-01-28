# coding: utf-8
import torch
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import numpy as np
from torch.autograd import Variable
from torch import nn
import torchvision
from torchvision import transforms
import sys


class Trainer(object):
    def __init__(self,
                 decoder,
                 encoder,
                 d_optimizer,
                 e_optimizer,
                 gan_type,
                 reg_type,
                 reg_param,
                 discriminator=None,
                 disc_optimizer=None,
                 label_discriminator=None,
                 label_generator=None,
                 lab_disc_optimizer=None,
                 lab_gen_optimizer=None,
                 is_cuda=True,
                 n_locallabels=1,
                 featureMatchingLoss=True,
                 recon_loss_label=True,
                 labRecon_true_equ=False,
                 recon_loss_img = True,
                 vgg_loss=True,
                 equiv_loss=False):

        self.decoder = decoder
        self.encoder = encoder
        self.d_optimizer = d_optimizer
        self.e_optimizer = e_optimizer
        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer

        self.label_discriminator = label_discriminator
        self.label_generator = label_generator
        self.lab_disc_optimizer = lab_disc_optimizer
        self.lab_gen_optimizer = lab_gen_optimizer

        self.gan_type = gan_type
        self.reg_type = reg_type
        self.reg_param = reg_param

        self.featureMatchingLoss=featureMatchingLoss
        self.recon_loss_label = recon_loss_label
        self.recon_loss_img = recon_loss_img
        self.true_equation = labRecon_true_equ
        self.vgg_loss=vgg_loss
        self.equiv_loss = equiv_loss
        print("USING featureMatching Loss : ", self.featureMatchingLoss)
        print("USING Recon Label Loss : ", self.recon_loss_label, " With True equation : ", labRecon_true_equ)
        print("USING recon img loss : ", self.recon_loss_img)
        print("USING vgg loss : ", self.vgg_loss)
        print("USING equiv loss : ", self.equiv_loss)

        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')

        self.FloatTensor = torch.cuda.FloatTensor if is_cuda \
            else torch.FloatTensor
        self.n_locallabels = n_locallabels

        print('D reg gamma', self.reg_param)
        self.mse = nn.MSELoss()
        self.criterionFeat = torch.nn.L1Loss()
        self.criterionVGG = VGGLoss()
        self.crossentropy = torch.nn.CrossEntropyLoss()
        self.logsoftmax = nn.LogSoftmax(dim=1)

        if self.equiv_loss:
            self.cj_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2),
                transforms.ToTensor(), ])
            self.kl = nn.KLDivLoss(reduction='batchmean')

    def crossentropy_soft(self,source, target):
        soft_source = -self.logsoftmax(source)
        soft_target = F.softmax(target, dim=1)

        cross_ent = soft_source * soft_target

        return torch.mean(torch.sum(cross_ent, dim=(1,2,3)))

    def encoderdecoder_trainstep(self, x_real, z, z_lab = None, zbis=None):
        toggle_grad(self.decoder, True)
        toggle_grad(self.encoder, True)
        toggle_grad(self.label_generator, True)

        toggle_grad(self.discriminator, False)
        toggle_grad(self.label_discriminator, False)

        self.encoder.train()
        self.decoder.train()
        self.label_generator.train()
        self.discriminator.train()
        self.label_discriminator.train()


        self.d_optimizer.zero_grad()
        self.e_optimizer.zero_grad()
        self.lab_gen_optimizer.zero_grad()
        
        
        
        #first part : train label and img gen pair with discriminator
        seg_fake_unorm, seg_fake = self.label_generator(z_lab)
        x_fake = self.decoder(z, seg_fake)

        # local_g_fake, g_fake = self.discriminator(x_fake,  seg=seg_fake)
        # gloss = self.compute_loss(g_fake, 1)
        # gloss.backward(retain_graph=True)
        # gloss_local = self.compute_loss(local_g_fake, 1)
        # gloss_local.backward(retain_graph=True)

        toggle_grad(self.encoder, False)
        label_map_fake_unorm, label_map_fake= self.encoder(x_fake)




        #2nd part : train encoder/decoder pair to reconstruct real images
        toggle_grad(self.encoder, True)
        label_map_real_unorm, label_map_real = self.encoder(x_real)
        x_fake_enc = self.decoder(zbis, label_map_real)

        ##use GAN loss for part 2:
        # x_fake_total = torch.cat((x_fake, x_fake_enc), dim=0)
        # seg_tot = torch.cat((seg_fake, label_map_real), dim=0)
        x_fake_total = x_fake
        seg_tot = seg_fake
        local_g_fake, g_fake = self.discriminator(x_fake_total, seg=seg_tot)
        gloss = self.compute_loss(g_fake, 1)
        gloss.backward(retain_graph=True)
        gloss_local = self.compute_loss(local_g_fake, 1)
        gloss_local.backward(retain_graph=True)

        recon_loss_label = 0.1 * self.crossentropy_soft(seg_fake_unorm.detach(), label_map_fake_unorm)
        recon_loss_label.backward()
        # local_g_fake_enc, g_fake_enc = self.discriminator(x_fake_enc, seg=label_map_real)
        # gloss_enc = self.compute_loss(g_fake_enc, 1)
        # gloss_enc.backward(retain_graph=True)
        # gloss_local_enc = self.compute_loss(local_g_fake_enc, 1)
        # gloss_local_enc.backward(retain_graph=True)

        recon_loss_img = torch.tensor(0., device='cuda')
        # if self.recon_loss_img:
        #     recon_loss_img = self.mse(x_fake_enc, x_real.detach())
        #
        VGG_loss = torch.tensor(0., device='cuda')
        # if self.vgg_loss:
        #     VGG_loss = self.criterionVGG(x_fake_enc, x_real) * 10

        #3rd part: train encoder with equivariance loss
        equiv_loss =  torch.tensor(0., device = 'cuda')
        if self.equiv_loss:
            #trasnformed images passed through the encoder
            images_cj = (x_real + 1.0) / 2.0
            images_cj = images_cj.cpu()
            for b in range(images_cj.shape[0]):
                images_cj[b] = torch.from_numpy((self.cj_transform(images_cj[b]).numpy() * 2) - 1)
            images_cj = images_cj.cuda()
            images_tps = images_cj.flip(-1)
            label_tps_unorm, label_tps_hard = self.encoder(images_tps)

            #directly transform the label map from the original img
            true_label_d = label_map_real_unorm.detach()
            true_label_d.requires_grad = False
            true_label_trans = true_label_d.flip(-1)

            equiv_loss = 0.1 * self.kl(F.log_softmax(label_tps_unorm, dim=1), F.softmax(true_label_trans, dim=1))



        # total_loss = ( recon_loss_img + VGG_loss + equiv_loss ).mean()
        # total_loss.backward()

        # for p in self.encoder.parameters():
        #     print(p.grad)

        self.d_optimizer.step()
        # self.e_optimizer.step()
        self.lab_gen_optimizer.step()

        return recon_loss_img.item(), recon_loss_label.item(), VGG_loss.item(), (gloss + gloss_local).item(), equiv_loss.item()

    def discriminator_trainstep(self, x_real, z, z_lab, zbis=None):
        toggle_grad(self.encoder, False)
        toggle_grad(self.decoder, False)
        toggle_grad(self.label_generator, False)
        toggle_grad(self.discriminator, True)
        toggle_grad(self.label_discriminator, True)
        self.encoder.train()
        self.decoder.train()
        self.label_generator.train()
        # self.label_discriminator.train()
        self.discriminator.train()

        self.disc_optimizer.zero_grad()
        # self.lab_disc_optimizer.zero_grad()

        # On real data
        with torch.no_grad():
            _,label_map = self.encoder(x_real)
        x_real.requires_grad_()

        local_d_real, d_real = self.discriminator(x_real, seg=label_map)
        dloss_real = self.compute_loss(d_real, 1)
        dloss_real.backward(retain_graph=True)
        dloss_local_real = self.compute_loss(local_d_real, 1)
        dloss_local_real.backward()


        # On fake data
        with torch.no_grad():
            seg_fake_unorm, seg_fake = self.label_generator(z_lab)
            x_fake = self.decoder(z, seg_fake)

        x_fake.requires_grad_()
        # local_d_fake, d_fake = self.discriminator(x_fake, seg=seg_fake)
        # dloss_fake= self.compute_loss(d_fake, 0)
        # dloss_fake.backward(retain_graph=True)
        # dloss_local_fake = self.compute_loss(local_d_fake, 0)
        # dloss_local_fake.backward()

        #on fake data using seg from encoder
        # with torch.no_grad():
        #     x_fake_enc = self.decoder(zbis, label_map)
        #
        # x_fake_enc.requires_grad_()
        # local_d_fake_enc, d_fake_enc = self.discriminator(x_fake_enc, seg=label_map)
        # dloss_fake_enc = self.compute_loss(d_fake_enc, 0)
        # dloss_fake_enc.backward(retain_graph=True)
        # dloss_local_fake_enc = self.compute_loss(local_d_fake_enc, 0)
        # dloss_local_fake_enc.backward()


        #test
        # x_fake_total = torch.cat((x_fake, x_fake_enc), dim=0)
        # seg_tot= torch.cat((seg_fake, label_map), dim = 0)
        x_fake_total = x_fake
        seg_tot = seg_fake
        local_d_fake, d_fake = self.discriminator(x_fake_total, seg=seg_tot)
        dloss_fake = self.compute_loss(d_fake, 0)
        dloss_fake.backward(retain_graph=True)
        dloss_local_fake = self.compute_loss(local_d_fake, 0)
        dloss_local_fake.backward()

        self.disc_optimizer.step()

        dloss = (dloss_real + dloss_fake + dloss_local_fake + dloss_local_real)
        if self.reg_type == 'none':
            reg = torch.tensor(0.)
        # print(dloss)
        return dloss.item(), reg.item()

    def compute_loss(self, d_out, target):
        targets = d_out.new_full(size=d_out.size(), fill_value=target)

        if self.gan_type == 'standard':
            loss = F.binary_cross_entropy_with_logits(d_out, targets)
        elif self.gan_type == 'wgan':
            loss = (2 * target - 1) * d_out.mean()
        else:
            raise NotImplementedError

        return loss

    def wgan_gp_reg(self, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1, 1)
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = self.discriminator(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg


# Utility functions
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(outputs=d_out.sum(),
                              inputs=x_in,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def update_average(model_tgt, model_src, beta):
    toggle_grad(model_src, False)
    toggle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()

        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
