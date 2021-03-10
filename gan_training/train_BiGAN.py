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
                 gan_type,

                 discriminator,
                 label_generator,
                 qhead_discriminator=None,
                 disc_optimizer=None,
                 encdec_optimizer=None,
                 decDeterministic = False,

                 con_loss = False,
                 equiv_loss =False,

                 lambda_LabConLoss=1,
                 n_locallabels=0,
                 reg_type=None,
                 reg_param=None,
                 con_loss_img=True):

        self.decoder = decoder
        self.encoder = encoder
        self.discriminator = discriminator
        self.label_generator = label_generator
        self.qhead_discriminator = qhead_discriminator

        self.disc_optimizer = disc_optimizer
        self.encdec_optimizer = encdec_optimizer

        self.lambda_LabConLoss = lambda_LabConLoss

        self.gan_type = gan_type
        self.reg_type = reg_type
        self.reg_param = reg_param

        self.n_locallabels = n_locallabels

        self.con_loss = con_loss
        self.con_loss_img = con_loss_img
        self.equiv_loss = equiv_loss
        self.decDeterministic = decDeterministic
        print("TRAINING WITH CON LOSS : ", con_loss)
        print("TRAINING WITH EQUIV LOSS : ", equiv_loss)

        if self.equiv_loss:
            self.cj_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2),
                transforms.ToTensor(), ])
            self.kl = nn.KLDivLoss(reduction='batchmean')


    def normalNLLLoss(self, x, mu, var):

        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll

    def gaussian(self, ins, mean=0, stddev=0.1):
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        return ins + noise


    def encoderdecoder_trainstep(self, x_real, z, z_lab = None,check_norm = False):
        toggle_grad(self.decoder, True)
        toggle_grad(self.encoder, True)
        toggle_grad(self.label_generator, True)
        toggle_grad(self.discriminator, False)

        if self.qhead_discriminator!=None:
            toggle_grad(self.qhead_discriminator, True)
            self.qhead_discriminator.train()

        self.encoder.train()
        self.decoder.train()
        self.label_generator.train()
        self.discriminator.train()


        self.encdec_optimizer.zero_grad()


        G_losses = {}


        # first part : train label generator and img decoder
        seg_fake_unorm, seg_fake = self.label_generator(z_lab)
        x_fake = self.decoder(seg = seg_fake, input = z)  #seg fake detach
        g_fake = self.discriminator(x_fake,  seg=seg_fake)


        con_loss_lab = torch.tensor(0., device='cuda')
        con_loss_img = torch.tensor(0., device='cuda')
        if self.con_loss:
            mu_lab, var_lab, mu_img, var_img = self.qhead_discriminator(g_fake[0])
            con_loss_lab = self.normalNLLLoss(z_lab, mu_lab, var_lab) * self.lambda_LabConLoss
            if self.con_loss_img:
                con_loss_img = self.normalNLLLoss(z, mu_img, var_img) * 0.1
            G_losses['con_loss_img'] = con_loss_img.item()
            G_losses['con_loss_lab'] = con_loss_lab.item()
            # print("mu and var max min", torch.min(var), torch.max(var))

        #second part : train encoder
        label_map_real_unorm, label_map_real= self.encoder(x_real)
        g_real_enc = self.discriminator(x_real, seg=label_map_real)

        if len(g_fake)>2:
            gloss = torch.stack([self.compute_loss(g_fake_e, 1) for g_fake_e in g_fake[1:]]).sum() + torch.stack([self.compute_loss(g_real_enc_e, 0) for g_real_enc_e in g_real_enc[1:]]).sum()
        else:
            gloss = self.compute_loss(g_fake[1], 1) + self.compute_loss(g_real_enc[1], 0)

        G_losses['gloss'] = gloss.item()


        #third part : equivariance loss
        equiv_loss = torch.tensor(0., device='cuda')
        if self.equiv_loss:
            # trasnformed images passed through the encoder
            images_cj = (x_real + 1.0) / 2.0
            images_cj = images_cj.cpu()
            for b in range(images_cj.shape[0]):
                images_cj[b] = torch.from_numpy((self.cj_transform(images_cj[b]).numpy() * 2) - 1)
            images_cj = images_cj.cuda()
            images_tps = images_cj.flip(-1)
            label_tps_unorm, label_tps_hard = self.encoder(images_tps)

            # directly transform the label map from the original img
            true_label_d = label_map_real_unorm.detach()
            true_label_d.requires_grad = False
            true_label_trans = true_label_d.flip(-1)

            equiv_loss = 0.1 * self.kl(F.log_softmax(label_tps_unorm, dim=1), F.softmax(true_label_trans, dim=1))
            G_losses['equiv_loss'] = equiv_loss.item()

        tot_loss = (gloss + con_loss_lab + con_loss_img + equiv_loss).mean()
        tot_loss.backward()

        # for p in self.encoder.parameters():
        #     print(p.grad)
        # sys.exit()
        # if check_norm and False:
        #     total_norm = 0
        #     for p in self.decoder.parameters():
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        #         total_norm = total_norm ** (1. / 2)
        #         print(total_norm)

        self.encdec_optimizer.step()

        toggle_grad(self.encoder, False)
        toggle_grad(self.decoder, False)
        toggle_grad(self.label_generator, False)

        return G_losses
        
    def discriminator_trainstep(self, x_real, z, z_lab):
        toggle_grad(self.encoder, False)
        toggle_grad(self.decoder, False)
        toggle_grad(self.label_generator, False)
        toggle_grad(self.discriminator, True)

        self.encoder.train()
        self.decoder.train()
        self.label_generator.train()
        self.discriminator.train()

        self.disc_optimizer.zero_grad()

        # On real data
        with torch.no_grad():
            _,label_map = self.encoder(x_real)
        x_real.requires_grad_()
        label_map.requires_grad_()
        d_real = self.discriminator(x_real, seg=label_map)

        if len(d_real)>2:
            dloss_real = torch.stack([self.compute_loss(d_real_e, 1) for d_real_e in d_real[1:]]).sum()
        else:
            dloss_real = self.compute_loss(d_real[1], 1)

        reg = torch.tensor(0., device='cuda')
        if self.reg_type == 'real' or self.reg_type == 'real_fake':
            dloss_real.backward(retain_graph=True)
            reg = self.reg_param * compute_grad2(d_real[1], [x_real, label_map]).mean()
            reg.backward()
        else:
            dloss_real.backward()

        # On fake data
        with torch.no_grad():
            seg_fake_unorm, seg_fake = self.label_generator(z_lab)
            x_fake = self.decoder(seg = seg_fake, input = z)

        x_fake.requires_grad_()
        seg_fake.requires_grad_()
        d_fake = self.discriminator(x_fake, seg=seg_fake)

        if len(d_fake)>2:
            dloss_fake = torch.stack([self.compute_loss(d_fake_e, 0) for d_fake_e in d_fake[1:]]).sum()
        else:
            dloss_fake= self.compute_loss(d_fake[1], 0)
        dloss_fake.backward()

        self.disc_optimizer.step()

        losses = {}
        losses['dloss_real'] = dloss_real.item()
        losses['dloss_fake'] = dloss_fake.item()
        losses['reg_loss'] = reg.item()


        toggle_grad(self.discriminator, False)

        return losses

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
    batch_size = x_in[0].size(0)
    grad_dout = autograd.grad(outputs=d_out.sum(),
                              inputs=x_in,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in[0].size())
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


    # for p_ema, p in zip(model_tgt.parameters(), model_src.parameters()):
    #     p_ema.copy_(p)
    # for b_ema, b in zip(model_tgt.buffers(), model_src.buffers()):
    #     b_ema.copy_(b)

    param_dict_src_buffer = dict(model_src.named_buffers())
    for b_name, b_tgt in model_tgt.named_buffers():
        b_src = param_dict_src_buffer[b_name]
        assert (b_src is not b_tgt)
        b_tgt.copy_(b_src)

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
