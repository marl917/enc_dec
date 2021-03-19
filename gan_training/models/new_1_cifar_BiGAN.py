import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from gan_training.models import blocks
from torch.autograd import Variable


class SEANDecoder(nn.Module):
    def __init__(self,
                 local_nlabels=0,
                 z_dim=128,
                 ngf=64,
                 size=0,
                 deterministicOnSeg=False,
                 **kwargs):
        super(SEANDecoder, self).__init__()
        self.sw = size // (2 ** 4)
        self.sh = self.sw  # assumption of square images

        self.deterministicOnSeg = True
        print("Decoder only depends on Segmentation : ", self.deterministicOnSeg)

        self.fc = nn.Conv2d(local_nlabels, ngf * 8, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)
        self.head_0 = blocks.SEANResnetBlock(8 * ngf, 8 * ngf, local_nlabels)
        self.G_middle_0 = blocks.SEANResnetBlock(8 * ngf, 8 * ngf, local_nlabels)

        self.up_1 = blocks.SEANResnetBlock(8 * ngf, 4 * ngf, local_nlabels)
        self.up_2 = blocks.SEANResnetBlock(4 * ngf, 2 * ngf, local_nlabels)
        self.up_3 = blocks.SEANResnetBlock(2 * ngf, 1 * ngf, local_nlabels, use_rgb=False)
        self.conv_img = nn.Conv2d(ngf, 3, 3, padding=1)

    def forward(self, seg, input=None):  # input=z
        # alternative : downsample label map
        style_codes = input
        out = F.interpolate(seg, size=(self.sh, self.sw))
        out = self.fc(out)

        x = self.head_0(out, seg, style_codes)
        x = self.up(x)
        x = self.G_middle_0(x, seg, style_codes)
        x = self.up(x)
        x = self.up_1(x, seg, style_codes)
        x = self.up(x)
        x = self.up_2(x, seg, style_codes)
        x = self.up(x)
        x = self.up_3(x, seg, style_codes)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        # print("output of decoder : ", torch.min(x), torch.max(x))

        return x


class Decoder(nn.Module):
    def __init__(self,
                 local_nlabels=0,
                 z_dim=128,
                 nc=3,
                 ngf=64,
                 embed_dim=256,
                 size=0,
                 deterministicOnSeg=False,
                 **kwargs):
        super(Decoder, self).__init__()
        self.sw = size // (2 ** 4)
        self.sh = self.sw  # assumption of square images

        self.deterministicOnSeg = deterministicOnSeg
        print("Decoder only depends on Segmentation : ", self.deterministicOnSeg)

        self.get_latent = blocks.Identity()
        if self.deterministicOnSeg:
            self.fc = nn.Conv2d(local_nlabels, ngf * 8, 3, padding=1)
        else:
            self.fc = nn.Linear(z_dim, self.sh * self.sw * ngf * 8)

        self.up = nn.Upsample(scale_factor=2)
        self.head_0 = blocks.SPADEResnetBlock(8 * ngf, 8 * ngf, local_nlabels)
        self.G_middle_0 = blocks.SPADEResnetBlock(8 * ngf, 8 * ngf, local_nlabels)

        self.up_1 = blocks.SPADEResnetBlock(8 * ngf, 4 * ngf, local_nlabels)
        self.up_2 = blocks.SPADEResnetBlock(4 * ngf, 2 * ngf, local_nlabels)
        self.up_3 = blocks.SPADEResnetBlock(2 * ngf, 1 * ngf, local_nlabels)
        self.conv_img = nn.Conv2d(ngf, 3, 3, padding=1)

    def forward(self, seg, input=None, y=None):  # input=z
        # alternative : downsample label map
        if self.deterministicOnSeg:
            out = F.interpolate(seg, size=(self.sh, self.sw))
            out = self.fc(out)
        else:
            out = self.get_latent(input, y)
            # print("out size in decoder : ", out.size())
            out = self.fc(out)
            out = out.view(out.size(0), -1, self.sh, self.sw)
        x = self.head_0(out, seg)
        x = self.up(x)
        x = self.G_middle_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        # print("output of decoder : ", torch.min(x), torch.max(x))

        return x



class LabelGenerator(nn.Module):
    def __init__(self,
                 z_dim=128,
                 local_nlabels=0,
                 ngf=64,
                 embed_dim=256,
                 label_size=0,
                 **kwargs):
        super(LabelGenerator, self).__init__()

        self.sw = label_size // (2 ** 2)
        self.sh = self.sw

        nc = local_nlabels

        self.fc = nn.Linear(z_dim, self.sh * self.sw * ngf * 8)

        bn = blocks.BatchNorm2d

        self.local_nlabels = local_nlabels

        self.conv1 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.bn1 = bn(ngf * 8)

        self.conv1bis = nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 1, 1)
        self.bn1bis = bn(ngf * 4)

        self.conv2 = nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, 2, 1)
        self.bn2 = bn(ngf * 4)

        self.conv2bis = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 1, 1)
        self.bn2bis = bn(ngf * 2)

        self.conv_out = nn.Sequential(nn.Conv2d(ngf * 2, nc, 3, 1, 1), nn.LogSoftmax(dim=1))

        self.FloatTensor = torch.cuda.FloatTensor if True \
            else torch.FloatTensor  # is_cuda

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=1), y

    def gumble_softmax(self, logits):
        y, y_unorm = self.gumbel_softmax_sample(logits, temperature=1.0)
        x = torch.argmax(y, dim=1)
        x = torch.unsqueeze(x, dim=1)
        bs, _, h, w = x.size()
        input_label = self.FloatTensor(bs, self.local_nlabels, h, w).zero_()
        y_hard = input_label.scatter_(1, x.long().cuda(), 1.0)
        return (y_hard - y).detach() + y, y_unorm

    def forward(self, input, y=None):
        out = self.fc(input)

        out = out.view(out.size(0), -1, self.sh, self.sw)
        out = F.relu(self.bn1(self.conv1(out), y))
        out = F.relu(self.bn1bis(self.conv1bis(out), y))
        out = F.relu(self.bn2(self.conv2(out), y))
        out = F.relu(self.bn2bis(self.conv2bis(out), y))
        logits = self.conv_out(out)
        label_map, y_unorm = self.gumble_softmax(logits)
        return y_unorm, label_map

class Encoder(nn.Module):
    def __init__(self,
                 local_nlabels=None,
                 nc=3,
                 ndf=64,
                 **kwargs):
        super(Encoder, self).__init__()

        # assert conditioning != 'unconditional' or nlabels == 1

        self.FloatTensor = torch.cuda.FloatTensor if True \
            else torch.FloatTensor  # is_cuda

        self.local_nlabels = local_nlabels

        self.conv1 = nn.Sequential(nn.Conv2d(3, ndf, 3, 1, 1),
                                   nn.InstanceNorm2d(ndf),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(ndf, ndf, 4, 2, 1),
                                   nn.InstanceNorm2d(ndf),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
                                   nn.InstanceNorm2d(ndf * 2),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.conv4 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1),
                                   nn.InstanceNorm2d(ndf * 2),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.conv5 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
                                   nn.InstanceNorm2d(ndf * 4),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.local_FeatureMapping = blocks.local_FeatureMapping(num_classes=self.local_nlabels,
                                                                n_channels=ndf * 4)  # modified : remove dilatation
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=1), y

    def gumble_softmax(self, logits):
        y, y_unorm = self.gumbel_softmax_sample(logits, temperature=1.0)
        x = torch.argmax(y, dim=1)
        x = torch.unsqueeze(x, dim=1)
        bs, _, h, w = x.size()
        input_label = self.FloatTensor(bs, self.local_nlabels, h, w).zero_()
        y_hard = input_label.scatter_(1, x.long().cuda(), 1.0)
        return (y_hard - y).detach() + y, y_unorm

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)  # out of size bs * 256 * 8 * 8
        out = self.local_FeatureMapping(
            out)  # local classifier for discriminator loss : map the features to K_2 classifiers : size K_2 * 8 * 8 label map
        # print("size of out after enc", out.size())
        logits = self.logSoftmax(out)
        label_map, label_map_unorm = self.gumble_softmax(logits)

        return label_map_unorm, label_map


class MUNITEncoder(nn.Module):
    def __init__(self,
                 local_nlabels=None,
                 nc=3,
                 ndf=64,
                 **kwargs):
        super(MUNITEncoder, self).__init__()

        # assert conditioning != 'unconditional' or nlabels == 1

        self.FloatTensor = torch.cuda.FloatTensor if True \
            else torch.FloatTensor  # is_cuda

        self.local_nlabels = local_nlabels

        self.conv1downsample = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(3, ndf, 7, 1),
                                             nn.InstanceNorm2d(ndf),
                                             nn.LeakyReLU(0.2, inplace=True))

        self.conv2downsample = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(ndf, ndf * 2, 4, 2),
                                             nn.InstanceNorm2d(ndf * 2),
                                             nn.LeakyReLU(0.2, inplace=True))

        self.conv3downsample = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(ndf * 2, ndf * 4, 4, 2),
                                             nn.InstanceNorm2d(ndf * 4),
                                             nn.LeakyReLU(0.2, inplace=True))

        self.conv4 = blocks.ResBlockMUNIT(ndf * 4)

        self.conv5 = blocks.ResBlockMUNIT(ndf * 4)

        self.local_FeatureMapping = blocks.local_FeatureMapping(num_classes=self.local_nlabels,
                                                                n_channels=ndf * 4)  # modified : remove dilatation
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=1), y

    def gumble_softmax(self, logits):
        y, y_unorm = self.gumbel_softmax_sample(logits, temperature=1.0)
        x = torch.argmax(y, dim=1)
        x = torch.unsqueeze(x, dim=1)
        bs, _, h, w = x.size()
        input_label = self.FloatTensor(bs, self.local_nlabels, h, w).zero_()
        y_hard = input_label.scatter_(1, x.long().cuda(), 1.0)
        return (y_hard - y).detach() + y, y_unorm

    def forward(self, input):
        out = self.conv1downsample(input)
        out = self.conv2downsample(out)
        out = self.conv3downsample(out)

        out = self.conv4(out)
        out = self.conv5(out)  # out of size bs * 256 * 8 * 8
        out = self.local_FeatureMapping(
            out)  # local classifier for discriminator loss : map the features to K_2 classifiers : size K_2 * 8 * 8 label map
        # print("size of out after enc", out.size())
        logits = self.logSoftmax(out)
        label_map, label_map_unorm = self.gumble_softmax(logits)

        return label_map_unorm, label_map



class BiGANDiscriminator(nn.Module):
    def __init__(self,
                 local_nlabels=None,

                 nc=3,
                 ndf=64,
                 img_size=0,
                 label_size=0,
                 **kwargs):
        super(BiGANDiscriminator, self).__init__()
        # print("USING BiGAN Discriminator", "qhead disc only with img network :", qhead_withImg)
        self.ndf = ndf
        self.local_nlabels = local_nlabels

        # self.final_res = img_size // (2 ** 3)  # if conv5 and conv6 are added

        # inference over img
        self.conv1 = nn.Sequential(nn.Conv2d(3, ndf, 3, 1, 1),
                                   nn.BatchNorm2d(ndf),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(ndf, ndf, 4, 2, 1),
                                   nn.BatchNorm2d(ndf),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
                                   nn.BatchNorm2d(ndf * 2),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.conv4 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1),
                                   nn.BatchNorm2d(ndf * 2),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.conv5 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
                                   nn.BatchNorm2d(ndf * 4),
                                   nn.LeakyReLU(0.2, inplace=True))

        # self.conv6 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1),
        #                            nn.BatchNorm2d(ndf * 4),
        #                            nn.LeakyReLU(0.2, inplace=True))

        # inference over seg
        self.conv1z = nn.Sequential(nn.Conv2d(self.local_nlabels, ndf * 4, 1, 1, padding=0, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.conv2z = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 4, 1, 1, padding=0, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))

        # Joint inference
        self.conv1xz = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 1, stride=1, bias=False),
                                     nn.LeakyReLU(0.2, inplace=True))
        self.conv2xz = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 1, stride=1, bias=False),
                                     nn.LeakyReLU(0.2, inplace=True))
        self.conv3xz = nn.Sequential(nn.Conv2d(ndf * 8, 1, 1, stride=1, bias=False), nn.LeakyReLU(0.2, inplace=True))
        # comment fc_out_joint to ouput a map of 8x8 to compute bin crossentropy on that :
        self.fc_out_joint = blocks.LinearUnconditionalLogits(8 * 8)
        # self.conv4xz = nn.Sequential(nn.Conv2d(1, 1, 4, 2,1, bias=False), nn.LeakyReLU(0.2, inplace=True))

    def inf_x(self, img):
        out = self.conv1(img)  # to try : with dropout as in initial bigan model
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        # out = self.conv6(out)
        return out

    def inf_seg(self, seg):
        out = self.conv1z(seg)
        out = self.conv2z(out)
        return out

    def inf_xseg(self, xseg):
        out = self.conv1xz(xseg)
        out = self.conv2xz(out)
        out = self.conv3xz(out)
        out = out.view(out.size(0), -1)
        out = self.fc_out_joint(out)
        return out

    def forward(self, input, seg):
        inputbis = self.inf_x(input)
        seg = self.inf_seg(seg)

        xseg = torch.cat((inputbis, seg), dim=1)
        xseg = self.inf_xseg(xseg)

        forQdisc = inputbis

        return forQdisc, xseg


class BiGANQHeadDiscriminator(nn.Module):
    def __init__(self,
                 z_dim_img,
                 z_dim_lab,
                 ndf=64,
                 size=0,
                 **kwargs):
        super(BiGANQHeadDiscriminator, self).__init__()
        print("z_dim in qhead disc :", z_dim_lab, z_dim_img)
        self.ndf = ndf
        input_nc = 3

        # self.conv1_img = nn.Conv2d(ndf *4, 256, 8, bias=False)
        #
        # self.bn1_img = nn.BatchNorm2d(256)
        #
        # self.conv_mu_img = nn.Conv2d(256, z_dim_img, 1)
        # self.conv_var_img = nn.Conv2d(256, z_dim_img, 1)

        self.conv1_lab = nn.Conv2d(ndf * 4, 256, 8, bias=False)

        self.bn1_lab = nn.BatchNorm2d(256)

        self.conv_mu_lab = nn.Conv2d(256, z_dim_lab, 1)
        self.conv_var_lab = nn.Conv2d(256, z_dim_lab, 1)

    def forward(self, x):
        # x_img = F.leaky_relu(self.bn1_img(self.conv1_img(x)), 0.1, inplace=True)
        # mu_img = self.conv_mu_img(x_img).squeeze()
        # var_img = torch.exp(self.conv_var_img(x_img).squeeze())
        # print("size of mu and var", mu.size(), var.size(), torch.min(mu), torch.max(mu))

        x_lab = F.leaky_relu(self.bn1_lab(self.conv1_lab(x)), 0.1, inplace=True)
        mu_lab = self.conv_mu_lab(x_lab).squeeze()
        var_lab = torch.exp(self.conv_var_lab(x_lab).squeeze())
        return mu_lab, var_lab, 0, 0


if __name__ == '__main__':
    z = torch.zeros((1, 128))
    g = Decoder()
    x = torch.zeros((1, 3, 32, 32))
    d = Encoder()

    g(z)
    d(g(z))
    d(x)
