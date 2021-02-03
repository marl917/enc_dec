import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed

from gan_training.models import blocks
from gan_training.models.blocks import ResnetBlock
from torch.nn.utils.spectral_norm import spectral_norm

class Decoder(nn.Module):
    def __init__(self,
                 nlabels,
                 local_nlabels=0,
                 z_dim=128,
                 nc=3,
                 ngf=64,
                 embed_dim=256,
                 deterministicOnSeg=False,
                 size=0,
                 **kwargs):
        super(Decoder, self).__init__()

        self.sw = size // (2 ** 6)
        self.sh = self.sw  # assumption of square images

        self.deterministicOnSeg = deterministicOnSeg
        print("Decoder only depends on Segmentation : ", self.deterministicOnSeg)

        self.get_latent = blocks.Identity()
        if self.deterministicOnSeg:
            self.fc = nn.Conv2d(local_nlabels, ngf * 16, 3, padding=1)
        else:
            self.fc = nn.Linear(z_dim, self.sh * self.sw * ngf * 16)

        self.up = nn.Upsample(scale_factor=2)

        self.head_0 = blocks.SPADEResnetBlock(16 * ngf, 16 * ngf, local_nlabels)

        self.G_middle_0 = blocks.SPADEResnetBlock(16 * ngf, 16 * ngf,local_nlabels)
        self.G_middle_1 = blocks.SPADEResnetBlock(16 * ngf, 16 * ngf, local_nlabels)

        self.up_0 = blocks.SPADEResnetBlock(16 * ngf, 8 * ngf, local_nlabels)
        self.up_1 = blocks.SPADEResnetBlock(8 * ngf, 4 * ngf, local_nlabels)
        self.up_2 = blocks.SPADEResnetBlock(4 * ngf, 2 * ngf, local_nlabels)
        self.up_3 = blocks.SPADEResnetBlock(2 * ngf, 1 * ngf, local_nlabels)
        self.conv_img = nn.Conv2d(ngf, 3, 3, padding=1)

    def forward(self, seg, input=None):  #input=z
        #y = y.clamp(None, self.global_nlabels - 1)

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
        x = self.G_middle_1(x, seg)
        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        return x

class Encoder(nn.Module):
    def __init__(self,
                 nlabels,
                 local_nlabels,
                 img_size,
                 label_size =0,
                 nfilter=64,
                 features='penultimate',
                 **kwargs):
        super(Encoder, self).__init__()
        # s0 = self.s0 = img_size // 32
        print("img size in encoder :", img_size)
        nf = self.nf = nfilter
        self.nlabels = nlabels
        self.local_nlabels = local_nlabels
        self.img_size =img_size
        self.label_size = label_size


        bn = blocks.BatchNorm2d

        self.conv_img = nn.Conv2d(3, 1 * nf, 3, padding=1)

        self.resnet_0_0 = ResnetBlock(1 * nf, 1 * nf, bn)
        self.resnet_0_1 = ResnetBlock(1 * nf, 2 * nf, bn)

        self.resnet_1_0 = ResnetBlock(2 * nf, 2 * nf, bn)
        self.resnet_1_1 = ResnetBlock(2 * nf, 4 * nf, bn)

        self.resnet_2_0 = ResnetBlock(4 * nf, 4 * nf, bn)
        self.resnet_2_1 = ResnetBlock(4 * nf, 8 * nf, bn)

        self.resnet_3_0 = ResnetBlock(8 * nf, 8 * nf, bn)
        self.resnet_3_1 = ResnetBlock(8 * nf, 16 * nf, bn)

        self.resnet_4_0 = ResnetBlock(16 * nf, 16 * nf, bn)
        self.resnet_4_1 = ResnetBlock(16 * nf, 16 * nf, bn)
        #
        # self.resnet_5_0 = ResnetBlock(16 * nf, 16 * nf, bn, nlabels)
        # self.resnet_5_1 = ResnetBlock(16 * nf, 16 * nf, bn, nlabels)

        # self.local_FeatureMapping = blocks.Classifier_Module(dilation_series = [3,6,9,12], padding_series = [3,6,9,12], num_classes=self.local_nlabels,
        #                                                         n_input_channels=nf * 16)  # modified : remove dilatation

        self.local_FeatureMapping = blocks.local_FeatureMapping(num_classes=self.local_nlabels, n_channels=16*nf)

        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.FloatTensor = torch.cuda.FloatTensor

    def sample_gumbel(self, shape, logits, eps=1e-20):
        U = logits.new(shape).uniform_(0, 1)  # = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits)
        return F.softmax(y / temperature, dim=1), y

    def gumble_softmax(self, logits):
        y, y_unorm = self.gumbel_softmax_sample(logits, temperature=1.0)
        x = torch.argmax(y, dim=1)
        x = torch.unsqueeze(x, dim=1)
        bs, _, h, w = x.size()
        # print("local n labels  : ", self.local_nlabels, torch.min(x), torch.max(x))
        input_label = self.FloatTensor(bs, self.local_nlabels, h, w).zero_()
        y_hard = input_label.scatter_(1, x.long().cuda(), 1.0)
        return (y_hard - y).detach() + y, y_unorm

    def forward(self, x):
        out = self.conv_img(x)

        out = self.resnet_0_0(out)
        out = self.resnet_0_1(out)
        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_1_0(out)
        out = self.resnet_1_1(out)

        out = self.resnet_2_0(out)
        out = self.resnet_2_1(out)
        out = F.avg_pool2d(out, 3, stride=2, padding=1)
        out = self.resnet_3_0(out)
        out = self.resnet_3_1(out)

        #if there is 8x8 label map:
        if self.img_size // self.label_size == 8:
            out = F.avg_pool2d(out, 3, stride=2, padding=1)

        out = self.resnet_4_0(out)
        out = self.resnet_4_1(out)
        out = actvn(out)
        out = self.local_FeatureMapping(out)

        logits = self.logSoftmax(out)
        label_map, label_map_unorm = self.gumble_softmax(logits)

        return label_map_unorm, label_map


# class Encoder(nn.Module):
#     def __init__(self,
#                  nlabels,
#                  local_nlabels=None,
#                  nc=3,
#                  ndf=64,
#                  pack_size=1,
#                  features='penultimate',
#                  **kwargs):
#
#         super(Encoder, self).__init__()
#
#         #assert conditioning != 'unconditional' or nlabels == 1
#
#         self.FloatTensor = torch.cuda.FloatTensor if True \
#             else torch.FloatTensor  #is_cuda
#
#         self.local_nlabels = local_nlabels
#
#         self.conv1 = nn.Sequential(nn.Conv2d(nc * pack_size, ndf, 3, 1, 1), nn.LeakyReLU(0.1))
#         self.conv2 = nn.Sequential(nn.Conv2d(ndf, ndf, 4, 2, 1), nn.LeakyReLU(0.1))
#         self.conv3 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 3, 1, 1), nn.LeakyReLU(0.1))
#         self.conv4 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1), nn.LeakyReLU(0.1))
#         self.conv5 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1), nn.LeakyReLU(0.1))
#
#
#         self.local_FeatureMapping = blocks.local_FeatureMapping(num_classes=self.local_nlabels,
#                                                                 n_channels=ndf * 4)  # modified : remove dilatation
#         self.logSoftmax = nn.LogSoftmax(dim=1)
#
#
#         self.features = features
#         self.pack_size = pack_size
#         print(f'Getting features from {self.features}')
#
#     def sample_gumbel(self, shape, eps=1e-20):
#         U = torch.rand(shape).cuda()
#         return -Variable(torch.log(-torch.log(U + eps) + eps))
#
#     def gumbel_softmax_sample(self, logits, temperature):
#         y = logits + self.sample_gumbel(logits.size())
#         return F.softmax(y / temperature, dim=1), y
#
#     def gumble_softmax(self, logits):
#         y, y_unorm = self.gumbel_softmax_sample(logits, temperature=1.0)
#         x = torch.argmax(y, dim=1)
#         x = torch.unsqueeze(x, dim=1)
#         bs, _, h, w = x.size()
#         input_label = self.FloatTensor(bs, self.local_nlabels, h, w).zero_()
#         y_hard = input_label.scatter_(1, x.long().cuda(), 1.0)
#         return (y_hard - y).detach() + y, y_unorm
#
#
#     def forward(self, input):
#         out = self.conv1(input)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.conv4(out)
#         out = self.conv5(out)  #out of size bs * 256 * 8 * 8
#         out = self.local_FeatureMapping(out)#local classifier for discriminator loss : map the features to K_2 classifiers : size K_2 * 8 * 8 label map
#
#
#         logits = self.logSoftmax(out)
#         label_map, label_map_unorm = self.gumble_softmax(logits)
#
#         return label_map_unorm, label_map

class LabelGenerator(nn.Module):
    def __init__(self,
                 z_dim,
                 nlabels,
                 label_size,
                 local_nlabels=0,
                 conditioning=None,
                 nfilter=64,
                 **kwargs):
        super(LabelGenerator, self).__init__()

        print(label_size)
        nf = self.nf = nfilter
        self.nlabels = nlabels
        self.z_dim = z_dim
        self.local_nlabels = local_nlabels



        #either use conditional batch norm, or use no batch norm
        bn = blocks.BatchNorm2d
        # bn = blocks.BatchNorm2d
        print("INIT LABEL GENERATOR")
        self.modified = False
        if not self.modified:
            s0 = self.s0 = label_size // 4
            self.fc = nn.Linear(z_dim, 8 * nf * s0 * s0,bn)
            self.resnet_0_0 = ResnetBlock(8 * nf, 8 * nf,bn)
            self.resnet_1_0 = ResnetBlock(8 * nf, 4 * nf,bn)

            self.resnet_2_0 = ResnetBlock(4 * nf, 4 * nf,bn)
            self.resnet_4_0 = ResnetBlock(4 * nf, 2 * nf,bn)

            self.resnet_5_0 = ResnetBlock(2 * nf, 1 * nf,bn)

        else:
            s0 = self.s0 = label_size // 4
            self.fc = nn.Linear(z_dim, 16 * nf * s0 * s0)
            self.resnet_0_0 = ResnetBlock(16 * nf, 16 * nf, bn = False)
            self.resnet_0_1 = ResnetBlock(16 * nf, 16 * nf, bn = False)

            self.resnet_1_0 = ResnetBlock(16 * nf, 8 * nf, bn = False)
            self.resnet_1_1 = ResnetBlock(8 * nf, 8 * nf, bn = False)

            self.resnet_2_0 = ResnetBlock(8 * nf, 4 * nf, bn = False)
            self.resnet_2_1 = ResnetBlock(4 * nf, 4 * nf, bn = False)

            self.resnet_3_0 = ResnetBlock(4 * nf, 2* nf, bn = False)
            self.resnet_3_1 = ResnetBlock(2 * nf, 2 * nf, bn = False)

            self.resnet_4_0 = ResnetBlock(2 * nf, 1 * nf, bn = False)
            self.resnet_4_1 = ResnetBlock(1 * nf, 1 * nf, bn = False)



        self.conv_img = nn.Sequential(nn.Conv2d(nf, self.local_nlabels, 3, padding = 1), nn.LogSoftmax(dim=1))

        self.FloatTensor = torch.cuda.FloatTensor

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=1), y

    def gumble_softmax(self, logits):
        y, y_unorm= self.gumbel_softmax_sample(logits, temperature=1.0)
        x = torch.argmax(y, dim=1)
        x = torch.unsqueeze(x, dim=1)
        bs, _, h, w = x.size()
        input_label = self.FloatTensor(bs, self.local_nlabels, h, w).zero_()
        y_hard = input_label.scatter_(1, x.long().cuda(), 1.0)
        return (y_hard - y).detach() + y, y_unorm

    def forward(self, z):
        out = self.fc(z)
        if not self.modified:
            out = out.view(z.size(0), 8 * self.nf, self.s0, self.s0)
            out = self.resnet_0_0(out)
            out = self.resnet_1_0(out)
            out = F.interpolate(out, scale_factor=2)
            out = self.resnet_2_0(out)
            out = self.resnet_4_0(out)
            out = F.interpolate(out, scale_factor=2)
            out = actvn(self.resnet_5_0(out))

        else:
            out = out.view(z.size(0), 8 * self.nf, self.s0, self.s0)
            out = self.resnet_0_0(out)
            out = self.resnet_0_1(out)
            out = F.interpolate(out, scale_factor=2)
            out = self.resnet_1_0(out)
            out = self.resnet_1_2(out)
            out = F.interpolate(out, scale_factor=2)
            out = self.resnet_2_0(out)
            out = self.resnet_2_1(out)
            out = F.interpolate(out, scale_factor=2)
            out = self.resnet_3_0(out)
            out = self.resnet_3_1(out)

            out = self.resnet_4_0(out)
            out = self.resnet_4_1(out)
            out = actvn(self.resnet_5_0(out))
        logits = self.conv_img(out)
        label_map, y_unorm = self.gumble_softmax(logits)

        return y_unorm, label_map

class BiGANDiscriminator(nn.Module):
    def __init__(self,
                 nlabels,
                 local_nlabels,
                 img_size,
                 label_size,
                 nfilter=64,
                 **kwargs):
        super().__init__()
        s0 = self.s0 = label_size
        print("value of s0 in BiGAN DISC : ", s0)
        nf = self.nf = nfilter
        self.nlabels = nlabels
        self.local_nlabels = local_nlabels
        self.img_size = img_size
        self.label_size =label_size

        bn = blocks.BatchNorm2d

        #inference over x
        # self.conv_img = nn.Conv2d(3, 1 * nf, 3, padding=1)
        #
        # self.resnet_0_0_img = ResnetBlock(1 * nf, 1 * nf, bn)
        # self.resnet_0_1_img = ResnetBlock(1 * nf, 2 * nf, bn)
        #
        # self.resnet_1_0_img = ResnetBlock(2 * nf, 2 * nf, bn)
        # self.resnet_1_1_img = ResnetBlock(2 * nf, 4 * nf, bn)
        #
        # self.resnet_2_0_img = ResnetBlock(4 * nf, 4 * nf, bn)
        # self.resnet_2_1_img = ResnetBlock(4 * nf, 8 * nf, bn)

        ndf = nf
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
        self.conv6 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1),
                                   nn.BatchNorm2d(ndf * 8),
                                   nn.LeakyReLU(0.2, inplace=True))
        #inference over seg
        self.conv1z = nn.Sequential(nn.Conv2d(self.local_nlabels, ndf * 2, 1, 1, padding=0, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.conv2z = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 1, 1, padding=0, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))
        self.conv3z = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, 1, 1, padding=0, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True))

        # self.resnet_3_0_seg = ResnetBlock(self.local_nlabels, 4 * nf, bn)
        # self.resnet_3_1_seg = ResnetBlock(4 * nf, 8 * nf, bn)
        # self.resnet_3_2_seg = ResnetBlock(2 * nf, 4 * nf, bn)
        # self.resnet_3_3_seg = ResnetBlock(4 * nf, 8 * nf, bn)

        #joint inference
        # self.resnet_4_0 = ResnetBlock(16 * nf, 16 * nf, bn=blocks.Identity, use_shortcut=True, is_bias=False)
        # self.resnet_4_1 = ResnetBlock(16 * nf, 16 * nf, bn=blocks.Identity, use_shortcut=True, is_bias=False)
        # self.conv1xz = nn.Sequential(nn.Conv2d(nf * 16, nf * 16, 1, stride=1, bias=False),
        #                              nn.LeakyReLU(0.2, inplace=True))
        # self.conv2xz = nn.Sequential(nn.Conv2d(nf * 16, 1, 1, stride=1, bias=False), nn.LeakyReLU(0.2, inplace=True))
        # self.fc_out_joint = blocks.LinearUnconditionalLogits(s0 * s0)

        self.conv1xz = nn.Sequential(nn.Conv2d(ndf * 16, ndf * 16, 1, stride=1, bias=False),
                                     nn.LeakyReLU(0.2, inplace=True))
        self.conv2xz = nn.Sequential(nn.Conv2d(ndf * 16, ndf * 16, 1, stride=1, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.conv3xz = nn.Sequential(nn.Conv2d(ndf * 16, 1, 1, stride=1, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.fc_out_joint = blocks.LinearUnconditionalLogits(s0*s0)


    def inf_x(self,img):
        out = self.conv1(img)  # to try : with dropout as in initial bigan model
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        # out = self.conv_img(img)
        # out = self.resnet_0_0_img(out)
        # out = self.resnet_0_1_img(out)
        # out = F.avg_pool2d(out, 3, stride=2, padding=1)
        # out = self.resnet_1_0_img(out)
        # out = self.resnet_1_1_img(out)
        # out = F.avg_pool2d(out, 3, stride=2, padding=1)
        # out = self.resnet_2_0_img(out)
        # if self.img_size // self.label_size ==8:
        #     out = F.avg_pool2d(out, 3, stride=2, padding=1)
        #
        # out = self.resnet_2_1_img(out)
        return out

    def inf_seg(self, seg):
        # out = self.resnet_3_0_seg(seg)
        # out = self.resnet_3_1_seg(out)
        # out = self.resnet_3_2_seg(out)
        # out = self.resnet_3_3_seg(out)

        out = self.conv1z(seg)
        out = self.conv2z(out)
        out = self.conv3z(out)
        return out

    def inf_xseg(self,xseg):
        # out = self.resnet_4_0(xseg)
        # out = self.resnet_4_1(out)
        # # out = F.avg_pool2d(out, 3, stride=2, padding=1)
        #
        # # out = self.conv1xz(out)
        # out = self.conv2xz(out)
        #
        # out = out.view(out.size(0),-1)
        # out = self.fc_out_joint(out)

        out = self.conv1xz(xseg)
        out = self.conv2xz(out)
        out = self.conv3xz(out)
        out = out.view(out.size(0), -1)
        out = self.fc_out_joint(out)
        return out

    def forward(self, input, seg):
        # print("dim of seg and input  : ", seg.size(), input.size())
        inputbis = self.inf_x(input)
        seg = self.inf_seg(seg)
        # print(seg.size(), inputbis.size())
        xseg = torch.cat((inputbis, seg), dim=1)
        xseg = self.inf_xseg(xseg)

        forQdisc = inputbis

        return forQdisc, xseg

# class BiGANDiscriminator(nn.Module):
#     def __init__(self,
#                  nlabels,
#                  local_nlabels=None,
#                  features='penultimate',
#                  pack_size=1,
#                  qhead_withImg=False,
#                  img_path=False,
#                  nc=3,
#                  ndf=64,
#                  img_size=0,
#                  label_size=0,
#                  **kwargs):
#         super(BiGANDiscriminator, self).__init__()
#         # print("USING BiGAN Discriminator", "qhead disc only with img network :", qhead_withImg)
#         self.ndf = ndf
#         self.nlabels = nlabels
#         self.local_nlabels = local_nlabels
#
#         self.qhead_withImg = qhead_withImg
#         self.img_path = img_path
#
#
#         # self.final_res = img_size // (2 ** 3)  # if conv5 and conv6 are added
#
#         #inference over img
#         self.conv1 = nn.Sequential(nn.Conv2d(3 * pack_size, ndf, 3, 1, 1),
#                                    nn.BatchNorm2d(ndf),
#                                    nn.LeakyReLU(0.2, inplace=True))
#
#         self.conv2 = nn.Sequential(nn.Conv2d(ndf, ndf, 4, 2, 1),
#                                    nn.BatchNorm2d(ndf),
#                                    nn.LeakyReLU(0.2, inplace=True))
#
#         self.conv3 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
#                                    nn.BatchNorm2d(ndf * 2),
#                                    nn.LeakyReLU(0.2, inplace=True))
#
#         self.conv4 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1),
#                                    nn.BatchNorm2d(ndf * 2),
#                                    nn.LeakyReLU(0.2, inplace=True))
#
#         self.conv5 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
#                                    nn.BatchNorm2d(ndf * 4),
#                                    nn.LeakyReLU(0.2, inplace=True))
#
#
#         if self.img_path:
#             self.conv7 = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1),
#                                        nn.BatchNorm2d(ndf * 8),
#                                        nn.LeakyReLU(0.2, inplace=True))
#             self.fc_out_img = blocks.LinearUnconditionalLogits(ndf * 8*4*4) #nn.Sequential(nn.Conv2d(ndf * 8, 1, 1, stride=1, bias=False), nn.LeakyReLU(0.2, inplace=True))
#
#
#         #inference over seg
#         self.conv1z =nn.Sequential(nn.Conv2d(self.local_nlabels, ndf * 4, 1, 1, padding = 0, bias = False), nn.LeakyReLU(0.2, inplace=True))
#         self.conv2z = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 4, 1, 1, padding=0, bias=False), nn.LeakyReLU(0.2, inplace=True))
#
#         # Joint inference
#         self.conv1xz = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 1, stride=1, bias=False), nn.LeakyReLU(0.2, inplace=True))
#         self.conv2xz = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 1, stride=1, bias=False), nn.LeakyReLU(0.2, inplace=True))
#         self.conv3xz = nn.Sequential(nn.Conv2d(ndf * 8, 1, 1, stride=1, bias=False), nn.LeakyReLU(0.2, inplace=True))
#         #comment fc_out_joint to ouput a map of 8x8 to compute bin crossentropy on that :
#         self.fc_out_joint = blocks.LinearUnconditionalLogits(8*8)
#         # self.conv4xz = nn.Sequential(nn.Conv2d(1, 1, 4, 2,1, bias=False), nn.LeakyReLU(0.2, inplace=True))
#
#     def inf_x(self, img):
#         out = self.conv1(img)   #to try : with dropout as in initial bigan model
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.conv4(out)
#         out = self.conv5(out)
#         return out
#
#     def inf_seg(self,seg):
#         out = self.conv1z(seg)
#         out = self.conv2z(out)
#         return out
#
#     def inf_xseg(self,xseg):
#         out = self.conv1xz(xseg)
#         forQdisc = self.conv2xz(out)
#         out = self.conv3xz(forQdisc)
#         out = out.view(out.size(0), -1)
#         out = self.fc_out_joint(out)
#         # out = self.conv4xz(out)
#         return forQdisc, out
#
#
#     def forward(self, input, seg):
#         inputbis = self.inf_x(input)
#         seg = self.inf_seg(seg)
#         # print("in disc : ", seg.size(), inputbis.size())
#         xseg = torch.cat((inputbis, seg), dim = 1)
#         forQdisc, xseg = self.inf_xseg(xseg)
#
#         if not self.qhead_withImg:
#             forQdisc = forQdisc
#         else:
#             forQdisc = inputbis
#
#         if not self.img_path:
#             return forQdisc, xseg
#         else:
#             input = self.conv6(inputbis)
#             input = self.conv7(input)
#             input = input.view(input.size(0),-1)
#             input = self.fc_out_img(input)
#             # print("img path", input.size(), xseg.size())
#             return forQdisc, xseg, input



class BiGANQHeadDiscriminator(nn.Module):
    def __init__(self,
                 nlabels,
                 local_nlabels=None,
                 size=None,
                 z_dim=1,
                 qhead_variant=False,
                 ndf=64,
                 **kwargs):
        super(BiGANQHeadDiscriminator, self).__init__()
        print("z_dim in qhead disc :", z_dim)
        self.ndf = ndf
        self.nlabels = nlabels
        self.local_nlabels = local_nlabels
        self.qhead_variant = qhead_variant

        # self.conv1 = nn.Conv2d(ndf *16, 512, 8, bias=False)
        #
        # self.bn1 = nn.BatchNorm2d(512)
        if not qhead_variant:
            # bn=blocks.Identity
            # self.resnet_0_0 = ResnetBlock(512, 256,bn = bn)
            # self.resnet_0_1 = ResnetBlock(256, 128,bn = bn)
            # self.conv2 = nn.Conv2d(128, 128, size, bias = False)
            # self.bn2 = nn.BatchNorm2d(128)

            self.conv2 = nn.Conv2d(512, 256, 3,1,1)
            self.bn2 = nn.BatchNorm2d(256)
            self.conv3 = nn.Conv2d(256, 128, 4,2,1)
            self.bn3 = nn.BatchNorm2d(128)
            self.conv4 = nn.Conv2d(128, 128,size//2 , bias = False )
            self.bn4 = nn.BatchNorm2d(128)


            self.conv_mu = nn.Conv2d(128, z_dim, 1)
            self.conv_var = nn.Conv2d(128, z_dim, 1)
            print("2nd Qhead discriminator with variant")
        else:
            self.conv2 = nn.Conv2d(512, 256, 3, 1, 1)
            self.bn2 = nn.BatchNorm2d(256)
            self.conv3 = nn.Conv2d(256, 256, size, bias=False)
            self.bn3 = nn.BatchNorm2d(256)
            print("Qhead discriminator with variant")
            self.conv_mu = nn.Conv2d(256, z_dim, 1)
            self.conv_var = nn.Conv2d(256, z_dim, 1)



    def forward(self,x):
        # x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        if self.qhead_variant:
            x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
            x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
        else:
            # x = self.resnet_0_0(x)
            # x = self.resnet_0_1(x)
            x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
            x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
            x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1, inplace=True)
        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())
        return mu, var

def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out