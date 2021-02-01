import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from gan_training.models import blocks
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self,
                 nlabels,
                 conditioning,
                 z_dim=128,
                 local_nlabels=0,
                 ngf=64,
                 embed_dim=256,
                 label_size=0,
                 **kwargs):
        super(Generator, self).__init__()
        size = 8
        assert conditioning != 'unconditional' or nlabels == 1
        self.sw = size // (2 ** 2)
        self.sh = self.sw

        nc= local_nlabels

        if conditioning == 'embedding':
            self.get_latent = blocks.LatentEmbeddingConcat(nlabels, embed_dim)
            self.fc = nn.Linear(z_dim + embed_dim, self.sh*self.sw * ngf * 8)
        elif conditioning == 'unconditional':
            self.get_latent = blocks.Identity()
            self.fc = nn.Linear(z_dim, self.sh*self.sw * ngf * 8)
        else:
            raise NotImplementedError(
                f"{conditioning} not implemented for generator")

        bn = blocks.BatchNorm2d

        self.nlabels = nlabels
        self.local_nlabels = local_nlabels

        self.conv1 = nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1)
        self.bn1 = bn(ngf * 8, nlabels)

        self.conv1bis = nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 1, 1)
        self.bn1bis = bn(ngf * 4, nlabels)

        self.conv2 = nn.ConvTranspose2d(ngf * 4, ngf * 4, 4, 2, 1)
        self.bn2 = bn(ngf * 4, nlabels)

        self.conv2bis = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 1, 1)
        self.bn2bis = bn(ngf * 2, nlabels)


        self.conv_out = nn.Sequential(nn.Conv2d(ngf*2, nc, 3, 1, 1), nn.LogSoftmax(dim=1))

        self.FloatTensor = torch.cuda.FloatTensor if True \
            else torch.FloatTensor  # is_cuda

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

    def forward(self, input, y=None):
        out = self.get_latent(input, y)
        out = self.fc(out)

        out = out.view(out.size(0), -1, self.sh, self.sw)
        out = F.relu(self.bn1(self.conv1(out), y))
        out = F.relu(self.bn1bis(self.conv1bis(out), y))
        out = F.relu(self.bn2(self.conv2(out), y))
        out = F.relu(self.bn2bis(self.conv2bis(out), y))
        logits = self.conv_out(out)
        label_map, y_unorm = self.gumble_softmax(logits)
        return y_unorm, label_map


class LabDiscriminator(nn.Module):
    def __init__(self,
                 nlabels,
                 conditioning,
                 local_nlabels=None,
                 features='penultimate',
                 pack_size=1,
                 nc=3,
                 ndf=64,
                 size=0,
                 **kwargs):
        super(LabDiscriminator, self).__init__()

        assert conditioning != 'unconditional' or nlabels == 1
        print("Label discriminator is ", conditioning)

        self.ndf = ndf
        self.nlabels = nlabels
        self.local_nlabels=local_nlabels


        input_nc = self.local_nlabels
        self.final_res = 2

        # self.conv1 = nn.Sequential(nn.Conv2d(nc * pack_size, ndf, 4, 2, 1),
        #                            nn.BatchNorm2d(ndf),
        #                            nn.LeakyReLU(0.2, inplace=True))

        self.conv1 = nn.Sequential(nn.Conv2d(input_nc * pack_size, ndf, 3, 1, 1),nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(ndf, ndf , 4, 2, 1),nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(ndf, ndf *2, 3, 1, 1),nn.LeakyReLU(0.2, inplace=True))
        self.conv4 = nn.Conv2d(ndf*2, ndf * 2, 4, 2, 1)

        if conditioning == 'mask':
            self.fc_out = blocks.LinearConditionalMaskLogits(ndf * 2 * self.final_res * self.final_res, nlabels)
        elif conditioning == 'unconditional':
            self.fc_out = blocks.LinearUnconditionalLogits(ndf * 2 * self.final_res * self.final_res)
        else:
            raise NotImplementedError(
                f"{conditioning} not implemented for discriminator")

        self.pack_size = pack_size
        self.features = features
        print(f'Getting features from {self.features}')

    def forward(self, input):

        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        out = out.view(out.size(0), -1)

        result = self.fc_out(out)

        assert (len(result.shape) == 1)
        return result


class ImgDiscriminator(nn.Module):
    def __init__(self,
                 nlabels,
                 conditioning,
                 local_nlabels=None,
                 img_gen=False,
                 features='penultimate',
                 pack_size=1,
                 ndf=64,
                 size=0,
                 label_size =0,
                 **kwargs):
        super(ImgDiscriminator, self).__init__()

        assert conditioning != 'unconditional' or nlabels == 1
        print("Img discriminator is ", conditioning, ' and it is for imggen : ', img_gen)

        self.ndf = ndf
        self.nlabels = nlabels
        self.local_nlabels=local_nlabels

        self.label_size = label_size


        input_nc = 3
        self.final_res = size // (2 ** 3)  # if conv5 and conv6 are added



        # self.conv1 = nn.Sequential(nn.Conv2d(nc * pack_size, ndf, 4, 2, 1),
        #                            nn.BatchNorm2d(ndf),
        #                            nn.LeakyReLU(0.2, inplace=True))

        self.conv1 = nn.Sequential(nn.Conv2d(input_nc * pack_size, ndf, 3, 1, 1),
                                   nn.BatchNorm2d(ndf),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(ndf, ndf , 4, 2, 1),
                                   nn.BatchNorm2d(ndf),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.conv3 = nn.Sequential(nn.Conv2d(ndf, ndf *2, 3, 1, 1),
                                   nn.BatchNorm2d(ndf * 2),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.conv4 = nn.Sequential(nn.Conv2d(ndf*2, ndf * 2, 4, 2, 1),
                                   nn.BatchNorm2d(ndf * 2),
                                   nn.LeakyReLU(0.2, inplace=True))


        self.conv5 = nn.Sequential(nn.Conv2d(ndf*2, ndf * 4, 3, 1, 1),
                                   nn.BatchNorm2d(ndf * 4),
                                   nn.LeakyReLU(0.2, inplace=True))

        self.conv6 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1),
                                   nn.BatchNorm2d(ndf * 4),
                                   nn.LeakyReLU(0.2, inplace=True))


        if conditioning == 'mask':
            self.fc_out = blocks.LinearConditionalMaskLogits(ndf * 4 * self.final_res * self.final_res,
                                                                 nlabels)  # to modify if conv5 and conv6 aren't added
        elif conditioning == 'unconditional':
            # if label_size == 16:
            #     self.local_FeatureMapping = blocks.local_FeatureMapping(local_nlabels, ndf * 2) #after conv5
            # elif label_size in [4,8]:
            #     self.local_FeatureMapping = blocks.local_FeatureMapping(local_nlabels, ndf * 4)  # after conv5

            self.fc_out = blocks.LinearUnconditionalLogits(ndf * 4 * self.final_res * self.final_res)
        else:
            raise NotImplementedError(
                f"{conditioning} not implemented for discriminator")

        self.pack_size = pack_size
        self.features = features
        print(f'Getting features from {self.features}')

    def select_mapDisc(self,out, seg):

        # print("in select map Disc : ", out.size(), seg.size())
        seg = torch.argmax(seg, dim=1)   #convert from one hot encoding to one channel tensor
        # print(seg.size(), out.size())
        seg = torch.unsqueeze(seg, dim=1)
        return torch.gather(out,1,seg)

    def forward(self, input, seg=None):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)

        out = self.conv4(out)
        out = self.conv5(out)

        # ftmap = self.local_FeatureMapping(out)
        # scoreLabelMap = self.select_mapDisc(ftmap, seg)

        out = self.conv6(out)
        out = out.view(out.size(0), -1)

        result = self.fc_out(out)

        assert (len(result.shape) == 1)

        return result


if __name__ == '__main__':
    z = torch.zeros((1, 128))
    g = Generator()
    x = torch.zeros((1, 3, 32, 32))
    d = Discriminator()

    g(z)
    d(g(z))
    d(x)
