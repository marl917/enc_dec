import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from gan_training.models import blocks
from torch.autograd import Variable





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
