import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from gan_training.models import blocks
from torch.autograd import Variable



class Decoder(nn.Module):
    def __init__(self,
                 nlabels,
                 local_nlabels=0,
                 z_dim=128,
                 nc=3,
                 ngf=64,
                 embed_dim=256,
                 size=0,
                 deterministicOnSeg = False,
                 **kwargs):
        super(Decoder, self).__init__()
        self.sw = size // (2 ** 4)
        self.sh = self.sw  # assumption of square images

        self.deterministicOnSeg = deterministicOnSeg
        print("Decoder only depends on Segmentation : ", self.deterministicOnSeg)

        self.get_latent = blocks.Identity()
        if self.deterministicOnSeg:
            self.fc = nn.Conv2d(local_nlabels,ngf * 8, 3, padding=1)
        else:
            self.fc = nn.Linear(z_dim, self.sh * self.sw * ngf * 8)

        self.up = nn.Upsample(scale_factor=2)
        self.head_0 = blocks.SPADEResnetBlock(8 * ngf, 8 * ngf, local_nlabels)
        self.G_middle_0 = blocks.SPADEResnetBlock(8 * ngf, 8 * ngf,local_nlabels)

        self.up_1 = blocks.SPADEResnetBlock(8 * ngf, 4 * ngf, local_nlabels)
        self.up_2 = blocks.SPADEResnetBlock(4 * ngf, 2 * ngf, local_nlabels)
        self.up_3 = blocks.SPADEResnetBlock(2 * ngf, 1 * ngf, local_nlabels)
        self.conv_img = nn.Conv2d(ngf, 3, 3, padding=1)


    def forward(self,  seg, input=None, y=None):  #input=z
        #alternative : downsample label map
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

class Encoder(nn.Module):
    def __init__(self,
                 nlabels,
                 local_nlabels=None,
                 nc=3,
                 ndf=64,
                 pack_size=1,
                 features='penultimate',
                 **kwargs):

        super(Encoder, self).__init__()

        #assert conditioning != 'unconditional' or nlabels == 1

        self.FloatTensor = torch.cuda.FloatTensor if True \
            else torch.FloatTensor  #is_cuda

        self.local_nlabels = local_nlabels


        self.conv1 = nn.Sequential(nn.Conv2d(nc * pack_size, ndf, 3, 1, 1), nn.LeakyReLU(0.1))
        self.conv2 = nn.Sequential(nn.Conv2d(ndf, ndf, 4, 2, 1), nn.LeakyReLU(0.1))
        self.conv3 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 3, 1, 1), nn.LeakyReLU(0.1))
        self.conv4 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 2, 4, 2, 1), nn.LeakyReLU(0.1))
        self.conv5 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1), nn.LeakyReLU(0.1))


        self.local_FeatureMapping = blocks.local_FeatureMapping(num_classes=self.local_nlabels,
                                                                n_channels=ndf * 4)  # modified : remove dilatation
        self.logSoftmax = nn.LogSoftmax(dim=1)


        self.features = features
        self.pack_size = pack_size
        print(f'Getting features from {self.features}')

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
        out = self.conv5(out)  #out of size bs * 256 * 8 * 8
        out = self.local_FeatureMapping(out)#local classifier for discriminator loss : map the features to K_2 classifiers : size K_2 * 8 * 8 label map


        logits = self.logSoftmax(out)
        label_map, label_map_unorm = self.gumble_softmax(logits)

        return label_map_unorm, label_map

class BiGANDiscriminator(nn.Module):
    def __init__(self,
                 nlabels,
                 local_nlabels=None,
                 features='penultimate',
                 pack_size=1,
                 qhead_withImg=False,
                 img_path=False,
                 nc=3,
                 ndf=64,
                 img_size=0,
                 label_size=0,
                 **kwargs):
        super(BiGANDiscriminator, self).__init__()
        # print("USING BiGAN Discriminator", "qhead disc only with img network :", qhead_withImg)
        self.ndf = ndf
        self.nlabels = nlabels
        self.local_nlabels = local_nlabels

        self.qhead_withImg = qhead_withImg
        self.img_path = img_path


        # self.final_res = img_size // (2 ** 3)  # if conv5 and conv6 are added

        #inference over img
        self.conv1 = nn.Sequential(nn.Conv2d(3 * pack_size, ndf, 3, 1, 1),
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
        if self.img_path:
            self.conv7 = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1),
                                       nn.BatchNorm2d(ndf * 8),
                                       nn.LeakyReLU(0.2, inplace=True))
            self.fc_out_img = blocks.LinearUnconditionalLogits(ndf * 8*4*4) #nn.Sequential(nn.Conv2d(ndf * 8, 1, 1, stride=1, bias=False), nn.LeakyReLU(0.2, inplace=True))


        #inference over seg
        self.conv1z =nn.Sequential(nn.Conv2d(self.local_nlabels, ndf * 4, 1, 1, padding = 0, bias = False), nn.LeakyReLU(0.2, inplace=True))
        self.conv2z = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 4, 1, 1, padding=0, bias=False), nn.LeakyReLU(0.2, inplace=True))

        # Joint inference
        self.conv1xz = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 1, stride=1, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.conv2xz = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 1, stride=1, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.conv3xz = nn.Sequential(nn.Conv2d(ndf * 8, 1, 1, stride=1, bias=False), nn.LeakyReLU(0.2, inplace=True))
        #comment fc_out_joint to ouput a map of 8x8 to compute bin crossentropy on that :
        self.fc_out_joint = blocks.LinearUnconditionalLogits(8*8)
        # self.conv4xz = nn.Sequential(nn.Conv2d(1, 1, 4, 2,1, bias=False), nn.LeakyReLU(0.2, inplace=True))

    def inf_x(self, img):
        out = self.conv1(img)   #to try : with dropout as in initial bigan model
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out

    def inf_seg(self,seg):
        out = self.conv1z(seg)
        out = self.conv2z(out)
        return out

    def inf_xseg(self,xseg):
        out = self.conv1xz(xseg)
        forQdisc = self.conv2xz(out)
        out = self.conv3xz(forQdisc)
        out = out.view(out.size(0), -1)
        out = self.fc_out_joint(out)
        # out = self.conv4xz(out)
        return forQdisc, out


    def forward(self, input, seg):
        inputbis = self.inf_x(input)
        seg = self.inf_seg(seg)

        xseg = torch.cat((inputbis, seg), dim = 1)
        forQdisc, xseg = self.inf_xseg(xseg)

        if not self.qhead_withImg:
            forQdisc = forQdisc
        else:
            forQdisc = inputbis

        if not self.img_path:
            return forQdisc, xseg
        else:
            input = self.conv6(inputbis)
            input = self.conv7(input)
            input = input.view(input.size(0),-1)
            input = self.fc_out_img(input)
            # print("img path", input.size(), xseg.size())
            return forQdisc, xseg, input






class BiGANQHeadDiscriminator(nn.Module):
    def __init__(self,
                 nlabels,
                 local_nlabels=None,
                 z_dim=1,
                 features='penultimate',
                 pack_size=1,
                 qhead_withImg=False,
                 nc=3,
                 ndf=64,
                 size=0,
                 **kwargs):
        super(BiGANQHeadDiscriminator, self).__init__()
        print("z_dim in qhead disc :", z_dim)
        self.ndf = ndf
        self.nlabels = nlabels
        self.local_nlabels = local_nlabels
        input_nc = 3
        if qhead_withImg:
            self.conv1 = nn.Conv2d(ndf *4, 256, 8, bias=False)
        else:
            self.conv1 = nn.Conv2d(ndf * 8, 256, 8, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv_mu = nn.Conv2d(256, z_dim, 1)
        self.conv_var = nn.Conv2d(256, z_dim, 1)



    def forward(self,x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)
        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())
        # print("size of mu and var", mu.size(), var.size(), torch.min(mu), torch.max(mu))
        return mu, var

class LocalAndGlobalDiscriminator(nn.Module):
    def __init__(self,
                 nlabels,
                 local_nlabels=None,
                 features='penultimate',
                 pack_size=1,
                 nc=3,
                 ndf=64,
                 size=0,
                 **kwargs):
        super(LocalAndGlobalDiscriminator, self).__init__()
        conditioning = 'unconditional'
        # assert conditioning != 'unconditional' or nlabels == 1
        print("Label discriminator is ", conditioning)

        self.ndf = ndf
        self.nlabels = nlabels
        self.local_nlabels=local_nlabels

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
            self.local_FeatureMapping = blocks.local_FeatureMapping(local_nlabels, ndf * 4) #after conv5
            self.fc_out = blocks.LinearUnconditionalLogits(ndf * 4 * self.final_res * self.final_res)
        else:
            raise NotImplementedError(
                f"{conditioning} not implemented for discriminator")

        self.pack_size = pack_size
        self.features = features
        print(f'Getting features from {self.features}')

    def select_mapDisc(self,out, seg, y=None):

        # print("in select map Disc : ", out.size(), seg.size())
        seg = torch.argmax(seg, dim=1)   #convert from one hot encoding to one channel tensor

        seg = torch.unsqueeze(seg, dim=1)
        return torch.gather(out,1,seg)

    def forward(self, input, seg=None, y=None):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        ftmap = self.local_FeatureMapping(out)
        scoreLabelMap = self.select_mapDisc(ftmap, seg)
        out = self.conv6(out)

        out = out.view(out.size(0), -1)

        result = self.fc_out(out)
        assert (len(result.shape) == 1)
        return result, scoreLabelMap
       

if __name__ == '__main__':
    z = torch.zeros((1, 128))
    g = Decoder()
    x = torch.zeros((1, 3, 32, 32))
    d = Encoder()

    g(z)
    d(g(z))
    d(x)
