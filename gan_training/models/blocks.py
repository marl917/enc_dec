import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, local_nlabels, detOnSeg):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        # if 'spectral' in opt.norm_G:
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = SPADE(fin, local_nlabels, detOnSeg)
        self.norm_1 = SPADE( fmiddle, local_nlabels,detOnSeg)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, local_nlabels,detOnSeg)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))
        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc,detOnSeg):
        super().__init__()
        ks = 3
        if detOnSeg:
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        else:
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear', align_corners=True)
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out

class ResnetBlock(nn.Module):
    def __init__(self,
                 fin,
                 fout,
                 bn,
                 use_shortcut=False,
                 fhidden=None,
                 is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout) or use_shortcut
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden
        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden,
                      self.fout,
                      3,
                      stride=1,
                      padding=1,
                      bias=is_bias)
        # self.conv_1 = spectral_norm(self.conv_1)
        # self.conv_0 = spectral_norm(self.conv_0)

        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin,
                          self.fout,
                          1,
                          stride=1,
                          padding=0,
                          bias=False)
            # self.conv_s = spectral_norm(self.conv_s)
        # if not bn:
        #     self.bn0 = Identity(self.fin)
        #     self.bn1 = Identity(self.fhidden)
        # else:
        print(bn)
        self.bn0 = bn(self.fin)
        self.bn1 = bn(self.fhidden)


    def forward(self, x):
        x_s = self._shortcut(x)
        # dx = self.conv_0(actvn(self.bn0(x)))
        # dx = self.conv_1(actvn(self.bn1(dx)))
        dx = self.conv_0(actvn(self.bn0(x)))
        dx = self.conv_1(actvn(self.bn1(dx)))
        # out = x_s + 0.1 * dx
        out = x_s +0.1*dx
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out


class LatentEmbeddingConcat(nn.Module):
    ''' projects class embedding onto hypersphere and returns the concat of the latent and the class embedding '''

    def __init__(self, nlabels, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(nlabels, embed_dim)

    def forward(self, z, y):
        assert (y.size(0) == z.size(0))
        yembed = self.embedding(y)
        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)
        yz = torch.cat([z, yembed], dim=1)
        return yz


class NormalizeLinear(nn.Module):
    def __init__(self, act_dim, k_value):
        super().__init__()
        self.lin = nn.Linear(act_dim, k_value)

    def normalize(self):
        self.lin.weight.data = F.normalize(self.lin.weight.data, p=2, dim=1)

    def forward(self, x):
        self.normalize()
        return self.lin(x)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inp, *args, **kwargs):
        return inp


# class Classifier_Module(nn.Module):
#
#     def __init__(self, num_classes, n_channels, dilation_series=[6,12,18,24], padding_series=[6,12,18,24]):
#         super(Classifier_Module, self).__init__()
#         self.conv2d_list = nn.ModuleList()
#         for dilation, padding in zip(dilation_series, padding_series):
#             self.conv2d_list.append(nn.Conv2d(n_channels, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))
#
#         for m in self.conv2d_list:
#             m.weight.data.normal_(0, 0.01)
#
#     def forward(self, x):
#         out = self.conv2d_list[0](x)
#         for i in range(len(self.conv2d_list)-1):
#             out += self.conv2d_list[i+1](x)
#         return out


class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes, n_input_channels):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(
                n_input_channels, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

class local_FeatureMapping(nn.Module):

    def __init__(self, num_classes, n_channels):
        super(local_FeatureMapping, self).__init__()

        # self.conv1 = nn.Sequential(nn.Conv2d(n_channels,512, 3, 1, 1), nn.LeakyReLU(0.1), nn.BatchNorm2d(512))
        self.conv1 = nn.Sequential(nn.Conv2d(n_channels, 512, 3, 1, 1), nn.LeakyReLU(0.1))
        self.conv2 = nn.Sequential(nn.Conv2d(512, 512, 1, 1, 0), nn.LeakyReLU(0.1))  #1 x 1 conv
        # self.conv3 = nn.Conv2d(512, num_classes, 3, 1, 1)
        self.conv3 = nn.Sequential(nn.Conv2d(512, num_classes, 3, 1, 1), nn.LeakyReLU(0.1))

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out


class LinearConditionalMaskLogits(nn.Module):
    ''' runs activated logits through fc and masks out the appropriate discriminator score according to class number'''

    def __init__(self, nc, nlabels):
        super().__init__()
        self.fc = nn.Linear(nc, nlabels)

    def forward(self, inp, y=None, take_best=False, get_features=False):
        out = self.fc(inp)
        if get_features: return out

        if not take_best:
            y = y.view(-1)
            index = Variable(torch.LongTensor(range(out.size(0))))
            if y.is_cuda:
                index = index.cuda()
            return out[index, y]
        else:
            # high activation means real, so take the highest activations
            best_logits, _ = out.max(dim=1)
            return best_logits


class ProjectionDiscriminatorLogits(nn.Module):
    ''' takes in activated flattened logits before last linear layer and implements https://arxiv.org/pdf/1802.05637.pdf '''

    def __init__(self, nc, nlabels):
        super().__init__()
        self.fc = nn.Linear(nc, 1)
        self.embedding = nn.Embedding(nlabels, nc)
        self.nlabels = nlabels

    def forward(self, x, y, take_best=False):
        output = self.fc(x)

        if not take_best:
            label_info = torch.sum(self.embedding(y) * x, dim=1, keepdim=True)
            return (output + label_info).view(x.size(0))
        else:
            #TODO: this may be computationally expensive, maybe we want to do the global pooling first to reduce x's size
            index = torch.LongTensor(range(self.nlabels)).cuda()
            labels = index.repeat((x.size(0), ))
            x = x.repeat_interleave(self.nlabels, dim=0)
            label_info = torch.sum(self.embedding(labels) * x,
                                   dim=1,
                                   keepdim=True).view(output.size(0),
                                                      self.nlabels)
            # high activation means real, so take the highest activations
            best_logits, _ = label_info.max(dim=1)
            return output.view(output.size(0)) + best_logits


class LinearUnconditionalLogits(nn.Module):
    ''' standard discriminator logit layer '''

    def __init__(self, nc):
        super().__init__()
        self.fc = nn.Linear(nc, 1)

    def forward(self, inp, take_best=False):
        assert (take_best == False)

        out = self.fc(inp)
        return out.view(out.size(0))


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(*((batch_size, ) + self.shape))


class ConditionalBatchNorm2d(nn.Module):
    ''' from https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775 '''

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(
            1, 0.02)  # Initialize scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_(
        )  # Initialize bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
            -1, self.num_features, 1, 1)
        return out


class BatchNorm2d(nn.Module):
    ''' identical to nn.BatchNorm2d but takes in y input that is ignored '''

    def __init__(self, nc, nchannels=None, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm2d(nc)

    def forward(self, x, y=None):
        return self.bn(x)
