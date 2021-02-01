from gan_training.models import (resnet2, dcgan_label_gen, dcgan_deep_onlylocal_clust, resnet, resnet_lsun)

decoder_dict = {
    'dcgan_deep_onlylocal_clust': dcgan_deep_onlylocal_clust.Decoder,
    'resnet' : resnet.Decoder,
    'resnet_lsun': resnet_lsun.Decoder
}

encoder_dict = {
    'dcgan_deep_onlylocal_clust':dcgan_deep_onlylocal_clust.Encoder,
    'resnet': resnet.Encoder,
    'resnet_lsun': resnet_lsun.Encoder
}


discriminator_dict = {
    'dcgan_deep_onlylocal_clust_bigan': dcgan_deep_onlylocal_clust.BiGANDiscriminator,
    'dcgan_deep_onlylocal_clust_LocalGlob': dcgan_deep_onlylocal_clust.LocalAndGlobalDiscriminator,
    'resnet': resnet.BiGANDiscriminator,
    'resnet_lsun': resnet_lsun.BiGANDiscriminator
}

label_generator_dict = {
    'dcgan_label_gen': dcgan_label_gen.Generator,
    'resnet': resnet.LabelGenerator,
    'resnet_lsun': resnet_lsun.LabelGenerator
}

qhead_discriminator_dict = {
    'dcgan_deep_onlylocal_clust_bigan': dcgan_deep_onlylocal_clust.BiGANQHeadDiscriminator,
    'resnet': resnet.BiGANQHeadDiscriminator,
    'resnet_lsun': resnet_lsun.BiGANQHeadDiscriminator
}

label_discriminator_dict = {
    'dcgan_label_gen': dcgan_label_gen.LabDiscriminator
}
