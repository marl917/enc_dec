from gan_training.models import (resnet2, dcgan_label_gen, dcgan_deep_onlylocal_clust, resnet, resnet_lsun, resnet_lsun256)

decoder_dict = {
    'dcgan_deep_onlylocal_clust': dcgan_deep_onlylocal_clust.Decoder,
    'resnet' : resnet.Decoder,
    'resnet_lsun': resnet_lsun.Decoder,
    'resnet_lsun256': resnet_lsun256.Decoder
}

encoder_dict = {
    'dcgan_deep_onlylocal_clust':dcgan_deep_onlylocal_clust.Encoder,
    'resnet': resnet.Encoder,
    'resnet_lsun': resnet_lsun.Encoder,
    'resnet_lsun256': resnet_lsun256.Encoder,
    'resnet_lsun_munit': resnet_lsun.MunitEncoder
}


discriminator_dict = {
    'dcgan_deep_onlylocal_clust_bigan': dcgan_deep_onlylocal_clust.BiGANDiscriminator,
    'dcgan_deep_onlylocal_clust_LocalGlob': dcgan_deep_onlylocal_clust.LocalAndGlobalDiscriminator,
    'resnet': resnet.BiGANDiscriminator,
    'resnet_lsun': resnet_lsun.BiGANDiscriminator,
    'resnet_lsun256': resnet_lsun256.BiGANDiscriminator,
    'resnet_lsun_small': resnet_lsun.smallBiGANDiscriminator
}

label_generator_dict = {
    'dcgan_label_gen': dcgan_label_gen.Generator,
    'resnet': resnet.LabelGenerator,
    'resnet_lsun': resnet_lsun.LabelGenerator,
'resnet_lsun256': resnet_lsun256.LabelGenerator
}

qhead_discriminator_dict = {
    'dcgan_deep_onlylocal_clust_bigan': dcgan_deep_onlylocal_clust.BiGANQHeadDiscriminator,
    'resnet': resnet.BiGANQHeadDiscriminator,
    'resnet_lsun': resnet_lsun.BiGANQHeadDiscriminator,
'resnet_lsun256': resnet_lsun256.BiGANQHeadDiscriminator
}

label_discriminator_dict = {
    'dcgan_label_gen': dcgan_label_gen.LabDiscriminator
}
