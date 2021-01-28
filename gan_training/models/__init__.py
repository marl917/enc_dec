from gan_training.models import (resnet2, dcgan_label_gen, dcgan_deep_onlylocal_clust)

decoder_dict = {
    'dcgan_deep_onlylocal_clust': dcgan_deep_onlylocal_clust.Decoder
}

encoder_dict = {
    'dcgan_deep_onlylocal_clust':dcgan_deep_onlylocal_clust.Encoder
}


discriminator_dict = {
    'dcgan_deep_onlylocal_clust_bigan': dcgan_deep_onlylocal_clust.BiGANDiscriminator,
    'dcgan_deep_onlylocal_clust_LocalGlob': dcgan_deep_onlylocal_clust.LocalAndGlobalDiscriminator
}

label_generator_dict = {
    'dcgan_label_gen': dcgan_label_gen.Generator
}

qhead_discriminator_dict = {
    'dcgan_deep_onlylocal_clust_bigan': dcgan_deep_onlylocal_clust.BiGANQHeadDiscriminator
}

label_discriminator_dict = {
    'dcgan_label_gen': dcgan_label_gen.LabDiscriminator
}
