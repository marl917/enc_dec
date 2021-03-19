from gan_training.models import (resnet2, cifar_BiGAN, resnet, resnet_lsun, resnet_lsun256)

decoder_dict = {
    'cifar_bigan': cifar_BiGAN.Decoder,
    'cifar_bigan_sean': cifar_BiGAN.SEANDecoder,
    'resnet' : resnet.Decoder,
    'resnet_lsun': resnet_lsun.Decoder,
    'resnet_lsun256': resnet_lsun256.Decoder
}

encoder_dict = {
    'cifar_bigan':cifar_BiGAN.Encoder,
    'cifar_bigan_munit': cifar_BiGAN.MUNITEncoder,
    'resnet': resnet.Encoder,
    'resnet_lsun': resnet_lsun.Encoder,
    'resnet_lsun256': resnet_lsun256.Encoder,
    'resnet_lsun_munit': resnet_lsun.MunitEncoder
}


discriminator_dict = {
    'cifar_bigan': cifar_BiGAN.BiGANDiscriminator,
    'resnet': resnet.BiGANDiscriminator,
    'resnet_lsun': resnet_lsun.BiGANDiscriminator,
    'resnet_lsun256': resnet_lsun256.BiGANDiscriminator,
    'resnet_lsun_small': resnet_lsun.smallBiGANDiscriminator
}

label_generator_dict = {
    'cifar_bigan': cifar_BiGAN.LabelGenerator,
    'resnet': resnet.LabelGenerator,
    'resnet_lsun': resnet_lsun.LabelGenerator,
'resnet_lsun256': resnet_lsun256.LabelGenerator
}

qhead_discriminator_dict = {
    'cifar_bigan': cifar_BiGAN.BiGANQHeadDiscriminator,
    'resnet': resnet.BiGANQHeadDiscriminator,
    'resnet_lsun': resnet_lsun.BiGANQHeadDiscriminator,
'resnet_lsun256': resnet_lsun256.BiGANQHeadDiscriminator
}
