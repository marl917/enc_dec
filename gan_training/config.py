import yaml
from torch import optim
from os import path
from gan_training.models import decoder_dict, encoder_dict, discriminator_dict, label_discriminator_dict, qhead_discriminator_dict, label_generator_dict
from gan_training.train import toggle_grad



# General config
def load_config(path, default_path):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        # Add item if not yet in dict1
        if k not in dict1:
            dict1[k] = None
        # Update
        if isinstance(dict1[k], dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v



def build_models(config):
    # Get classes
    Decoder = decoder_dict[config['decoder']['name']]
    Encoder = encoder_dict[config['encoder']['name']]

    # Build models
    decoder = Decoder(z_dim=config['z_dist']['dim'],
                      nlabels=config['decoder']['nlabels'],
                      deterministicOnSeg = config['decoder']['deterministicOnSeg'],
                      local_nlabels=config['decoder']['n_locallabels'],
                      size=config['data']['img_size'],
                      **config['decoder']['kwargs'])

    encoder = Encoder(
        nlabels=config['encoder']['nlabels'],
        local_nlabels=config['encoder']['n_locallabels'],
        img_size=config['data']['img_size'],
        label_size = config['label_generator']['label_size'],
        **config['encoder']['kwargs'])

    discriminator = None
    qhead_discriminator = None

    Discriminator = discriminator_dict[config['discriminator']['name']]
    discriminator = Discriminator(z_dim=config['z_dist']['dim'],
                      nlabels=config['discriminator']['nlabels'],
                      local_nlabels=config['discriminator']['n_locallabels'],
                      img_size=config['data']['img_size'],
                      label_size=config['label_generator']['label_size'],
                      **config['discriminator']['kwargs'])

    Label_generator = label_generator_dict[config['label_generator']['name']]
    label_generator = Label_generator(z_dim=config['label_generator']['zdim'],
                      nlabels=config['label_generator']['nlabels'],
                      local_nlabels = config['label_generator']['n_locallabels'],
                      label_size=config['label_generator']['label_size'],
                      conditioning = config['label_generator']['conditioning'],
                      **config['label_generator']['kwargs'])


    Qhead_discriminator = qhead_discriminator_dict[config['discriminator']['name']]
    qhead_discriminator = Qhead_discriminator(z_dim_lab=config['label_generator']['zdim'],
                                              z_dim_img = config['z_dist']['dim'],
                  nlabels=config['discriminator']['nlabels'],
                  local_nlabels=config['discriminator']['n_locallabels'],
                  size=config['label_generator']['label_size'],
                  qhead_variant = not config['decoder']['deterministicOnSeg'],
                  **config['discriminator']['kwargs'])

    return decoder, encoder, discriminator, label_generator, qhead_discriminator



def build_optimizers(decoder, encoder, discriminator, label_generator, config, qhead_disc=None):
    optimizer = config['training']['optimizer']
    lr_g = config['training']['lr_g']
    lr_d = config['training']['lr_d']
    

    toggle_grad(decoder, True)
    toggle_grad(encoder, True)

    dec_params = decoder.parameters()
    enc_params = encoder.parameters()
    disc_params = discriminator.parameters()
    labelGen_params = label_generator.parameters()
    if qhead_disc!=None:
        qhead_params = qhead_disc.parameters()

    if optimizer == 'rmsprop':
        g_optimizer = optim.RMSprop(g_params, lr=lr_g, alpha=0.99, eps=1e-8)
        d_optimizer = optim.RMSprop(d_params, lr=lr_d, alpha=0.99, eps=1e-8)
    elif optimizer == 'adam':
        beta1 = config['training']['beta1']
        beta2 = config['training']['beta2']
        if qhead_disc!=None:
            dec_optimizer = optim.Adam(list(dec_params) + list(qhead_params), lr=lr_g, betas=(beta1, beta2), eps=1e-8)
        else:
            dec_optimizer = optim.Adam(dec_params, lr=lr_g,
                                       betas=(beta1, beta2), eps=1e-8)
        label_gen_optimizer = optim.Adam(labelGen_params, lr = lr_g, betas = (beta1, beta2), eps = 1e-8)
        enc_optimizer = optim.Adam(enc_params, lr=lr_d, betas=(beta1, beta2), eps=1e-8)
        disc_optimizer = optim.Adam(disc_params, lr=lr_d, betas=(beta1, beta2), eps=1e-8)
    elif optimizer == 'sgd':
        g_optimizer = optim.SGD(g_params, lr=lr_g, momentum=0.)
        d_optimizer = optim.SGD(d_params, lr=lr_d, momentum=0.)

    return dec_optimizer, enc_optimizer, disc_optimizer, label_gen_optimizer


# Some utility functions
def get_parameter_groups(parameters, gradient_scales, base_lr):
    param_groups = []
    for p in parameters:
        c = gradient_scales.get(p, 1.)
        param_groups.append({'params': [p], 'lr': c * base_lr})
    return param_groups
