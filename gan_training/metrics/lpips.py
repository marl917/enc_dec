

def compute_fid_from_npz(path):
    print(path)
    with np.load(path) as data:
        fake_imgs = data['fake']

        name_dat = None
        for name in ['imagenet', 'cifar', 'places', 'lsun']:
            if name in path:
                name_dat = name
                break
        print('Inferred name', name_dat)

    n_images = fake_imgs.shape[0]
    if name_dat == "lsun":
        real_data = datasets.LSUN(root='data/lsun/train',
                             classes=['church_outdoor_train'],
                             transform=get_transform(sizes[dataset]))


        for x, y in tqdm(real_data):
            utils.save
            os.system(f'CUDA_VISIBLE_DEVICES={device} python gan_training/metrics/lpips.py {arguments}')


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser('compute LPIPS similarity')
    parser.add_argument('--samples', help='path to samples')
    parser.add_argument('--it', type=str, help='path to samples')
    parser.add_argument('--results_dir', help='path to results_dir')
    args = parser.parse_args()

    it = args.it
    results_dir = args.results_dir

    mean = compute_fid_from_npz(args.samples)
    print(f'FID: {mean}')

    if args.results_dir is not None:
        with open(os.path.join(args.results_dir, 'fid_results.json')) as f:
            fid_results = json.load(f)

        fid_results[it] = mean
        print(f'{results_dir} iteration {it} FID: {mean}')

        with open(os.path.join(args.results_dir, 'fid_results.json'), 'w') as f:
            f.write(json.dumps(fid_results))