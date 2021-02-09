N=10 #nb of fake images to comparer lpips metric with images from dataset
from gan_training import utils
from os import path

def compute_lpips_from_npz(path, results_dir):
    with np.load(path) as data:
        fake_imgs = data['fake']

        for name in ['imagenet', 'cifar', 'places', 'lsun']:
            if name in path:
                name_imgs = name
                break
        print('Inferred name', name_imgs)

    data = datasets.LSUN(root='data/lsun/train',
                                       classes=['church_outdoor_train'],
                                       transform=get_transform(sizes[dataset]))
    fake_imgs_dir = path.joint(results_dir, 'fake_imgs')
    if not path.exists(fake_imgs_dir):
        os.makedirs(fake_imgs_dir)
    for i in range(N):
        utils.save_images(fake_imgs[i], path.join(results_dir, '%d.png'%(i)))

    real_oneimg_dir = path.join(results_dir, 'real.png')
    for x, y in tqdm(data):
        utils.save_images(x, real_img_dir)
        for i in range(N):
            fake_oneimg_dir =  path.join(results_dir, '%d.png'%(i))
            os.system(f'CUDA_VISIBLE_DEVICES={device} python lpips_2imgs.py -p0 {real_oneimg_dir} -p1 {fake_oneimg_dir} --use_gpu')


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser('compute lpips')
    parser.add_argument('--samples', help='path to samples')
    parser.add_argument('--it', type=str, help='path to samples')
    parser.add_argument('--results_dir', help='path to results_dir')
    args = parser.parse_args()

    it = args.it
    results_dir = args.results_dir


    mean = compute_fid_from_npz(args.samples, args.results_dir)
    print(f'FID: {mean}')

    if args.results_dir is not None:
        with open(os.path.join(args.results_dir, 'fid_results.json')) as f:
            fid_results = json.load(f)

        fid_results[it] = mean
        print(f'{results_dir} iteration {it} FID: {mean}')

        with open(os.path.join(args.results_dir, 'fid_results.json'), 'w') as f:
            f.write(json.dumps(fid_results))