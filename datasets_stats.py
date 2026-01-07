import numpy as np
np.bool=np.bool_
import os
import argparse
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="hovernext",
                    choices=["cellvit", "hovernext", "acs"], help='model')
parser.add_argument('--batch_size', type=int, default=5, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=256, help='img size of per batch')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--variant', type=str, default='1', help='fusion variant')
parser.add_argument('--iter', type=int, default=300, help='iteration')
parser.add_argument('--dataset_path', type=str, default='1', help='path to dataset')
parser.add_argument('--dataset_name', type=str, default='1', help='dataset name')
args = parser.parse_args()


def main(args):
    ## load data
    data_pth = args.dataset_path #'/home/ntorbati/STORAGE/NucleiAnalysis/tif/1.kumar/train/'
    images = glob.glob(f"{data_pth}/**/*.tif")
    masks = glob.glob(f"{data_pth}/**/*.npy")
    if len(images) < len(masks) - 4:
        print('wt...')
    # images = [im for im in os.listdir(data_pth) if im.endswith('.tif')]
    mask_data = []
    for im in images:
        msk = np.load(im.replace('.tif', '.npy'))
        mask_data.append(msk)

    number_images = len(images)
    num_inst = 0
    min_size = 1e10
    max_size = 0
    total_size = 0
    for msk in mask_data:
        if msk.shape[0] * msk.shape[1] < min_size:
            min_res = (msk.shape[0], msk.shape[1])
            min_size = msk.shape[0] * msk.shape[1]
        if msk.shape[0] * msk.shape[1] > max_size:
            max_res = (msk.shape[0], msk.shape[1])
            max_size = msk.shape[0] * msk.shape[1]
        num_inst += (len(np.unique(msk)) - 1)
        total_size += (msk.shape[0] * msk.shape[1])/(256*256)

    return [number_images, num_inst, min_res, max_res,num_inst/total_size]




if __name__ == "__main__":
    dataset_tags = ['pan-cancer-nuclei-seg', 'monuseg', 'cpm17', 'tnbc',
                    'cryonuseg', 'nuinsseg', 'consep', 'puma', 'monusac', 'data science bowl']
    data_path = '/home/ntorbati/STORAGE/NucleiAnalysis/tif/'
    logs_dir = '/home/ntorbati/PycharmProjects/NucleiAnalysis/Models/CellVit/CellViT/logs_paper/logs'
    datasets_names = os.listdir(data_path)
    dict = {}
    for dataset_tag in dataset_tags:
        for dataset_name in datasets_names:
            if dataset_tag in dataset_name:
                print(dataset_name)
                args.dataset_name = dataset_name
                args.dataset_path = os.path.join(data_path,dataset_name)
                stat = main(args)
                dict[dataset_tag] = stat
                print(dict[dataset_tag])
    print(dict)

